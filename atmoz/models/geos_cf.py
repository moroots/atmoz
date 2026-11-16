"""NASA GEOS-CF model output at a point, via OPeNDAP.

Follows the package conventions (``docs/conventions.md``): ``download()`` returns an ``AtmozDataset``,
``fetch()`` returns ``{name: Path}`` (cached parquet files), results are cached and downloads are all-or-nothing.

Data access: https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/

>>> ds = GEOS_CF.download(39.05, -76.87, "2025-07-28", "2025-08-01", variables=["o3"])
>>> ds["o3"]                                  # AtmozFrame: index t (UTC), columns value (ppb), lat, lon
>>> prof = GEOS_CF.download(39.05, -76.87, "2025-07-28", "2025-08-01", collection="chm", variables=["o3"])
>>> prof["o3"]                                # long: index t, columns value (ppb), lat, lon, alt (km), lev
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atmoz.dataclasses import AtmozDataset, AtmozFrame
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.utilities import PathLike, run_jobs

# --------------------------------------------------------------------------------------------------------------------------------- #
# OPeNDAP collection catalogue
# --------------------------------------------------------------------------------------------------------------------------------- #
_OPENDAP_BASE = "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf"
_GRID_DEG = 0.25   # g1440x721: grid points sit on multiples of 0.25 degrees

_COLLECTION_CATALOG: Dict[str, dict] = {
    "aqc": {
        "urls": {
            "assim": f"{_OPENDAP_BASE}/assim/aqc_tavg_1hr_g1440x721_v1",
            "fcast": f"{_OPENDAP_BASE}/fcast/aqc_tavg_1hr_g1440x721_v1.latest",   # the rolling 5-day forecast
        },
        "variables": ["o3", "no2", "co", "so2", "pm25_rh35_gcc"],
        "description": "Surface air quality, 1-hr average (2018 to present)",
    },
    "chm": {
        "urls": {"assim": f"{_OPENDAP_BASE}/assim/chm_tavg_1hr_g1440x721_v36"},
        "variables": ["o3", "no2", "co", "so2", "hno3"],
        "description": "Chemistry profiles on the lowest 36 model levels (37 = highest ... 72 = surface), 1-hr average (2020 to present); 50+ species",
    },
    "chm_sfc": {
        "urls": {"assim": f"{_OPENDAP_BASE}/assim/chm_tavg_1hr_g1440x721_v1"},
        "variables": ["o3", "no2", "co", "so2", "hno3"],
        "description": "Surface chemistry, all 50+ species, 1-hr average (2018 to present)",
    },
    "met": {
        "urls": {"assim": f"{_OPENDAP_BASE}/assim/met_tavg_1hr_g1440x721_x1"},
        "variables": ["t2m", "u10m", "v10m", "q2m", "zpbl", "ps"],
        "description": "Surface meteorology, 1-hr average (2018 to present)",
    },
    "met3d": {
        "urls": {"assim": f"{_OPENDAP_BASE}/assim/met_tavg_1hr_g1440x721_v36"},
        "variables": ["zl"],
        "description": "Meteorology on the same 36 model levels as `chm` (zl = layer height in m, t, u, v, q, rh, ...), 1-hr average (2020 to present)",
    },
}

# Gas mixing ratios are served as mol/mol; ``units="ppbv"`` (the default) multiplies these by 1e9. Anything else is returned as served.
_GASES = {"o3", "no2", "no", "co", "so2", "hno3", "nh3", "hcho", "ch4", "isop", "pan", "h2o2"}
_NATIVE_UNITS = {"pm25_rh35_gcc": "µg/m3", "t2m": "K", "u10m": "m/s", "v10m": "m/s", "q2m": "kg/kg", "zpbl": "m", "ps": "Pa",
                 "zl": "m", "t": "K", "u": "m/s", "v": "m/s", "q": "kg/kg", "rh": "1", **{g: "mol/mol" for g in _GASES}}

# Collections on model levels: long profiles whose altitude comes from the layer height ``zl`` of ``met3d`` (same 36 levels)
_LEVEL_COLLECTIONS = ("chm", "met3d")
_HEIGHT_JOB = {"collection": "met3d", "mode": "assim", "variables": ["zl"]}

# the HDF5 / netCDF-C libraries underneath xarray are not thread-safe: talk to the server one request at a time
_netcdf_lock = threading.Lock()

# The NASA server often answers 502 / "DAP server error" or returns a damaged time axis: retry with a growing pause.
_RETRIES = 4
_RETRY_DELAY = 3.0   # seconds before the first retry (doubles each time)

_FINAL_AFTER = timedelta(days=14)    # assimilation chunks this old are final (cached forever)
_RECENT_MAX_AGE = timedelta(hours=3)  # newer chunks, and forecasts, are re-downloaded after this long


# --------------------------------------------------------------------------------------------------------------------------------- #
# Query model
# --------------------------------------------------------------------------------------------------------------------------------- #
class GEOS_CF_QUERY(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    start_date: str
    end_date: str
    collection: Literal["aqc", "chm", "chm_sfc", "met", "met3d"] = Field(default="aqc")
    mode: Literal["assim", "fcast"] = Field(default="assim")
    variables: Optional[List[str]] = Field(default=None)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v) -> str:
        """Accept YYYY-MM-DD, YYYYMMDD, date or datetime; store as YYYY-MM-DD."""
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        v = str(v).strip()
        if len(v) == 10 and "-" in v:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        if len(v) == 8:
            return datetime.strptime(v, "%Y%m%d").strftime("%Y-%m-%d")
        raise ValueError(f"'{v}' must be YYYY-MM-DD or YYYYMMDD.")

    @model_validator(mode="after")
    def _check(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        entry = _COLLECTION_CATALOG[self.collection]
        if self.mode not in entry["urls"]:
            raise ValueError(f"Mode '{self.mode}' is not available for collection '{self.collection}'. "
                             f"Available modes: {list(entry['urls'])}.")
        self.variables = [v.lower() for v in (self.variables or entry["variables"])]
        return self


def _monthly_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    """``(chunk_start, chunk_end)`` pairs covering [start, end] in calendar-month steps (servers cap time slices)."""
    chunks = []
    for period in pd.period_range(start=start, end=end, freq="M"):
        chunks.append((max(pd.Timestamp(start), period.start_time).strftime("%Y-%m-%d"),
                       min(pd.Timestamp(end), period.end_time).strftime("%Y-%m-%d")))
    return chunks


_GRADS_START = re.compile(r"(\d{2})(?::(\d{2}))?z(\d{2})([a-z]{3})(\d{4})", re.I)
_GRADS_STEP = re.compile(r"(\d+)\s*(mn|hr|dy)", re.I)
_STEP_MINUTES = {"mn": 1, "hr": 60, "dy": 1440}


def _decode_time(time: "xr.DataArray") -> pd.DatetimeIndex:
    """The server's GrADS time axis -> UTC-naive timestamps.

    The raw axis is ``days since 1-1-1`` (ambiguous: xarray decodes it into a broken Julian/Gregorian calendar), and the values
    are sometimes damaged in transit (seen live on ``met3d``: out-of-order entries). A GrADS axis is *linear*, so it is rebuilt
    from its own attributes: ``minimum`` (e.g. ``00:30z01jan2018``, the first step), ``grads_step`` (``60mn``) and ``grads_size``.
    The transferred values are only used when those are missing, and are then checked. Already-decoded axes pass through.
    """
    if pd.api.types.is_datetime64_any_dtype(time.dtype):
        return pd.DatetimeIndex(time.values)
    m = _GRADS_START.fullmatch(str(time.attrs.get("minimum", "")).strip())
    if not m:
        raise ValueError(f"Cannot decode the server's time axis (units={time.attrs.get('units')!r}, "
                         f"minimum={time.attrs.get('minimum')!r}).")
    hh, mm, dd, mon, yyyy = m.groups()
    start = pd.Timestamp(f"{yyyy}-{mon.title()}-{dd} {hh}:{mm or '00'}")
    n = time.sizes["time"] if "time" in time.sizes else len(time)

    step = _GRADS_STEP.fullmatch(str(time.attrs.get("grads_step", "")).strip())
    if step and str(time.attrs.get("grads_mapping", "linear")).lower() == "linear" and str(time.attrs.get("grads_size", n)) == str(n):
        minutes = int(step.group(1)) * _STEP_MINUTES[step.group(2).lower()]
        return start + pd.to_timedelta(np.arange(n) * minutes, unit="min")

    days = time.values.astype("float64")
    if not (np.isfinite(days).all() and (np.diff(days) > 0).all() and days[-1] - days[0] < 366 * 60):
        raise OSError("The server returned a damaged time axis (not finite, increasing and within 60 years).")
    return start + pd.to_timedelta(np.round((days - days[0]) * 1440.0), unit="min")


def _snap(value: float) -> float:
    return round(round(value / _GRID_DEG) * _GRID_DEG, 4)


# --------------------------------------------------------------------------------------------------------------------------------- #
# GEOS_CF
# --------------------------------------------------------------------------------------------------------------------------------- #
class GEOS_CF:
    """Single-site GEOS-CF client.

    * :meth:`download` -> :class:`~atmoz.dataclasses.AtmozDataset`
    * :meth:`fetch`    -> ``{name: parquet path}`` (cached tidy tables, one per month of data)

    ``ds`` holds one :class:`~atmoz.dataclasses.AtmozFrame` per variable, keyed by the variable name (``"o3"``). Each has
    time (UTC) as its index and the columns ``value``, ``lat`` and ``lon`` (the grid cell). Collections on model levels
    (``chm``, ``met3d``) are long profiles with an extra ``alt`` column (layer height above the surface, km, from the
    ``zl`` field of ``met3d``) and ``lev`` (the model level). ``chm`` therefore also reads ``met3d`` for the heights.

    Gas species are in **ppb** by default (``units="mol/mol"`` keeps the server's raw units). The point is the
    nearest 0.25-degree grid cell; the cell's coordinates are in ``ds.metadata["lat"]`` / ``["lon"]``.
    """

    catalog = _COLLECTION_CATALOG

    # ---- process: validate -> url -> fetch_one -> parse_one -> assemble ------------------------------------
    @staticmethod
    def _validate(lat, lon, start_date, end_date, collection, mode, variables) -> Tuple[GEOS_CF_QUERY, List[dict]]:
        """Check the request and expand it into one job per calendar month."""
        q = GEOS_CF_QUERY(lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                          collection=collection, mode=mode, variables=variables)
        jobs = [{"collection": q.collection, "mode": q.mode, "variables": q.variables,
                 "lat": _snap(q.lat), "lon": _snap(q.lon), "start": cs, "end": ce}
                for cs, ce in _monthly_chunks(q.start_date, q.end_date)]
        return q, jobs

    @staticmethod
    def _url(job: dict) -> str:
        return _COLLECTION_CATALOG[job["collection"]]["urls"][job["mode"]]

    @staticmethod
    def _name(job: dict) -> str:
        key = hashlib.md5("_".join(sorted(job["variables"])).encode()).hexdigest()[:8]
        return (f"{job['collection']}_{job['mode']}_{job['lat']}_{job['lon']}_"
                f"{job['start'].replace('-', '')}_{job['end'].replace('-', '')}_{key}.parquet")

    @staticmethod
    def _is_fresh(path: Path, job: dict) -> bool:
        """Is the cached file still good? Old assimilation data is final; recent data and forecasts get revised."""
        if not path.exists():
            return False
        final = job["mode"] == "assim" and pd.Timestamp(job["end"]) < pd.Timestamp.now() - _FINAL_AFTER
        return final or (time.time() - path.stat().st_mtime) < _RECENT_MAX_AGE.total_seconds()

    @classmethod
    def _download_chunk(cls, job: dict) -> "xr.Dataset":
        """The server side of one job: nearest grid cell, the job's days, only the requested variables (retried)."""
        delay = _RETRY_DELAY
        for attempt in range(_RETRIES):
            try:
                with _netcdf_lock, xr.open_dataset(cls._url(job), engine="netcdf4", decode_times=False) as ds:
                    site = ds.sel(lat=job["lat"], lon=job["lon"], method="nearest")
                    missing = [v for v in job["variables"] if v not in site.data_vars]
                    if missing:
                        raise ValueError(f"{missing} not in {job['collection']}/{job['mode']}. "
                                         f"Available: {sorted(site.data_vars)}")
                    site = site.assign_coords(time=_decode_time(site["time"]))
                    return site[job["variables"]].sel(time=slice(job["start"], job["end"])).load()
            except (OSError, OverflowError, pd.errors.OutOfBoundsTimedelta, pd.errors.OutOfBoundsDatetime):
                if attempt == _RETRIES - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    @classmethod
    def _fetch_one(cls, job: dict, *, dest_dir: Path, refresh: bool, use_cache: bool) -> Path:
        """One job (one month) -> a tidy parquet file: columns ``time``, [``lev``], ``lat``, ``lon`` and the variables."""
        dest = dest_dir / cls._name(job)
        if use_cache and not refresh and cls._is_fresh(dest, job):
            return dest
        chunk = cls._download_chunk(job)
        df = chunk.to_dataframe().reset_index()
        if df.empty:
            raise ValueError(f"The server returned no data for {job['start']} .. {job['end']}.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(f"{dest.name}.{threading.get_ident()}.part")
        try:
            df.to_parquet(part, index=False)
            part.replace(dest)   # atomic: an interrupted write never looks like a valid cache file
        finally:
            part.unlink(missing_ok=True)
        return dest

    @staticmethod
    def _parse_one(path: PathLike) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    @classmethod
    def _load_one(cls, job: dict, **kw) -> pd.DataFrame:
        return cls._parse_one(cls._fetch_one(job, **kw))

    @staticmethod
    def _assemble(tidy: pd.DataFrame, heights: Optional[pd.DataFrame], q: GEOS_CF_QUERY, units: str,
                  files: List[str]) -> AtmozDataset:
        """All months -> the dataset: one frame per variable (long with ``alt`` and ``lev`` on model levels)."""
        leveled = q.collection in _LEVEL_COLLECTIONS
        tidy = tidy.drop_duplicates().sort_values(["time", "lev"] if leveled else ["time"])
        if leveled:
            zl = heights.drop_duplicates(["time", "lev"]).set_index(["time", "lev"])["zl"]
            tidy["alt"] = zl.reindex(pd.MultiIndex.from_frame(tidy[["time", "lev"]])).to_numpy() / 1000.0
        scale = {v: (1e9 if units == "ppbv" and v in _GASES else 1.0) for v in q.variables}
        unit_of = {v: ("ppbv" if scale[v] != 1.0 else _NATIVE_UNITS.get(v)) for v in q.variables}
        cell = {"lat": float(tidy["lat"].iloc[0]), "lon": float(tidy["lon"].iloc[0])}
        axes = ["lat", "lon"] + (["alt", "lev"] if leveled else [])
        indexed = tidy.set_index("time")

        frames: Dict[str, AtmozFrame] = {}
        for v in q.variables:
            df = indexed[axes].copy()
            df.insert(0, "value", indexed[v] * scale[v])
            frame_units = {"value": unit_of[v], **({"alt": "km"} if leveled else {})}
            frames[v] = AtmozFrame(df, units={c: u for c, u in frame_units.items() if u}, lat="lat", lon="lon",
                                   alt="alt" if leveled else None,
                                   metadata={"variable": v, "collection": q.collection, "mode": q.mode, **cell,
                                             **({"alt_reference": "agl"} if leveled else {})})
        meta = {"endpoint": "geos_cf", "collection": q.collection, "mode": q.mode, "variables": list(q.variables),
                "lat_requested": q.lat, "lon_requested": q.lon, **cell, "start": q.start_date, "end": q.end_date,
                "units": {v: u for v, u in unit_of.items() if u}, "files": files,
                "rows": {k: len(f) for k, f in frames.items()},
                "altitude": "met3d zl (layer height above the surface, km)" if leveled else None}
        return AtmozDataset(name=f"GEOS-CF {q.collection} {q.mode}", source="GEOS-CF", frames=frames, metadata=meta)

    # ---- public ------------------------------------------------------------------------------------------------
    @classmethod
    def download(cls,
                 lat: float,
                 lon: float,
                 start_date: Union[str, date, datetime],
                 end_date: Union[str, date, datetime],
                 *,
                 collection: Literal["aqc", "chm", "chm_sfc", "met", "met3d"] = "aqc",
                 mode: Literal["assim", "fcast"] = "assim",
                 variables: Optional[Sequence[str]] = None,
                 units: Literal["ppbv", "mol/mol"] = "ppbv",
                 max_workers: int = 1,
                 use_cache: bool = True,
                 refresh: bool = False,
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False) -> AtmozDataset:
        """Download (once, then cached) a GEOS-CF point time series and return it as an ``AtmozDataset``.

        Parameters
        ----------
        lat, lon:
            Site coordinates in decimal degrees; the nearest 0.25-degree grid cell is used.
        start_date, end_date:
            Inclusive UTC days: ``YYYY-MM-DD``, ``YYYYMMDD``, ``date`` or ``datetime``.
        collection:
            ``"aqc"`` (surface air quality, default), ``"chm"`` (chemistry profiles, 36 levels), ``"chm_sfc"``, ``"met"`` (surface meteorology) or ``"met3d"``.
            See ``GEOS_CF.catalog``.
        mode:
            ``"assim"`` (replay/historical, default) or ``"fcast"`` (forecast; ``aqc`` only).
        variables:
            Lower-case names (default: all of the collection's, see ``GEOS_CF.catalog``).
        units:
            ``"ppbv"`` (default; stored as ``ppb``) converts gas mixing ratios from the server's mol/mol; ``"mol/mol"`` keeps them raw.
        max_workers:
            Months fetched at once. Server access is serialized anyway (the netCDF libraries are not thread-safe).
        use_cache, refresh, cache_dir:
            File-cache controls. Assimilation data older than two weeks is final; newer data and forecasts are
            re-downloaded after three hours.

        Returns
        -------
        AtmozDataset
            One frame per variable (see the class docs). ``.metadata`` has ``collection``, ``mode``, ``variables``,
            ``lat_requested`` / ``lon_requested``, ``lat`` / ``lon`` (the grid cell), ``units``, ``files`` and the request.

        Raises
        ------
        DownloadError: any month fails (nothing is returned; finished months stay cached).
        """
        q, jobs = cls._validate(lat, lon, start_date, end_date, collection, mode, variables)
        if units not in ("ppbv", "mol/mol"):
            raise ValueError("units must be 'ppbv' or 'mol/mol'")
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("geos_cf")
        height_jobs = [{**j, **_HEIGHT_JOB} for j in jobs] if q.collection in _LEVEL_COLLECTIONS else []
        all_jobs = list({cls._name(j): j for j in jobs + height_jobs}.values())
        loaded = run_jobs(
            lambda job: cls._load_one(job, dest_dir=dest_dir, refresh=refresh, use_cache=use_cache),
            all_jobs, key=cls._name, describe=lambda j: f"{j['collection']}/{j['mode']} {j['start']}..{j['end']}",
            noun="GEOS-CF request", desc=f"Downloading GEOS-CF {q.collection}/{q.mode}",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
        tidy = pd.concat([loaded[cls._name(j)] for j in jobs], ignore_index=True)
        heights = pd.concat([loaded[cls._name(j)] for j in height_jobs], ignore_index=True) if height_jobs else None
        return cls._assemble(tidy, heights, q, units, [cls._name(j) for j in jobs])

    @classmethod
    def fetch(cls,
              lat: float,
              lon: float,
              start_date: Union[str, date, datetime],
              end_date: Union[str, date, datetime],
              *,
              collection: Literal["aqc", "chm", "chm_sfc", "met", "met3d"] = "aqc",
              mode: Literal["assim", "fcast"] = "assim",
              variables: Optional[Sequence[str]] = None,
              max_workers: int = 1,
              refresh: bool = False,
              cache_dir: Optional[PathLike] = None,
              silent: bool = False,
              show_traceback: bool = False) -> Dict[str, Path]:
        """Same request as :meth:`download`, but only fill the cache and return ``{file_name: parquet_path}`` (one per month).

        The files are tidy tables in the server's raw units (``time``, ``lev`` for profiles, ``lat``, ``lon`` and the variables).
        """
        _, jobs = cls._validate(lat, lon, start_date, end_date, collection, mode, variables)
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("geos_cf")
        return run_jobs(
            lambda job: cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh, use_cache=True),
            jobs, key=cls._name, describe=lambda j: f"{j['collection']}/{j['mode']} {j['start']}..{j['end']}",
            noun="GEOS-CF request", desc="Fetching GEOS-CF", max_workers=max_workers,
            silent=silent, show_traceback=show_traceback,
        )


if __name__ == "__main__":
    ds = GEOS_CF.download(39.05, -76.87, "2025-07-28", "2025-08-01", variables=["o3"])
    o3 = ds["o3"]["value"]
    print(f"O3 range: {o3.min():.1f} - {o3.max():.1f} ppb")
