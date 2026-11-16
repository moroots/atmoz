"""NASA GEOS-CF model output at a point, via OPeNDAP.

Follows the package conventions (``docs/conventions.md``): ``download()`` returns an ``atmoz_dataset`` of DataFrames,
``fetch()`` returns ``{name: Path}`` (cached parquet files), results are cached and downloads are all-or-nothing.

Data access: https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/

>>> ds = GEOS_CF.download(39.05, -76.87, "2025-07-28", "2025-08-01", variables=["o3"])
>>> ds.data["aqc_assim"]                      # index: UTC time, column: o3 (ppbv)
>>> prof = GEOS_CF.download(39.05, -76.87, "2025-07-28", "2025-08-01", collection="chm", variables=["o3"])
>>> prof.data["chm_assim_o3"]                 # index: UTC time, columns: model level (the server's `lev` values)
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atmoz.resources.atmoz_dataclasses import atmoz_dataset
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
            "fcast": f"{_OPENDAP_BASE}/fcast/aqc_tavg_1hr_g1440x721_v1",
        },
        "variables": ["o3", "no2", "co", "so2", "pm25_rh35_gcc"],
        "description": "Surface air quality, 1-hr average (2018 to present)",
    },
    "chm": {
        "urls": {"assim": f"{_OPENDAP_BASE}/assim/chm_tavg_1hr_g1440x721_v1"},
        "variables": ["o3", "no2", "co", "so2", "oh", "hno3"],
        "description": "Chemistry profiles, 1-hr average, 72 model levels (2018 to present)",
    },
    "met": {
        "urls": {"assim": f"{_OPENDAP_BASE}/assim/met_tavg_1hr_g1440x721_x1"},
        "variables": ["t2m", "u10m", "v10m", "q2m", "zpbl", "ps"],
        "description": "Surface meteorology, 1-hr average (2018 to present)",
    },
}

# Raw units on the server. Gas mixing ratios are mol/mol; ``units="ppbv"`` (the default) multiplies them by 1e9.
_GASES = {"o3", "no2", "co", "so2", "oh", "hno3"}
_NATIVE_UNITS = {"pm25_rh35_gcc": "ug/m3", "t2m": "K", "u10m": "m/s", "v10m": "m/s", "q2m": "kg/kg", "zpbl": "m", "ps": "Pa",
                 **{g: "mol/mol" for g in _GASES}}

# the HDF5 / netCDF-C libraries underneath xarray are not thread-safe: talk to the server one request at a time
_netcdf_lock = threading.Lock()

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
    collection: Literal["aqc", "chm", "met"] = Field(default="aqc")
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


def _snap(value: float) -> float:
    return round(round(value / _GRID_DEG) * _GRID_DEG, 4)


# --------------------------------------------------------------------------------------------------------------------------------- #
# GEOS_CF
# --------------------------------------------------------------------------------------------------------------------------------- #
class GEOS_CF:
    """Single-site GEOS-CF client.

    * :meth:`download` -> :class:`~atmoz.atmoz_dataset` (``datatype="model"``)
    * :meth:`fetch`    -> ``{name: parquet path}`` (cached tidy tables, one per month of data)

    ``ds.data`` holds, per collection and mode:

    * surface collections (``aqc``, ``met``): ``"{collection}_{mode}"``, a DataFrame indexed by UTC time with one column per variable;
    * profile collection (``chm``): ``"{collection}_{mode}_{variable}"``, indexed by UTC time with one column per model level.

    Gas species are returned in **ppbv** by default (``units="mol/mol"`` keeps the server's raw units). The point is the
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
    def _fetch_one(cls, job: dict, *, dest_dir: Path, refresh: bool, use_cache: bool) -> Path:
        """One job (one month) -> a tidy parquet file: columns ``time``, [``lev``], ``lat``, ``lon`` and the variables."""
        dest = dest_dir / cls._name(job)
        if use_cache and not refresh and cls._is_fresh(dest, job):
            return dest
        with _netcdf_lock:
            with xr.open_dataset(cls._url(job), engine="netcdf4") as ds:
                site = ds.sel(lat=job["lat"], lon=job["lon"], method="nearest")
                missing = [v for v in job["variables"] if v not in site.data_vars]
                if missing:
                    raise ValueError(f"{missing} not in {job['collection']}/{job['mode']}. "
                                     f"Available: {sorted(site.data_vars)}")
                chunk = site[job["variables"]].sel(time=slice(job["start"], job["end"])).load()
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
    def _assemble(tidy: pd.DataFrame, q: GEOS_CF_QUERY, units: str) -> atmoz_dataset:
        """All months -> the dataset (surface: one wide frame; profiles: one time x level frame per variable)."""
        tidy = tidy.drop_duplicates().sort_values(["time", "lev"] if "lev" in tidy else ["time"])
        scale = {v: (1e9 if units == "ppbv" and v in _GASES else 1.0) for v in q.variables}
        unit_of = {v: ("ppbv" if scale[v] != 1.0 else _NATIVE_UNITS.get(v, "unknown")) for v in q.variables}
        for v, f in scale.items():
            if f != 1.0:
                tidy[v] = tidy[v] * f
        base = f"{q.collection}_{q.mode}"
        attrs = {"lat_requested": q.lat, "lon_requested": q.lon, "lat": float(tidy["lat"].iloc[0]),
                 "lon": float(tidy["lon"].iloc[0]), "collection": q.collection, "mode": q.mode}

        data: Dict[str, pd.DataFrame] = {}
        if "lev" in tidy and tidy["lev"].nunique() > 1:                   # profiles
            for v in q.variables:
                frame = tidy.pivot(index="time", columns="lev", values=v)
                frame.columns.name = "lev"
                frame.attrs = {**attrs, "units": unit_of[v], "variable": v}
                data[f"{base}_{v}"] = frame
        else:                                                             # surface
            frame = tidy.drop(columns=[c for c in ("lev", "lat", "lon") if c in tidy]).set_index("time")[q.variables]
            frame.attrs = {**attrs, "units": unit_of}
            data[base] = frame
        meta = {"endpoint": "geos_cf", "files": list(data), "rows": {k: len(f) for k, f in data.items()},
                "start": q.start_date, "end": q.end_date, "units": unit_of, **attrs}
        return atmoz_dataset(data=data, metadata=meta, datatype="model")

    # ---- public ------------------------------------------------------------------------------------------------
    @classmethod
    def download(cls,
                 lat: float,
                 lon: float,
                 start_date: Union[str, date, datetime],
                 end_date: Union[str, date, datetime],
                 *,
                 collection: Literal["aqc", "chm", "met"] = "aqc",
                 mode: Literal["assim", "fcast"] = "assim",
                 variables: Optional[Sequence[str]] = None,
                 units: Literal["ppbv", "mol/mol"] = "ppbv",
                 max_workers: int = 1,
                 use_cache: bool = True,
                 refresh: bool = False,
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False) -> atmoz_dataset:
        """Download (once, then cached) a GEOS-CF point time series and return it as an ``atmoz_dataset``.

        Parameters
        ----------
        lat, lon:
            Site coordinates in decimal degrees; the nearest 0.25-degree grid cell is used.
        start_date, end_date:
            Inclusive UTC days: ``YYYY-MM-DD``, ``YYYYMMDD``, ``date`` or ``datetime``.
        collection:
            ``"aqc"`` (surface air quality, default), ``"chm"`` (chemistry profiles, 72 levels) or ``"met"`` (surface meteorology).
            See ``GEOS_CF.catalog``.
        mode:
            ``"assim"`` (replay/historical, default) or ``"fcast"`` (forecast; ``aqc`` only).
        variables:
            Lower-case names (default: all of the collection's, see ``GEOS_CF.catalog``).
        units:
            ``"ppbv"`` (default) converts gas mixing ratios from the server's mol/mol; ``"mol/mol"`` keeps them raw.
        max_workers:
            Months fetched at once. Server access is serialized anyway (the netCDF libraries are not thread-safe).
        use_cache, refresh, cache_dir:
            File-cache controls. Assimilation data older than two weeks is final; newer data and forecasts are
            re-downloaded after three hours.

        Returns
        -------
        atmoz_dataset
            ``datatype="model"``; see the class docs for ``.data``. ``.metadata`` has ``lat``, ``lon`` (the grid cell),
            ``units`` and the request.

        Raises
        ------
        DownloadError: any month fails (nothing is returned; finished months stay cached).
        """
        q, jobs = cls._validate(lat, lon, start_date, end_date, collection, mode, variables)
        if units not in ("ppbv", "mol/mol"):
            raise ValueError("units must be 'ppbv' or 'mol/mol'")
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("geos_cf")
        frames = run_jobs(
            lambda job: cls._load_one(job, dest_dir=dest_dir, refresh=refresh, use_cache=use_cache),
            jobs, key=lambda j: j["start"], describe=lambda j: f"{j['collection']}/{j['mode']} {j['start']}..{j['end']}",
            noun="GEOS-CF request", desc=f"Downloading GEOS-CF {q.collection}/{q.mode}",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
        return cls._assemble(pd.concat(list(frames.values()), ignore_index=True), q, units)

    @classmethod
    def fetch(cls,
              lat: float,
              lon: float,
              start_date: Union[str, date, datetime],
              end_date: Union[str, date, datetime],
              *,
              collection: Literal["aqc", "chm", "met"] = "aqc",
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
    o3 = ds.data["aqc_assim"]["o3"]
    print(f"O3 range: {o3.min():.1f} - {o3.max():.1f} ppbv")
