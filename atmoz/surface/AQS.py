# -*- coding: utf-8 -*-
"""
EPA AQS pre-generated files: https://aqs.epa.gov/aqsweb/airdata/

* Files stream to a disk cache (never buffered in memory) and are parsed with Apache
  Arrow's multi-threaded CSV reader.
* :class:`EPA_PreGen` follows the package-wide process (see ``docs/conventions.md``):
  ``_validate -> _url -> _fetch_one -> _parse_one -> _assemble``, with ``download()`` and ``fetch()``
  as the public entry points.
* Each zip is parsed once and cached beside it as parquet; ``download()`` filters that parquet, not the network.
"""
from __future__ import annotations

import csv
import difflib
import os
import shutil
import tempfile
import threading
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from atmoz.dataclasses import AtmozDataset, AtmozFrame
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.cache import download_to_cache
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.urls import build_url
from atmoz.resources.utilities import BBox, Job, PathLike, as_list, default_session_handler, run_jobs

_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"

EPA_PARAMETERS = {
    "ozone":           {"code": "44201",           "hourly": True,  "daily": True,  "8hour": True,  "annual": False},
    "so2":             {"code": "42401",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "co":              {"code": "42101",           "hourly": True,  "daily": True,  "8hour": True,  "annual": False},
    "no2":             {"code": "42602",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pm25":            {"code": "88101",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pm25_frm":        {"code": "88101",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pm25_nonfrm":     {"code": "88502",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pm10":            {"code": "81102",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pmc":             {"code": "86101",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pm25_spec":       {"code": "SPEC",            "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pm10_spec":       {"code": "PM10SPEC",        "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "wind":            {"code": "WIND",            "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "temp":            {"code": "TEMP",            "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "pressure":        {"code": "PRESS",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "rh_dp":           {"code": "RH_DP",           "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "haps":            {"code": "HAPS",            "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "vocs":            {"code": "VOCS",            "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "nonoxnoy":        {"code": "NONOxNOy",        "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "lead":            {"code": "LEAD",            "hourly": True,  "daily": True,  "8hour": False, "annual": False},
    "aqi_by_cbsa":     {"code": "aqi_by_cbsa",     "hourly": False, "daily": True,  "8hour": False, "annual": True},
    "aqi_by_county":   {"code": "aqi_by_county",   "hourly": False, "daily": True,  "8hour": False, "annual": True},
    "conc_by_monitor": {"code": "conc_by_monitor", "hourly": False, "daily": False, "8hour": False, "annual": True},
}

RESOLUTIONS = ("hourly", "daily", "8hour", "annual")

# ---------------------------------------------------------------------------
# Column handling for the AQS CSV files.
#   * Latitude / Longitude -> float64 (exact; sign preserved, never rounded)
#   * date columns         -> datetime
#   * low-cardinality text -> category (3-4x less memory)
#   * everything else      -> inferred; identifiers such as "Method Code" stay text
#     so leading zeros ("087") survive.
# ---------------------------------------------------------------------------
_AQS_FLOAT64 = ("Latitude", "Longitude")
_AQS_DATES = ("Date Local", "Date GMT", "Date of Last Change", "Date")
_AQS_CATEGORICAL = frozenset({
    "State Code", "County Code", "Site Num", "Parameter Code", "POC", "Datum",
    "Parameter Name", "Time Local", "Time GMT", "Units of Measure", "Qualifier",
    "Method Type", "Method Code", "Method Name", "State Name", "County Name",
    "City Name", "CBSA Name", "CBSA", "Sample Duration", "Pollutant Standard",
    "Event Type", "Metric Used", "Category", "Defining Parameter",
    "Completeness Indicator", "Certification Indicator", "Local Site Name", "Address",
})


def _read_header(zf: zipfile.ZipFile, member: str) -> List[str]:
    with zf.open(member) as fh:
        first = fh.readline().decode("utf-8-sig")
    return next(csv.reader([first]))


def _select_columns(names: Sequence[str], columns: Optional[Sequence[str]], bbox: Optional[BBox]) -> Optional[List[str]]:
    """Columns to read (``None`` = all). Validates names; adds lat/lon when filtering."""
    if bbox is not None and not all(c in names for c in _AQS_FLOAT64):
        raise ValueError("bbox filtering needs Latitude and Longitude columns, which this file does not have.")
    if columns is None:
        return None
    missing = [c for c in columns if c not in names]
    if missing:
        raise KeyError(f"Columns not in this file: {missing}. Available: {list(names)}")
    wanted = list(columns)
    if bbox is not None:
        wanted += [c for c in _AQS_FLOAT64 if c not in wanted]
    return wanted


def _nulls_to_float(table: pa.Table) -> pa.Table:
    """Columns that are entirely empty come back as Arrow ``null``; pandas users expect float NaN."""
    for i, f in enumerate(table.schema):
        if pa.types.is_null(f.type):
            table = table.set_column(i, f.name, table.column(i).cast(pa.float64()))
    return table


def _read_aqs_arrow(zf: zipfile.ZipFile, member: str, names: List[str],
                    include: Optional[List[str]], bbox: Optional[BBox]) -> pd.DataFrame:
    """Fast path: Arrow's multi-threaded CSV parser reading straight from the zip member."""
    use = names if include is None else include
    types: Dict[str, pa.DataType] = {}
    for col in use:
        if col in _AQS_CATEGORICAL:
            types[col] = pa.dictionary(pa.int32(), pa.string())
        elif col in _AQS_FLOAT64:
            types[col] = pa.float64()
        elif col in _AQS_DATES:
            types[col] = pa.timestamp("s")
    convert = pacsv.ConvertOptions(
        column_types=types,
        include_columns=include,
        strings_can_be_null=True,
    )
    read = pacsv.ReadOptions(block_size=32 << 20)

    with zf.open(member) as fh:
        if bbox is None:
            table = pacsv.read_csv(fh, read_options=read, convert_options=convert)
        else:
            # Stream batch by batch and keep only rows inside the box, so peak
            # memory is one block plus the (small) selection.
            min_lon, min_lat, max_lon, max_lat = bbox
            reader = pacsv.open_csv(fh, read_options=read, convert_options=convert)
            kept = []
            for batch in reader:
                lon, lat = batch["Longitude"], batch["Latitude"]
                mask = pc.and_(
                    pc.and_(pc.greater_equal(lon, min_lon), pc.less_equal(lon, max_lon)),
                    pc.and_(pc.greater_equal(lat, min_lat), pc.less_equal(lat, max_lat)),
                )
                kept.append(batch.filter(mask))
            table = pa.Table.from_batches(kept, schema=reader.schema).unify_dictionaries()
    return _nulls_to_float(table).to_pandas()


def _read_aqs_pandas(zf: zipfile.ZipFile, member: str, names: List[str],
                     include: Optional[List[str]], bbox: Optional[BBox]) -> pd.DataFrame:
    """Slower, whole-column type inference. Used if Arrow cannot infer a column consistently."""
    use = names if include is None else include
    # Read identifiers as text first: dtype="category" would infer numbers and drop leading zeros ("087").
    dtype: Dict[str, Any] = {c: str for c in use if c in _AQS_CATEGORICAL}
    dtype.update({c: "float64" for c in use if c in _AQS_FLOAT64})
    kwargs: Dict[str, Any] = dict(
        usecols=include, dtype=dtype, parse_dates=[c for c in use if c in _AQS_DATES], low_memory=False,
    )
    if int(pd.__version__.split(".")[0]) < 3:  # pandas 3 parses floats exactly by default
        kwargs["float_precision"] = "round_trip"

    with zf.open(member) as fh:
        if bbox is None:
            df = pd.read_csv(fh, **kwargs)
        else:
            min_lon, min_lat, max_lon, max_lat = bbox
            parts = []
            for chunk in pd.read_csv(fh, chunksize=500_000, **kwargs):
                parts.append(chunk[chunk["Longitude"].between(min_lon, max_lon) & chunk["Latitude"].between(min_lat, max_lat)])
            df = pd.concat(parts, ignore_index=True)
    for col in df.columns.intersection(list(_AQS_CATEGORICAL)):
        df[col] = df[col].astype("category")
    return df


def read_aqs_zip(path: PathLike, member: Optional[str] = None, *,
                 columns: Optional[Sequence[str]] = None,
                 bbox: Optional[BBox] = None) -> pd.DataFrame:
    """Parse one EPA AQS pre-generated zip into a DataFrame.

    Parameters
    ----------
    path:
        The ``.zip`` file (for example from :meth:`epa_pregen.fetch`).
    member:
        CSV inside the zip. Defaults to the zip's only / first ``.csv``.
    columns:
        Optional subset of columns to load. Default: all of them.
    bbox:
        Optional ``(min_lon, min_lat, max_lon, max_lat)``; only rows inside are kept
        (applied while streaming, so the full national file is never held in memory).
    """
    with zipfile.ZipFile(path) as zf:
        if member is None:
            member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        names = _read_header(zf, member)
        include = _select_columns(names, columns, bbox)
        try:
            df = _read_aqs_arrow(zf, member, names, include, bbox)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            df = _read_aqs_pandas(zf, member, names, include, bbox)
    if columns is not None:
        df = df[list(columns)]
    return df


# ---------------------------------------------------------------------------
# Filters (applied to the cached parquet) and the per-site/parameter frames.
# ---------------------------------------------------------------------------
_UNITS = {
    "Parts per million": "ppm", "Parts per billion": "ppb", "Parts per trillion": "ppt",
    "Micrograms/cubic meter (LC)": "µg/m3", "Micrograms/cubic meter (25 C)": "µg/m3",
    "Degrees Celsius": "°C", "Millibars": "mbar", "Percent relative humidity": "%",
    "Knots": "kn", "Meters/second": "m/s",
}

_FRAME_COLUMNS = {"Sample Measurement": "value", "Latitude": "lat", "Longitude": "lon", "POC": "poc", "MDL": "mdl",
                  "Qualifier": "qualifier", "Method Type": "method_type", "Method Code": "method_code"}
_TEXT_COLUMNS = ("qualifier", "method_type", "method_code")
_KEY_COLUMNS = ["State Code", "County Code", "Site Num", "Parameter Code"]
_NEEDED = {*_KEY_COLUMNS, *_FRAME_COLUMNS, "Date GMT", "Time GMT", "Date Local", "Time Local", "Units of Measure",
           "Parameter Name", "State Name", "County Name", "Datum"}


def _norm_unit(raw: Any) -> Optional[str]:
    return None if raw is None or pd.isna(raw) else _UNITS.get(str(raw), str(raw))


def _many(value: Any) -> list:
    """A filter value as a list: ``None`` -> ``[]``, a scalar -> ``[scalar]``."""
    return [] if value is None else as_list(value)


def _site_id(site: Any) -> Tuple[str, str, str]:
    """``"240230002"`` / ``"840240230002"`` / ``("24", "023", "0002")`` -> zero-padded ``(state, county, site)``."""
    if isinstance(site, tuple):
        parts = [str(p).strip() for p in site]
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"site {site!r}: give an AQS id such as '240230002' or digits (state, county, site).")
        return parts[0].zfill(2), parts[1].zfill(3), parts[2].zfill(4)
    text = str(site).strip()
    if len(text) == 12 and text.startswith("840"):
        text = text[3:]
    if len(text) != 9 or not text.isdigit():
        raise ValueError(f"site {site!r}: an AQS site id is 9 digits (state 2, county 3, site 4), e.g. '240230002'.")
    return text[:2], text[2:5], text[5:]


def _isin(col: pd.Series, values: Iterable[Any], *, lower: bool = False) -> pd.Series:
    """Rows of *col* equal to one of *values* (text as text, numbers as numbers, dates as dates)."""
    vals = list(values)
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.isin(pd.to_datetime(vals))
    if pd.api.types.is_numeric_dtype(col):
        return col.isin(pd.to_numeric(vals, errors="coerce"))
    text = [str(v).lower() if lower else str(v) for v in vals]
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = pd.Index(col.cat.categories.astype(str))
        return col.isin(col.cat.categories[(cats.str.lower() if lower else cats).isin(text)])
    return (col.astype(str).str.lower() if lower else col.astype(str)).isin(text)


@dataclass
class _Filters:
    """The normalised ``download()`` filters; ``None`` / empty means "no filter"."""

    bbox: Optional[BBox] = None
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    site: List[Tuple[str, str, str]] = field(default_factory=list)
    state: List[str] = field(default_factory=list)
    county: List[str] = field(default_factory=list)
    parameter: List[str] = field(default_factory=list)
    poc: List[str] = field(default_factory=list)
    where: Dict[str, List[Any]] = field(default_factory=dict)

    def applied(self) -> Dict[str, Any]:
        """The filters in use as plain values (``metadata["filters"]``)."""
        out: Dict[str, Any] = {k: getattr(self, k) for k in ("bbox", "site", "state", "county", "parameter", "poc", "where")
                               if getattr(self, k)}
        for key in ("start_date", "end_date"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key).date().isoformat()
        return out

    def columns(self) -> set:
        """File columns the filters need."""
        need = set(self.where)
        if self.bbox:
            need |= {"Latitude", "Longitude"}
        if self.start_date is not None or self.end_date is not None:
            need.add("Date Local")
        for values, cols in ((self.site, _KEY_COLUMNS[:3]), (self.state, ["State Code", "State Name"]),
                             (self.county, ["County Code", "County Name"]), (self.parameter, ["Parameter Code"]),
                             (self.poc, ["POC"])):
            if values:
                need.update(cols)
        return need

    def expression(self, schema: pa.Schema) -> Optional[pads.Expression]:
        """Pyarrow predicate for the bbox and date filters (evaluated while reading the parquet)."""
        parts = []
        if self.bbox is not None:
            west, south, east, north = self.bbox
            lon, lat = pads.field("Longitude"), pads.field("Latitude")
            parts += [lon >= west, lon <= east, lat >= south, lat <= north]
        if self.start_date is not None or self.end_date is not None:
            kind, day = schema.field("Date Local").type, pads.field("Date Local")
            if self.start_date is not None:
                parts.append(day >= pa.scalar(self.start_date.to_pydatetime(), type=kind))
            if self.end_date is not None:
                parts.append(day <= pa.scalar(self.end_date.to_pydatetime(), type=kind))
        expr = None
        for part in parts:
            expr = part if expr is None else expr & part
        return expr

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pandas predicates for the code, name and free-column filters."""
        mask = pd.Series(True, index=df.index)
        if self.site:
            hit = pd.Series(False, index=df.index)
            for state, county, num in self.site:
                hit |= _isin(df["State Code"], [state]) & _isin(df["County Code"], [county]) & _isin(df["Site Num"], [num])
            mask &= hit
        for values, code_col, name_col in ((self.state, "State Code", "State Name"),
                                           (self.county, "County Code", "County Name")):
            if values:
                codes = [v for v in values if v.isdigit()]
                names = [v for v in values if not v.isdigit()]
                hit = _isin(df[code_col], codes) if codes else pd.Series(False, index=df.index)
                if names:
                    hit = hit | _isin(df[name_col], names, lower=True)
                mask &= hit
        if self.parameter:
            mask &= _isin(df["Parameter Code"], self.parameter)
        if self.poc:
            mask &= _isin(df["POC"], self.poc)
        for column, values in self.where.items():
            mask &= _isin(df[column], values)
        return df[mask]


def _frames_from_aqs(df: pd.DataFrame, name: str) -> Dict[str, AtmozFrame]:
    """Split hourly AQS rows into one frame per site and parameter, keyed ``"{state}{county}{site}/{parameter code}"``."""
    needed = ["Sample Measurement", "Latitude", "Longitude", "Date GMT", "Time GMT", *_KEY_COLUMNS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: cannot build time series, the file lacks {missing}. Frames need UTC time (Date GMT + Time GMT); local "
            "standard time cannot be converted without the site's UTC offset. Use fetch() and read_aqs_zip() for such files.")
    if df.empty:
        return {}
    clock = pd.to_datetime(df["Time GMT"].astype(str).str.strip(), format="%H:%M")
    stamp = pd.to_datetime(df["Date GMT"]).dt.normalize() + (clock - clock.dt.normalize())
    work = df.assign(time=stamp.dt.tz_localize("UTC")).dropna(subset=["time"])
    if "Date Local" in work and "Time Local" in work:
        local = pd.to_datetime(work["Time Local"].astype(str).str.strip(), format="%H:%M")
        work["local_time"] = pd.to_datetime(work["Date Local"]).dt.normalize() + (local - local.dt.normalize())
    frames: Dict[str, AtmozFrame] = {}
    for (state, county, num, code), grp in work.groupby(_KEY_COLUMNS, observed=True, sort=True):
        out = grp.rename(columns=_FRAME_COLUMNS)
        keep = [c for c in ("value", "lat", "lon", "poc", "local_time", "mdl", *_TEXT_COLUMNS) if c in out.columns]
        out = out[["time", *keep]].sort_values("time", kind="stable").set_index("time")
        for col in _TEXT_COLUMNS:
            if col in out:
                out[col] = out[col].astype(object)
        if "poc" in out:
            out["poc"] = pd.to_numeric(out["poc"]).astype("int64")
        first = grp.iloc[0]
        unit = _norm_unit(first.get("Units of Measure"))

        def text(column: str) -> Optional[str]:
            return None if column not in grp or pd.isna(first[column]) else str(first[column])

        meta = {"site_id": f"{state}{county}{num}", "state_code": str(state), "county_code": str(county),
                "site_num": str(num), "state_name": text("State Name"), "county_name": text("County Name"),
                "parameter_code": str(code), "parameter_name": text("Parameter Name"), "datum": text("Datum"),
                "units_raw": text("Units of Measure"), "lat": float(first["Latitude"]), "lon": float(first["Longitude"]),
                "pocs": sorted(out["poc"].unique().tolist()) if "poc" in out else []}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # units AtmozFrame does not know are kept as the file spells them
            frames[f"{state}{county}{num}/{code}"] = AtmozFrame(
                out, units={"value": unit} if unit else None, lat="lat", lon="lon", metadata=meta)
    return frames


class EPA_PreGen:
    """Download and parse the EPA pre-generated AQS files.

    Public entry points:

    * :meth:`download` -> :class:`~atmoz.AtmozDataset` (one frame per site and parameter)
    * :meth:`fetch`    -> ``{csv_name: zip_path}`` (files only, parse them yourself)

    Notes:
        * Files live in the on-disk cache (``atmoz.cache_dir("aqs")``) and are only downloaded
          again when EPA has changed them.
        * Downloads run on a thread pool (network bound); each file is parsed with Arrow's own
          multi-threaded reader, which releases the GIL, so a process pool is not needed and
          would only add the cost of pickling large DataFrames back.
        * All-or-nothing: if any file fails, :class:`~atmoz.exceptions.DownloadError` is raised
          and nothing is returned. Finished files stay cached, so re-running repeats only the failures.
    """

    parameters = EPA_PARAMETERS
    base_url = _BASE_URL
    resolutions = RESOLUTIONS

    # O(1) lookups: code -> resolutions, and lower-cased code -> canonical code.
    param_code_map: Dict[str, Dict[str, Any]] = {}
    for _v in EPA_PARAMETERS.values():
        param_code_map.setdefault(_v["code"], _v)
    _code_lookup = {c.lower(): c for c in param_code_map}
    del _v

    # ---- 1. validate ---------------------------------------------------
    @classmethod
    def _resolve_parameter(cls, parameter: Any) -> str:
        """Turn a friendly name (``"ozone"``) or a code (``"44201"``, ``"SPEC"``) into the file-name code."""
        text = str(parameter).strip()
        key = text.lower()
        if key in cls.parameters:
            return cls.parameters[key]["code"]
        if key in cls._code_lookup:
            return cls._code_lookup[key]
        if text.isdigit():
            return text  # AQS code we have no table entry for: let EPA's server decide
        hint = difflib.get_close_matches(key, list(cls.parameters), n=3)
        raise ValueError(
            f"Unknown AQS parameter {parameter!r}."
            + (f" Did you mean {hint}?" if hint else f" Known names: {sorted(cls.parameters)}")
        )

    @classmethod
    def _validate(cls,
                  parameters: Union[str, Iterable[str]],
                  resolutions: Union[str, Iterable[str]],
                  years: Union[int, str, Iterable[Union[int, str]]],
                  *,
                  silent: bool = False) -> List[Dict[str, Any]]:
        """Expand parameter x resolution x year into jobs ``{"resolution", "parameter", "year"}`` (no duplicates).

        Unknown parameter names or resolutions raise ``ValueError``. Combinations EPA does not
        publish (per ``EPA_PARAMETERS``) are skipped with a ``UserWarning`` unless ``silent=True``.
        Raises ``ValueError`` if nothing valid is left.
        """
        codes = [cls._resolve_parameter(p) for p in as_list(parameters)]
        ress = [str(r).strip().lower() for r in as_list(resolutions)]
        bad = [r for r in ress if r not in cls.resolutions]
        if bad:
            raise ValueError(f"Unknown resolution(s) {bad}; choose from {list(cls.resolutions)}.")
        yrs = [int(y) for y in as_list(years)]

        seen = set()
        jobs: List[Dict[str, Any]] = []
        for year in yrs:
            for code in codes:
                for res in ress:
                    if (res, code, year) in seen:
                        continue
                    seen.add((res, code, year))
                    entry = cls.param_code_map.get(code)
                    if entry is not None and not entry.get(res, False):
                        if not silent:
                            warnings.warn(f"Skipping {code}: no {res!r} file is listed for it.", stacklevel=3)
                        continue
                    jobs.append({"resolution": res, "parameter": code, "year": year})
        if not jobs:
            raise ValueError("No valid parameter/resolution/year combination to download.")
        return jobs

    # ---- 2. url ----------------------------------------------------------
    @staticmethod
    def _name(job: Job) -> str:
        """The csv name a job produces, e.g. ``hourly_44201_2025.csv`` (also the key in results)."""
        return f"{job['resolution']}_{job['parameter']}_{job['year']}.csv"

    @classmethod
    def _url(cls, job: Job) -> str:
        return build_url(cls.base_url, cls._name(job).replace(".csv", ".zip"))

    @staticmethod
    def _default_max_age(year: int) -> Optional[timedelta]:
        # EPA keeps revising the current and previous year; older years are frozen.
        return timedelta(days=1) if year >= date.today().year - 1 else None

    # ---- 3. fetch (network -> file on disk) ------------------------------------
    @classmethod
    def _fetch_one(cls, job: Job, *, dest_dir: Path, refresh: bool = False,
                   max_age: Union[str, timedelta, None] = "auto",
                   session_handler: Optional[SessionHandler] = None) -> Path:
        """Stream one zip into *dest_dir* (skipped if the cached copy is still current). Returns its path."""
        age = cls._default_max_age(job["year"]) if max_age == "auto" else max_age
        dest = dest_dir / cls._name(job).replace(".csv", ".zip")
        return download_to_cache(cls._url(job), dest, (session_handler or default_session_handler()).session(),
                                 max_age=age, refresh=refresh)


    # ---- 4. parse (file on disk -> parquet cache -> filtered DataFrame) --------------
    @staticmethod
    def _ensure_parquet(zip_path: Path, member: str) -> Path:
        """The parquet copy of *zip_path*, written once (and again only when the zip is newer)."""
        target = zip_path.with_suffix(".parquet")
        if target.exists() and target.stat().st_mtime >= zip_path.stat().st_mtime:
            return target
        table = pa.Table.from_pandas(read_aqs_zip(zip_path, member), preserve_index=False)
        part = target.with_name(f"{target.name}.{os.getpid()}-{threading.get_ident()}.part")
        try:
            pq.write_table(table, part)
            os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)
        return target

    @classmethod
    def _parse_one(cls, job: Job, path: Path, *, filters: Optional[_Filters] = None) -> pd.DataFrame:
        """Rows of one file that pass *filters*, read from its parquet cache (only the columns that are needed)."""
        filters = filters or _Filters()
        name = cls._name(job)
        parquet = cls._ensure_parquet(path, name)
        schema = pq.read_schema(parquet)
        absent = [c for c in sorted(filters.columns()) if c not in schema.names]
        if absent:
            raise KeyError(f"{name}: filter columns not in this file: {absent}. Available: {schema.names}")
        columns = [c for c in schema.names if c in _NEEDED or c in filters.columns()]
        table = pq.read_table(parquet, columns=columns, filters=filters.expression(schema))
        return filters.apply(_nulls_to_float(table).to_pandas())

    # ---- fetch + parse for one job ---------------------------------------------
    @classmethod
    def _load_one(cls, job: Job, *, filters: Optional[_Filters] = None, use_cache: bool = True, refresh: bool = False,
                  max_age: Union[str, timedelta, None] = "auto", cache_dir: Optional[PathLike] = None,
                  session_handler: Optional[SessionHandler] = None) -> pd.DataFrame:
        tmp = None
        try:
            if use_cache:
                dest_dir = Path(cache_dir) if cache_dir else _cache_root("aqs")
            else:
                tmp = tempfile.mkdtemp(prefix="atmoz_aqs_")
                dest_dir = Path(tmp)
            path = cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh or not use_cache,
                                  max_age=max_age, session_handler=session_handler)
            return cls._parse_one(job, path, filters=filters)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    # ---- 5. assemble -------------------------------------------------------------
    @staticmethod
    def _assemble(tables: Dict[str, pd.DataFrame], filters: _Filters) -> AtmozDataset:
        meta = {"endpoint": "aqs", "files": list(tables), "rows": {name: len(df) for name, df in tables.items()},
                "filters": filters.applied()}
        name = "EPA AQS " + ", ".join(Path(n).stem for n in tables)
        rows = [df for df in tables.values() if len(df)]
        frames = _frames_from_aqs(pd.concat(rows, ignore_index=True), name) if rows else {}
        return AtmozDataset(name=name, source="EPA AQS", frames=frames, metadata=meta)

    @staticmethod
    def _filters(bbox, start_date, end_date, site, state, county, parameter, poc, where) -> _Filters:
        """Validate and normalise the ``download()`` filters (before anything is downloaded)."""
        sites = [] if site is None else [site] if isinstance(site, (str, int, tuple)) else list(site)
        start = None if start_date is None else pd.Timestamp(start_date).tz_localize(None).normalize()
        end = None if end_date is None else pd.Timestamp(end_date).tz_localize(None).normalize()
        if start is not None and end is not None and start > end:
            raise ValueError(f"start_date {start.date()} is after end_date {end.date()}.")
        if bbox is not None:
            if len(bbox) != 4 or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise ValueError("bbox must be (west, south, east, north) with west <= east and south <= north.")
            bbox = tuple(float(v) for v in bbox)
        codes = [str(p) for p in _many(parameter)]
        return _Filters(
            bbox=bbox, start_date=start, end_date=end, site=[_site_id(s) for s in sites],
            state=[str(s).strip().zfill(2) if str(s).strip().isdigit() else str(s).strip() for s in _many(state)],
            county=[str(c).strip().zfill(3) if str(c).strip().isdigit() else str(c).strip() for c in _many(county)],
            parameter=[str(EPA_PARAMETERS[c.lower()]["code"]) if c.lower() in EPA_PARAMETERS else c for c in codes],
            poc=[str(p) for p in _many(poc)],
            where={str(k): _many(v) for k, v in (where or {}).items()},
        )

    # ---- public --------------------------------------------------------------------
    @classmethod
    def download(cls,
                 parameters: Union[str, List[str]],
                 resolutions: Union[str, List[str]],
                 years: Union[int, List[int]],
                 *,
                 bbox: Optional[BBox] = None,
                 start_date: Union[str, date, pd.Timestamp, None] = None,
                 end_date: Union[str, date, pd.Timestamp, None] = None,
                 site: Union[str, Tuple[str, str, str], Sequence, None] = None,
                 state: Union[str, int, Sequence, None] = None,
                 county: Union[str, int, Sequence, None] = None,
                 parameter: Union[str, int, Sequence, None] = None,
                 poc: Union[int, Sequence[int], None] = None,
                 where: Optional[Mapping[str, Any]] = None,
                 max_workers: int = 5,
                 use_cache: bool = True,
                 refresh: bool = False,
                 max_age: Union[str, timedelta, None] = "auto",
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False,
                 session_handler: Optional[SessionHandler] = None) -> AtmozDataset:
        """Download every valid (parameter, resolution, year) combination as one frame per site and parameter.

        Only ``"hourly"`` files carry timestamps (UTC and local); other resolutions raise ``ValueError``, use
        :meth:`fetch` and :func:`read_aqs_zip` for them.

        Parameters
        ----------
        parameters, resolutions, years:
            One or more of each. Parameters may be names (``"ozone"``) or AQS codes. They choose the files.
        bbox:
            ``(west, south, east, north)`` on the site latitude / longitude.
        start_date, end_date:
            Inclusive local (standard time) dates. The files are still chosen by *years*; these trim the rows.
        site:
            An AQS id (``"240230002"``), a ``(state, county, site)`` tuple, or a list of those.
        state, county:
            Codes (``"24"``, ``"031"``) or names (``"Maryland"``); one or several. Codes are zero-padded.
        parameter, poc:
            Rows of one AQS parameter code (``44201``) or POC (monitor number) inside the file; one or several.
        where:
            ``{column: value-or-list}`` for any column of the file, e.g. ``{"Method Code": "087"}``.
        max_workers:
            Download threads (default 5).
        use_cache, refresh, max_age, cache_dir:
            File-cache controls. ``max_age="auto"`` revalidates the current and previous
            year's files daily (cheap ``ETag`` check) and treats older years as frozen.
            ``refresh=True`` forces a fresh download; ``use_cache=False`` keeps nothing on disk.
            Each zip is parsed once into a parquet file next to it; filters read that parquet.
        silent:
            No progress bar and no warnings.

        Returns
        -------
        AtmozDataset
            ``source="EPA AQS"``, ``metadata`` with ``endpoint``, ``files``, ``rows`` (per file, after filtering) and
            ``filters``. Keys are ``"{state}{county}{site}/{parameter code}"``, e.g. ``"240230002/44201"``. Each frame is
            indexed by UTC ``time`` (hour-beginning, from ``Date GMT`` + ``Time GMT``) with columns ``value`` (the file's
            unit, declared in ``units``: ppm stays ppm), ``lat``, ``lon``, ``poc``, ``local_time`` (naive local standard
            time, hour-beginning) and, when present, ``mdl``, ``qualifier``, ``method_type``, ``method_code``. Several
            POCs at a site share the frame; the ``poc`` column tells them apart. Nothing left after filtering gives an
            empty dataset.

        Raises
        ------
        DownloadError
            If any file fails. No partial result is returned.
        """
        jobs = cls._validate(parameters, resolutions, years, silent=silent)
        if any(job["resolution"] != "hourly" for job in jobs):
            raise ValueError("download() builds time series from the 'hourly' files only; use fetch() and read_aqs_zip() "
                             "for daily, 8hour and annual files.")
        filters = cls._filters(bbox, start_date, end_date, site, state, county, parameter, poc, where)
        tables = run_jobs(
            lambda job: cls._load_one(job, filters=filters, use_cache=use_cache, refresh=refresh,
                                      max_age=max_age, cache_dir=cache_dir, session_handler=session_handler),
            jobs, key=cls._name, describe=cls._name, noun="download", desc="Downloading EPA AQS data",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
        return cls._assemble(tables, filters)

    @classmethod
    def fetch(cls,
              parameters: Union[str, List[str]],
              resolutions: Union[str, List[str]],
              years: Union[int, List[int]],
              *,
              max_workers: int = 5,
              refresh: bool = False,
              max_age: Union[str, timedelta, None] = "auto",
              cache_dir: Optional[PathLike] = None,
              silent: bool = False,
              show_traceback: bool = False,
              session_handler: Optional[SessionHandler] = None) -> Dict[str, Path]:
        """Download the zips only (cached) and return ``{csv_name: zip_path}``. Parse them yourself with :func:`read_aqs_zip`."""
        jobs = cls._validate(parameters, resolutions, years, silent=silent)
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("aqs")
        dest_dir.mkdir(parents=True, exist_ok=True)
        return run_jobs(
            lambda job: cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh, max_age=max_age,
                                       session_handler=session_handler),
            jobs, key=cls._name, describe=cls._name, noun="download", desc="Downloading EPA AQS files",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
