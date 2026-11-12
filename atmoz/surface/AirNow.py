# -*- coding: utf-8 -*-
"""
Created on 2025-10-02 09:24:37

@author: Maurice Roots

Description:
     - EPA surface air-quality data: the AirNow API and the EPA AQS pre-generated files.

Two backends, one module:

* :class:`epa_pregen` – bulk yearly files from https://aqs.epa.gov/aqsweb/airdata/.
  Files are streamed to a disk cache (never buffered in memory) and parsed with
  Apache Arrow's multi-threaded CSV reader.
* :class:`AirNow` – the live AirNow web API, requested one window (default: one
  day) at a time on a thread pool.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import requests

from atmoz.exceptions import APIError, DownloadError
from atmoz.resources import credentials
from atmoz.resources.atmoz_dataclasses import atmoz_dataset
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.cache import download_to_cache
from atmoz.resources.parallelExecutor import JobResult, ParallelExecutor
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.urls import build_url, redact

PathLike = Union[str, Path]
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

# ---------------------------------------------------------------------------
# Module-level session handler: one instance shared by all threads. Each thread
# transparently gets its own requests.Session via threading.local, with
# connection pooling, retries on 429/5xx and a default timeout.
# ---------------------------------------------------------------------------
_session_handler = SessionHandler(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=3,
    backoff_factor=0.5,
    timeout=(5.0, 60.0),
)

# ---------------------------------------------------------------------------
# Endpoints and parameter mappings for EPA PreGenerated (AQS) and AirNow API.
# ---------------------------------------------------------------------------
_BASE_URL_AQS = "https://aqs.epa.gov/aqsweb/airdata"
_BASE_URL_AIRNOW = "https://www.airnowapi.org/aq/data"

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


def _as_list(value: Any) -> list:
    if isinstance(value, (str, bytes, int, float)) or value is None:
        return [value]
    return list(value)


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


class epa_pregen:
    """Download and parse the EPA pre-generated AQS files.

    * Files stream to the on-disk cache (``atmoz.cache_dir("aqs")``) and are only
      downloaded again when EPA has changed them.
    * Downloads run on a thread pool (network bound); each file is parsed with
      Arrow's own multi-threaded reader, which releases the GIL, so a process pool
      is not needed and would only add the cost of pickling large DataFrames back.
    * Downloads are all-or-nothing: if any file fails, :class:`~atmoz.exceptions.DownloadError`
      is raised and no partial result is returned. Finished files stay cached, so
      re-running only repeats the failed ones.
    """

    parameters = EPA_PARAMETERS
    base_url_aqs = _BASE_URL_AQS
    resolutions = RESOLUTIONS

    # O(1) lookups: code -> resolutions, and lower-cased code -> canonical code.
    param_code_map: Dict[str, Dict[str, Any]] = {}
    for _v in EPA_PARAMETERS.values():
        param_code_map.setdefault(_v["code"], _v)
    _code_lookup = {c.lower(): c for c in param_code_map}
    del _v

    # ------------------------------------------------------------------
    # Validation of parameter <-> resolution <-> year combinations
    # ------------------------------------------------------------------
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
    def _parameters_validator(cls,
                              parameters: Union[str, Iterable[str]],
                              resolutions: Union[str, Iterable[str]],
                              years: Union[int, str, Iterable[Union[int, str]]],
                              **kwargs) -> List[Dict[str, Any]]:
        """Expand and validate parameter x resolution x year into download jobs.

        Returns a list of ``{"resolution", "parameter", "year"}`` dicts (no duplicates).
        Unknown parameter names or resolutions raise ``ValueError``. Combinations that
        EPA does not publish (per ``EPA_PARAMETERS``) are skipped with a ``UserWarning``
        unless ``silent=True``.
        """
        silent = kwargs.get("silent", False)
        codes = [cls._resolve_parameter(p) for p in _as_list(parameters)]
        ress = [str(r).strip().lower() for r in _as_list(resolutions)]
        bad = [r for r in ress if r not in cls.resolutions]
        if bad:
            raise ValueError(f"Unknown resolution(s) {bad}; choose from {list(cls.resolutions)}.")
        yrs = [int(y) for y in _as_list(years)]

        seen = set()
        combos: List[Dict[str, Any]] = []
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
                    combos.append({"resolution": res, "parameter": code, "year": year})
        return combos

    # ------------------------------------------------------------------
    # URLs, cache and single-file work
    # ------------------------------------------------------------------
    @classmethod
    def _stem(cls, resolution: str, parameter: str, year: int) -> str:
        return f"{resolution}_{parameter}_{year}"

    @classmethod
    def _url(cls, resolution: str, parameter: str, year: int) -> str:
        return build_url(cls.base_url_aqs, cls._stem(resolution, parameter, year) + ".zip")

    @staticmethod
    def _default_max_age(year: int) -> Optional[timedelta]:
        # EPA keeps revising the current and previous year; older years are frozen.
        return timedelta(days=1) if year >= date.today().year - 1 else None

    @classmethod
    def _fetch_zip(cls, resolution: str, parameter: str, year: int, *, dest_dir: Path,
                   refresh: bool = False, max_age: Union[str, timedelta, None] = "auto",
                   session_handler: Optional[SessionHandler] = None) -> Path:
        age = cls._default_max_age(year) if max_age == "auto" else max_age
        dest = dest_dir / (cls._stem(resolution, parameter, year) + ".zip")
        return download_to_cache(
            cls._url(resolution, parameter, year), dest, (session_handler or _session_handler).session(),
            max_age=age, refresh=refresh,
        )

    @classmethod
    def _run(cls, work, combos: List[Dict[str, Any]], *, max_workers: int, desc: str,
             show_traceback: bool, silent: bool) -> Dict[str, Any]:
        """Run ``work(**combo)`` on threads; all-or-nothing."""
        executor = ParallelExecutor(
            max_workers=max_workers, desc=desc, mode="thread",
            show_traceback=show_traceback, progress=not silent, print_failures=False,
        )
        result: JobResult = executor(work)(combos)
        if result.failed:
            lines = [
                "  {resolution}_{parameter}_{year}: {exc}".format(**f.job, exc=f.exc)
                for f in result.failed
            ]
            raise DownloadError(
                f"{len(result.failed)} of {len(combos)} download(s) failed (nothing returned):\n" + "\n".join(lines),
                failed=[(f.job, f.exc) for f in result.failed],
            )
        # Return in the order requested, not completion order.
        return {cls._stem(**c) + ".csv": result.succeeded[cls._stem(**c) + ".csv"] for c in combos}

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
              show_traceback: bool = False,
              silent: bool = False,
              session_handler: Optional[SessionHandler] = None,
              **kwargs) -> Dict[str, Path]:
        """Download the zips only (cached) and return ``{csv_name: zip_path}``.

        Use this when you want to parse the files yourself.
        """
        combos = cls._validated(parameters, resolutions, years, silent)
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("aqs")
        dest_dir.mkdir(parents=True, exist_ok=True)

        def work(resolution, parameter, year):
            path = cls._fetch_zip(resolution, parameter, year, dest_dir=dest_dir, refresh=refresh,
                                  max_age=max_age, session_handler=session_handler)
            return {"key": cls._stem(resolution, parameter, year) + ".csv", "value": path}

        return cls._run(work, combos, max_workers=max_workers, desc="Downloading EPA AQS files",
                        show_traceback=show_traceback, silent=silent)

    @classmethod
    def _validated(cls, parameters, resolutions, years, silent) -> List[Dict[str, Any]]:
        combos = cls._parameters_validator(parameters, resolutions, years, silent=silent)
        if not combos:
            raise ValueError("No valid parameter/resolution/year combination to download.")
        return combos

    @classmethod
    def _download_single(cls, resolution: str, parameter: str, year: int, *,
                         columns: Optional[Sequence[str]] = None, bbox: Optional[BBox] = None,
                         use_cache: bool = True, refresh: bool = False,
                         max_age: Union[str, timedelta, None] = "auto",
                         cache_dir: Optional[PathLike] = None,
                         session_handler: Optional[SessionHandler] = None) -> Dict[str, Any]:
        """Download (stream to disk) and parse one file. Returns ``{"key": csv_name, "value": DataFrame}``."""
        name = cls._stem(resolution, parameter, year) + ".csv"
        tmp = None
        try:
            if use_cache:
                dest_dir = Path(cache_dir) if cache_dir else _cache_root("aqs")
            else:
                tmp = tempfile.mkdtemp(prefix="atmoz_aqs_")
                dest_dir = Path(tmp)
            zip_path = cls._fetch_zip(resolution, parameter, year, dest_dir=dest_dir,
                                      refresh=refresh or not use_cache, max_age=max_age,
                                      session_handler=session_handler)
            df = read_aqs_zip(zip_path, name, columns=columns, bbox=bbox)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        return {"key": name, "value": df}

    @classmethod
    def _download(cls,
                  parameters: Union[str, List[str]],
                  resolutions: Union[str, List[str]],
                  years: Union[int, List[int]],
                  max_workers: int = 5,
                  show_traceback: bool = False,
                  *,
                  columns: Optional[Sequence[str]] = None,
                  bbox: Optional[BBox] = None,
                  use_cache: bool = True,
                  refresh: bool = False,
                  max_age: Union[str, timedelta, None] = "auto",
                  cache_dir: Optional[PathLike] = None,
                  silent: bool = False,
                  session_handler: Optional[SessionHandler] = None,
                  **kwargs) -> Dict[str, pd.DataFrame]:
        """Download and parse every valid (parameter, resolution, year) combination.

        Parameters
        ----------
        parameters, resolutions, years:
            One or more of each. Parameters may be names (``"ozone"``) or AQS codes.
        max_workers:
            Download threads (default 5).
        columns:
            Load only these columns. Default: everything.
        bbox:
            ``(min_lon, min_lat, max_lon, max_lat)``; keep only rows inside the box.
        use_cache, refresh, max_age, cache_dir:
            File-cache controls. ``max_age="auto"`` revalidates the current and previous
            year's files daily (cheap ``ETag`` check) and treats older years as frozen.
            ``refresh=True`` forces a fresh download; ``use_cache=False`` keeps nothing on disk.
        silent:
            No progress bar and no warnings.

        Returns
        -------
        dict
            ``{"hourly_44201_2025.csv": DataFrame, ...}``

        Raises
        ------
        DownloadError
            If any file fails. No partial result is returned.
        """
        combos = cls._validated(parameters, resolutions, years, silent)

        def work(resolution, parameter, year):
            return cls._download_single(
                resolution, parameter, year, columns=columns, bbox=bbox, use_cache=use_cache,
                refresh=refresh, max_age=max_age, cache_dir=cache_dir, session_handler=session_handler,
            )

        return cls._run(work, combos, max_workers=max_workers, desc="Downloading EPA AQS data",
                        show_traceback=show_traceback, silent=silent)


# ---------------------------------------- #
# AirNow API Handler Class
# ---------------------------------------- #
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HOUR_FMT = "%Y-%m-%dT%H"


@dataclass
class AirNowParams:
    startDate: str = field(default_factory=lambda: (datetime.now(timezone.utc) - timedelta(days=1)).strftime(_HOUR_FMT))
    endDate: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime(_HOUR_FMT))
    parameters: list = field(default_factory=lambda: ["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"])
    BBOX: list = field(default_factory=lambda: ["-80.655479", "35.574398", "-72.086143", "41.415693"])
    dataType: str = field(default='B')
    format: str = field(default='application/json')
    verbose: str = field(default='1')
    monitorType: str = field(default='0')
    includerawconcentrations: str = field(default='1')

    def __post_init__(self) -> None:
        """Replace ``None`` with the field default; normalise ``parameters`` / ``BBOX`` to lists of str."""
        for f in fields(self):
            if getattr(self, f.name) is None:
                if f.default_factory is not MISSING:
                    setattr(self, f.name, f.default_factory())
                elif f.default is not MISSING:
                    setattr(self, f.name, f.default)
        if isinstance(self.parameters, str):
            self.parameters = [p.strip() for p in self.parameters.split(",")]
        if isinstance(self.BBOX, str):
            self.BBOX = [p.strip() for p in self.BBOX.split(",")]
        self.BBOX = [str(v) for v in self.BBOX]  # str() keeps full float precision and the sign
        if len(self.BBOX) != 4:
            raise ValueError("BBOX must be [minLon, minLat, maxLon, maxLat].")


def _parse_bound(value: Union[str, date, datetime, pd.Timestamp], *, is_end: bool) -> pd.Timestamp:
    """Normalise a user date to a naive-UTC hour. A bare date as the *end* means 'through 23:00 that day'."""
    date_only = isinstance(value, str) and bool(_DATE_ONLY.match(value.strip()))
    date_only = date_only or (isinstance(value, date) and not isinstance(value, datetime))
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ts = ts.floor("h")
    if date_only and is_end:
        ts = ts + pd.Timedelta(hours=23)
    return ts


def make_windows(start: Union[str, date, datetime], end: Union[str, date, datetime],
                 chunk: Union[str, timedelta] = "1D", *, end_inclusive: bool = True) -> List[Tuple[str, str]]:
    """Split ``[start, end]`` into non-overlapping ``(startDate, endDate)`` request windows.

    Each window covers *chunk* of time (default one day). With ``end_inclusive=True``
    (AirNow returns the ``endDate`` hour) a window ends one hour before the next starts, so
    no hour is requested twice. ``start == end`` gives a single window.
    """
    t0, t1 = _parse_bound(start, is_end=False), _parse_bound(end, is_end=True)
    if t1 < t0:
        raise ValueError(f"end_date ({end}) is before start_date ({start}).")
    step = pd.Timedelta(chunk)
    if step < pd.Timedelta(hours=1):
        raise ValueError("chunk must be at least one hour.")
    hour = pd.Timedelta(hours=1)
    windows = []
    cursor = t0
    while cursor <= t1:
        last = cursor + step - hour if end_inclusive else cursor + step
        windows.append((cursor.strftime(_HOUR_FMT), min(last, t1).strftime(_HOUR_FMT)))
        cursor += step
    return windows


class AirNow:
    """
    Fetch air-quality data from the AirNow API (and, via :meth:`download`, the EPA AQS files).

    Features:
        - API key from the ``api_key`` argument, the ``ATMOZ_AIRNOW_API_KEY`` environment
          variable, or a ``.atmoz.env`` file (save it once with :meth:`set_api_key`).
        - One request per time window (default one day), run on a thread pool. A request
          for the whole country, all pollutants and many days in one go gets blocked by
          AirNow, so you choose the window size (``chunk=``) and the pollutants / box.
        - Finished windows older than ``settle_hours`` are cached on disk.
        - All-or-nothing: if any window fails, :class:`~atmoz.exceptions.DownloadError` is
          raised and no partial result is returned. Completed windows stay cached.
        - The API key never appears in exceptions or reprs.

    Attributes:
        base_url (str): Base endpoint for the AirNow API.
        window_end_inclusive (bool): Whether AirNow returns the ``endDate`` hour. ``True``
            (the default) makes request windows end one hour before the next begins.
        settle_hours (int): Windows that end less than this many hours ago are never cached,
            because AirNow may still revise recent observations.
    """

    base_url = _BASE_URL_AIRNOW
    window_end_inclusive = True
    settle_hours = 72

    def __init__(self, api_key: Optional[str] = None, *, session_handler: Optional[SessionHandler] = None) -> None:
        """
        Args:
            api_key: AirNow API key. If ``None`` it is looked up lazily on first use (see class docs).
            session_handler: Shared HTTP session pool (retries, timeouts, connection reuse).
        """
        self._api_key = api_key
        self._session_handler = session_handler or _session_handler
        self._queries_strings: List[str] = []  # request URLs with the key removed
        self._data: List[pd.DataFrame] = []

    def __repr__(self) -> str:
        return f"AirNow(api_key={'***' if self._api_key else None})"

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        """The API key (resolved on first access). Raises ``MissingCredentialError`` if there is none."""
        if not self._api_key:
            self._api_key = credentials.get_secret("airnow")
        return self._api_key

    @staticmethod
    def set_api_key(key: Optional[str] = None, *, scope: Literal["user", "project"] = "user") -> Path:
        """Save your AirNow key so every future session finds it.

        Call it once. With no argument you are prompted (hidden input). ``scope="user"``
        writes ``~/.atmoz/.atmoz.env``; ``scope="project"`` writes ``./.atmoz.env``.
        Returns the file that was written (mode 0600). Keep it out of version control.
        """
        return credentials.set_secret("airnow", key, scope=scope)

    # ------------------------------------------------------------------
    # URL building and requests
    # ------------------------------------------------------------------
    def _build_url(self, **kwargs) -> str:
        """
        Build the request URL for :class:`AirNowParams` fields given as keyword arguments.

        Pass ``api_key=...`` to include the key; without it the URL is safe to print or log.
        """
        key = kwargs.pop("api_key", None)
        query = asdict(AirNowParams(**kwargs))
        if key:
            query["API_KEY"] = key
        return build_url(self.base_url, params=query)

    def _pull(self, **kwargs) -> requests.Response:
        """
        Send one request to the AirNow API and return the response.

        Keyword arguments are :class:`AirNowParams` fields. Raises :class:`~atmoz.exceptions.APIError`
        with the key scrubbed from the message.
        """
        key = self.api_key
        self._queries_strings.append(self._build_url(**kwargs))
        url = self._build_url(api_key=key, **kwargs)
        try:
            response = self._session_handler.session().get(url)
        except requests.RequestException as exc:
            # requests puts the full URL (with the key) in its messages, so re-raise a clean one.
            raise APIError(f"AirNow request failed: {type(exc).__name__}: {redact(exc, key)}") from None
        if response.status_code != 200:
            raise APIError(
                f"AirNow API returned HTTP {response.status_code}: {redact(response.text[:500], key)}",
                status_code=response.status_code,
            )
        return response

    def _process(self, response: requests.Response, **kwargs) -> None:
        """Store a response as a DataFrame in ``self._data``."""
        self._data.append(pd.DataFrame(response.json()))

    def _metadata(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extracts and formats metadata from a given DataFrame by removing data columns, deduplicating entries, and aggregating parameter information.

        Args:
            df (pd.DataFrame): Input DataFrame containing air quality data with columns such as "UTC", "Value", "RawConcentration", "AQI", "Category", "Parameter", "Unit", and "FullAQSCode".
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            pd.DataFrame: A DataFrame containing unique metadata entries with aggregated parameter and unit information for each unique "FullAQSCode". The resulting DataFrame includes a "Parameters" column (list of parameter/unit dictionaries) and an "id" column (row index).
        """
        data_columns = ["UTC", "Value", "RawConcentration", "AQI", "Category"]
        metadata = df.copy().drop(
            columns=data_columns
            ).drop_duplicates(keep="first").reset_index(drop=True)
        
        parameters = [
            metadata[["Parameter", "Unit"]][metadata["FullAQSCode"] == site].to_dict(orient="list")
            for site in metadata["FullAQSCode"].unique()
        ]
        
        metadata = metadata.copy().drop(
            columns=["Parameter", "Unit"]
            ).drop_duplicates(keep="first").reset_index(drop=True)
        
        metadata["Parameters"] = parameters
        metadata["id"] = metadata.index
        return metadata

    def _nest(self,
              df: pd.DataFrame,
              metadata: pd.DataFrame,
              **kwargs
              ) -> Dict[str, Dict[Any, pd.DataFrame]]:
        """
        Nest air quality measurements by pollutant and station ID.

        Args:
            df (pd.DataFrame): Measurements with columns like 'UTC', 'Value', 'Parameter', etc.
            metadata (pd.DataFrame): Station metadata with 'Latitude', 'Longitude', 'id'.
            **kwargs:
                data_cols (list of str, optional): Columns to retain (default: standard set).

        Returns:
            dict: Nested dictionary {parameter: {station_id: DataFrame}}.
        """
        keep_cols = kwargs.get(
            "data_cols", 
            ["UTC", "Value", "RawConcentration", "AQI", "Category", "Latitude", "Longitude", "Parameter"]
        )
        
        df = df[[c for c in df.columns if c in keep_cols]].copy()

        metadata_small = metadata[["Latitude", "Longitude", "id"]].drop_duplicates()
        df = df.merge(metadata_small, on=["Latitude", "Longitude"], how="left")
        df.drop(columns=["Latitude", "Longitude"], inplace=True)

        df["UTC"] = pd.to_datetime(df["UTC"])
        df.set_index("UTC", inplace=True)
        
        data = {
            parameter: 
                    {
                        id_: group.drop(columns=["Parameter", "id"]) 
                        for id_, group in df_param.groupby("id")
                    }
            for parameter, df_param in df.groupby("Parameter")
            }

        return data

    # ------------------------------------------------------------------
    # Windows, caching and parallel fetching
    # ------------------------------------------------------------------
    def _window_cache_path(self, query: Mapping[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()[:24]
        return _cache_root("airnow_api") / f"{digest}.parquet"

    def _is_settled(self, end: str) -> bool:
        end_ts = pd.Timestamp(datetime.strptime(end, _HOUR_FMT))
        return end_ts < pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=self.settle_hours)

    def _fetch_window(self, startDate: str, endDate: str, query: Dict[str, Any],
                      use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
        full = {**query, "startDate": startDate, "endDate": endDate}
        cache_path = None
        if use_cache and self._is_settled(endDate):
            cache_path = self._window_cache_path(full)
            if cache_path.exists() and not refresh:
                return pd.read_parquet(cache_path)

        payload = self._pull(**full).json()
        if isinstance(payload, dict):  # AirNow reports some errors as a JSON object with HTTP 200
            raise APIError(f"AirNow API error: {redact(json.dumps(payload)[:500], self._api_key)}")
        df = pd.DataFrame(payload)

        if cache_path is not None and not df.empty:
            tmp = cache_path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(cache_path)
        return df

    def _collect(self,
                 start_date: Optional[Union[str, date, datetime]],
                 end_date: Optional[Union[str, date, datetime]],
                 *,
                 max_workers: int = 3,
                 chunk: Union[str, timedelta] = "1D",
                 use_cache: bool = True,
                 refresh: bool = False,
                 silent: bool = False,
                 show_traceback: bool = False,
                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch every window on threads and return ``(DataFrame, query_params)``. All-or-nothing."""
        if (start_date is None) != (end_date is None):
            raise ValueError("Give both start_date and end_date, or neither (default: the last 24 hours).")
        base = AirNowParams(**kwargs)
        start = start_date if start_date is not None else base.startDate
        end = end_date if end_date is not None else base.endDate
        windows = make_windows(start, end, chunk, end_inclusive=self.window_end_inclusive)
        query = asdict(base)
        query.pop("startDate"), query.pop("endDate")
        self.api_key  # fail fast on a missing key, before any threads start

        def work(startDate: str, endDate: str):
            return {"key": startDate, "value": self._fetch_window(startDate, endDate, query, use_cache, refresh)}

        executor = ParallelExecutor(
            max_workers=max_workers, desc="Downloading AirNow data", mode="thread",
            show_traceback=show_traceback, progress=not silent, print_failures=False,
        )
        result: JobResult = executor(work)([{"startDate": s, "endDate": e} for s, e in windows])
        if result.failed:
            lines = [f"  {f.job['startDate']} -> {f.job['endDate']}: {redact(f.exc, self._api_key)}" for f in result.failed]
            raise DownloadError(
                f"{len(result.failed)} of {len(windows)} AirNow request(s) failed (nothing returned):\n" + "\n".join(lines),
                failed=[(f.job, f.exc) for f in result.failed],
            )

        frames = [result.succeeded[s] for s, _ in windows]  # chronological order
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df = df.drop_duplicates(ignore_index=True)  # safety net if AirNow's endDate semantics differ
        if df.empty:
            raise ValueError("AirNow returned no observations for this request.")
        return df, {**query, "startDate": windows[0][0], "endDate": windows[-1][1]}

    def import_data(self,
                    start_date: Optional[Union[str, date, datetime]] = None,
                    end_date: Optional[Union[str, date, datetime]] = None,
                    *,
                    max_workers: int = 3,
                    chunk: Union[str, timedelta] = "1D",
                    use_cache: bool = True,
                    refresh: bool = False,
                    silent: bool = False,
                    **kwargs):
        """
        Fetch observations for a date range and return them as an ``atmoz_dataset``.

        Parameters:
            start_date, end_date: Inclusive range. A bare date (``"2025-08-07"``) as *end_date*
                means "through 23:00 UTC that day"; pass a datetime or ``"2025-08-07T12"`` for an
                exact hour. ``start_date == end_date`` returns that single day. Omit both for
                the last 24 hours. Times are UTC.
            max_workers: Concurrent requests (default 3, be kind to the API).
            chunk: Size of each request window, any ``pd.Timedelta`` string (default ``"1D"``).
                Use smaller windows for a wide box and many pollutants.
            use_cache, refresh: Cache controls (see class docs).
            silent: No progress bar.
            **kwargs: :class:`AirNowParams` fields: ``parameters``, ``BBOX``, ``dataType``, ...

        Raises:
            DownloadError: if any window fails (no partial result).
            MissingCredentialError: if no API key can be found.
        """
        data, _ = self._collect(start_date, end_date, max_workers=max_workers, chunk=chunk,
                                use_cache=use_cache, refresh=refresh, silent=silent, **kwargs)
        metadata: pd.DataFrame = self._metadata(data)
        return atmoz_dataset(data=data, metadata=metadata, datatype="surface")

    def download_data(self,
                      start_date: Optional[Union[str, date, datetime]] = None,
                      end_date: Optional[Union[str, date, datetime]] = None,
                      output_dir: Optional[PathLike] = None,
                      max_workers: int = 3,
                      **kwargs) -> Path:
        """
        Fetch a date range (same arguments as :meth:`import_data`) and save it as one Parquet file.

        The query is stored in the file's ``attrs["query_params"]`` (the API key is not).

        Returns:
            Path: The Parquet file written to ``output_dir`` (default: the current directory).
        """
        df, query = self._collect(start_date, end_date, max_workers=max_workers, **kwargs)
        df.attrs = {"query_params": query}
        stamp = lambda s: s.replace(":", "")
        filename = f"airnow_api_query_{stamp(query['startDate'])}_{stamp(query['endDate'])}.parquet"
        folder = Path(output_dir) if output_dir else Path(".")
        folder.mkdir(parents=True, exist_ok=True)
        filepath = folder / filename
        df.to_parquet(filepath, index=False)
        return filepath

    @classmethod
    def download(cls, endpoint: str = "airnow", **kwargs) -> Union[Any, Dict[str, pd.DataFrame]]:
        """
        One entry point for both backends.

        * ``endpoint="airnow"`` -> ``AirNow(api_key=...).import_data(**kwargs)`` (an ``atmoz_dataset``)
        * ``endpoint="aqs"``    -> ``epa_pregen._download(**kwargs)`` (``{filename: DataFrame}``)

        .. note::
            The two endpoints still return different types; unifying them is planned
            before the API is frozen.
        """
        if endpoint == "airnow":
            api_key = kwargs.pop("api_key", None)
            return cls(api_key=api_key).import_data(**kwargs)
        elif endpoint == "aqs":
            return epa_pregen._download(**kwargs)
        raise ValueError(f"Unsupported endpoint: {endpoint}. Valid options are 'airnow' and 'aqs'.")
