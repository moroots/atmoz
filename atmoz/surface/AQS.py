# -*- coding: utf-8 -*-
"""
EPA AQS pre-generated files: https://aqs.epa.gov/aqsweb/airdata/

* Files stream to a disk cache (never buffered in memory) and are parsed with Apache
  Arrow's multi-threaded CSV reader.
* :class:`EPA_PreGen` follows the package-wide process (see ``docs/conventions.md``):
  ``_validate -> _url -> _fetch_one -> _parse_one -> _assemble``, with ``download()`` and ``fetch()``
  as the public entry points.
"""
from __future__ import annotations

import csv
import difflib
import shutil
import tempfile
import warnings
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from atmoz.resources.atmoz_dataclasses import atmoz_dataset
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


class EPA_PreGen:
    """Download and parse the EPA pre-generated AQS files.

    Public entry points:

    * :meth:`download` -> :class:`~atmoz.atmoz_dataset` (``.data`` is ``{csv_name: DataFrame}``)
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

    # ---- 4. parse (file on disk -> DataFrame) ----------------------------------
    @classmethod
    def _parse_one(cls, job: Job, path: Path, *, columns: Optional[Sequence[str]] = None,
                   bbox: Optional[BBox] = None) -> pd.DataFrame:
        return read_aqs_zip(path, cls._name(job), columns=columns, bbox=bbox)

    # ---- fetch + parse for one job ---------------------------------------------
    @classmethod
    def _load_one(cls, job: Job, *, columns=None, bbox=None, use_cache: bool = True, refresh: bool = False,
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
            return cls._parse_one(job, path, columns=columns, bbox=bbox)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    # ---- 5. assemble -------------------------------------------------------------
    @staticmethod
    def _assemble(frames: Dict[str, pd.DataFrame]) -> atmoz_dataset:
        meta = {"endpoint": "aqs", "files": list(frames), "rows": {name: len(df) for name, df in frames.items()}}
        return atmoz_dataset(data=frames, metadata=meta, datatype="surface")

    # ---- public --------------------------------------------------------------------
    @classmethod
    def download(cls,
                 parameters: Union[str, List[str]],
                 resolutions: Union[str, List[str]],
                 years: Union[int, List[int]],
                 *,
                 columns: Optional[Sequence[str]] = None,
                 bbox: Optional[BBox] = None,
                 max_workers: int = 5,
                 use_cache: bool = True,
                 refresh: bool = False,
                 max_age: Union[str, timedelta, None] = "auto",
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False,
                 session_handler: Optional[SessionHandler] = None) -> atmoz_dataset:
        """Download and parse every valid (parameter, resolution, year) combination.

        Parameters
        ----------
        parameters, resolutions, years:
            One or more of each. Parameters may be names (``"ozone"``) or AQS codes.
        columns:
            Load only these columns. Default: everything.
        bbox:
            ``(min_lon, min_lat, max_lon, max_lat)``; keep only rows inside the box.
        max_workers:
            Download threads (default 5).
        use_cache, refresh, max_age, cache_dir:
            File-cache controls. ``max_age="auto"`` revalidates the current and previous
            year's files daily (cheap ``ETag`` check) and treats older years as frozen.
            ``refresh=True`` forces a fresh download; ``use_cache=False`` keeps nothing on disk.
        silent:
            No progress bar and no warnings.

        Returns
        -------
        atmoz_dataset
            ``.data == {"hourly_44201_2025.csv": DataFrame, ...}``

        Raises
        ------
        DownloadError
            If any file fails. No partial result is returned.
        """
        jobs = cls._validate(parameters, resolutions, years, silent=silent)
        frames = run_jobs(
            lambda job: cls._load_one(job, columns=columns, bbox=bbox, use_cache=use_cache, refresh=refresh,
                                      max_age=max_age, cache_dir=cache_dir, session_handler=session_handler),
            jobs, key=cls._name, describe=cls._name, noun="download", desc="Downloading EPA AQS data",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
        return cls._assemble(frames)

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
