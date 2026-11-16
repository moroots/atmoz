# -*- coding: utf-8 -*-
"""
UCN: the Unified Ceilometer Network (Hampton University), https://ucn.cas.hamptonu.edu/viewer/

Ceilometers are small ground-based lidars that measure attenuated backscatter (and so cloud base and
boundary-layer height). UCN publishes one NetCDF file per ceilometer per UTC day:

``L2``  10-minute attenuated backscatter profiles + boundary-layer / cloud retrievals (~1.4 MB a day). **The default.**
``L3``  hourly boundary-layer statistics (~1.3 MB a day, but its backscatter array duplicates L2's, so only the
        hourly table is kept).

(L0b, the raw 16 s profiles at ~33 MB a day, is not supported yet.)

* The cache holds **one compact parquet file per ceilometer-day** (about half the size of the NetCDF), not the NetCDF.
* ``download()`` returns DataFrames: one row per time step, one column per variable; profile variables
  (``attenuated_backscatter``, ``cloud_layers``, ...) hold one numpy array per row.
* Gentle on the server by design: 2 workers, a pause between requests, an ``atmoz/<version>`` User-Agent,
  retries with backoff, files fetched once and then served from the cache.
* Follows the package-wide process (see ``docs/conventions.md``):
  ``_validate -> _url -> _fetch_one -> _parse_one -> _assemble``. Requires ``pip install "atmoz[lidar]"``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from atmoz.exceptions import NoDataError, RequestTooLargeError
from atmoz.resources.atmoz_dataclasses import atmoz_dataset
from atmoz.resources.cache import _meta_path, _read_meta, _write_meta
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.cache import download_to_cache
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.utilities import Job, PathLike, as_list, default_session_handler, run_jobs

_BASE_URL = "https://ucn.cas.hamptonu.edu/viewer"
_LEVELS = {"l2": "L2", "l3": "L3"}
_NETCDF_MAGIC = (b"\x89HDF\r\n\x1a\n", b"CDF\x01", b"CDF\x02")
_META_KEY = b"atmoz_ucn"

# Rough sizes used to warn before a big pull (measured on ceilometer 9, Nov 2026).
EST_DOWNLOAD_MB = {"L2": 1.4, "L3": 1.3}
EST_MEMORY_MB = {"L2": 1.2, "L3": 0.01}
LARGE_DOWNLOAD_MB = 1000.0   # ~700 days of L2
LARGE_MEMORY_MB = 2000.0
_AVAILABILITY_MAX_AGE = timedelta(hours=6)

_gate = threading.Lock()
_netcdf_lock = threading.Lock()   # the HDF5 library under netCDF4 is not thread-safe: read one file at a time (downloads stay parallel)
_last_request = [0.0]


def _pace(pause: float) -> None:
    """Space out request *starts* across all threads (sleeping inside the lock is the point)."""
    if pause <= 0:
        return
    with _gate:
        wait = _last_request[0] + pause - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def _is_netcdf(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(8)
    return any(head.startswith(magic) for magic in _NETCDF_MAGIC)


def _to_date(value: Union[str, date, datetime, pd.Timestamp]) -> date:
    return pd.Timestamp(value).date()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class UCN:
    """Unified Ceilometer Network: ``UCN.download(9, "2026-11-01")`` returns L2 as DataFrames."""

    levels = tuple(_LEVELS.values())
    planned_levels = {"L0b": "raw 16 s profiles, about 33 MB per ceilometer-day (planned, not supported yet)"}

    # ---- 1. validate (request -> jobs) -------------------------------------------
    @classmethod
    def _validate(cls,
                  ceilometer: Union[int, str, List[Union[int, str]]],
                  start_date: Union[str, date, datetime],
                  end_date: Union[str, date, datetime, None] = None,
                  level: Union[str, List[str]] = "L2") -> List[Dict[str, Any]]:
        """Expand ceilometer x day x level into jobs ``{"ceilometer", "date", "level"}`` (no duplicates).

        ``end_date`` defaults to ``start_date`` (one day, inclusive). Unknown levels raise ``ValueError``.
        """
        ceilos = [int(c) for c in as_list(ceilometer)]
        levels = []
        for lv in as_list(level):
            key = str(lv).strip().lower()
            if key == "l0b":
                raise ValueError("L0b (raw 16 s profiles, ~33 MB a day) is planned but not supported yet; "
                                 f"use {' or '.join(cls.levels)} for now (see UCN.planned_levels).")
            try:
                levels.append(_LEVELS[key])
            except KeyError:
                raise ValueError(f"Unknown level {lv!r}; choose from {list(cls.levels)}.") from None
        start = _to_date(start_date)
        end = _to_date(end_date) if end_date is not None else start
        if end < start:
            raise ValueError(f"end_date {end} is before start_date {start}.")
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        jobs, seen = [], set()
        for c in ceilos:
            for d in days:
                for lv in levels:
                    if (c, d, lv) not in seen:
                        seen.add((c, d, lv))
                        jobs.append({"ceilometer": c, "date": d, "level": lv})
        return jobs

    @staticmethod
    def _name(job: Job) -> str:
        """Result / cache name, e.g. ``CEILO-9_20261101_L2`` (the cache file adds ``.parquet``)."""
        return f"CEILO-{job['ceilometer']}_{job['date']:%Y%m%d}_{job['level']}"

    @staticmethod
    def _cache_dir(cache_dir: Optional[PathLike]) -> Path:
        return Path(cache_dir) if cache_dir else _cache_root("ucn")

    # ---- 2. url ----------------------------------------------------------------
    @staticmethod
    def _url(job: Job) -> str:
        return f"{_BASE_URL}/pbl_plots.php?" + urlencode({
            "action": "netcdf_download", "ceilometer": job["ceilometer"],
            "date": f"{job['date']:%Y-%m-%d}", "level": job["level"][1:],
        })

    @staticmethod
    def _default_max_age(job: Job) -> Optional[timedelta]:
        # The newest days are still being processed; older days are final.
        recent = job["date"] >= datetime.now(timezone.utc).date() - timedelta(days=2)
        return timedelta(hours=1) if recent else None

    # ---- 3. fetch (network -> parquet file in the cache) -------------------------
    @classmethod
    def _fetch_one(cls, job: Job, *, dest_dir: Path, refresh: bool = False,
                   max_age: Union[str, timedelta, None] = "auto", pause: float = 0.5,
                   session_handler: Optional[SessionHandler] = None) -> Path:
        """Make sure the compact parquet file for *job* is in *dest_dir* and return its path.

        A current cached file means no network. Otherwise the NetCDF is downloaded once (never kept), converted
        to parquet, and the cache keeps only the parquet. A day UCN has no file for raises
        :class:`~atmoz.exceptions.NoDataError` and caches nothing.
        """
        age = cls._default_max_age(job) if max_age == "auto" else max_age
        dest = dest_dir / f"{cls._name(job)}.parquet"
        if dest.exists() and not refresh:
            fetched = _read_meta(dest).get("fetched_at", dest.stat().st_mtime)
            if age is None or (time.time() - fetched) < age.total_seconds():
                return dest
        legacy = dest_dir / f"{cls._name(job)}.nc"      # a NetCDF left by an earlier atmoz: convert, no network
        if legacy.exists() and not refresh:
            cls._convert(job, legacy, dest)
            legacy.unlink(missing_ok=True)
            _meta_path(legacy).unlink(missing_ok=True)
            return dest

        dest_dir.mkdir(parents=True, exist_ok=True)
        _pace(pause)
        tmp = Path(tempfile.mkdtemp(prefix=".ucn_", dir=dest_dir))
        try:
            nc = tmp / "file.nc"
            try:
                download_to_cache(cls._url(job), nc, (session_handler or default_session_handler()).session(),
                                  refresh=True)
            except requests.HTTPError as exc:            # a day with no file answers 404
                if getattr(exc.response, "status_code", None) == 404 or str(exc).startswith("404"):
                    raise NoDataError(f"No {job['level']} file for ceilometer {job['ceilometer']} "
                                      f"on {job['date']} (404).") from exc
                raise
            if not _is_netcdf(nc):                       # ...or an ordinary page saying so
                note = " ".join(re.sub(r"<[^>]+>", " ", nc.read_text(errors="replace")).split())[:120] or "empty response"
                raise NoDataError(f"No {job['level']} file for ceilometer {job['ceilometer']} on {job['date']} ({note}).")
            cls._convert(job, nc, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return dest

    @classmethod
    def _convert(cls, job: Job, nc_path: Path, dest: Path) -> None:
        """NetCDF -> one parquet file (atomic). Profile variables become list columns; labels go in the metadata."""
        import xarray as xr

        with _netcdf_lock, xr.open_dataset(nc_path) as raw:
            ds = raw.load()
        table = cls._to_table(job, ds)
        part = dest.with_name(f"{dest.name}.{os.getpid()}-{threading.get_ident()}.part")
        try:
            pq.write_table(table, part, compression="zstd", compression_level=3,
                           use_dictionary=False, use_byte_stream_split=True)
            os.replace(part, dest)
        finally:
            part.unlink(missing_ok=True)
        _write_meta(dest, {"url": cls._url(job), "fetched_at": time.time(), "bytes": dest.stat().st_size})

    @staticmethod
    def _to_table(job: Job, ds) -> pa.Table:
        """L2: one row per 10-minute step. L3: one row per hour (its duplicate backscatter array is dropped)."""
        level = job["level"]
        tdim = "time" if level == "L2" else "hourly_time"
        cols: Dict[str, Any] = {"time": pa.array(ds[tdim].values)}
        labels: Dict[str, Any] = {}
        units: Dict[str, str] = {}
        for name, da in ds.data_vars.items():
            if da.ndim == 0 or tdim not in da.dims or (level == "L3" and name == "attenuated_backscatter"):
                continue
            if "units" in da.attrs:
                units[name] = da.attrs["units"]
            if da.ndim == 1:
                if level == "L3" and name == tdim:
                    continue
                cols[name] = pa.array(da.values)
                continue
            other = [d for d in da.dims if d != tdim][0]
            arr = np.ascontiguousarray(da.transpose(tdim, other).values)
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel()), arr.shape[1])
            if other in ds.coords and other != "altitude":      # heights are stored once, as altitude_m
                labels[name] = ds[other].values.tolist()
        meta = {
            "ceilometer": job["ceilometer"], "level": level, "date": f"{job['date']:%Y-%m-%d}",
            "latitude": float(ds["latitude"]) if "latitude" in ds else None,
            "longitude": float(ds["longitude"]) if "longitude" in ds else None,
            "altitude_m": ds["altitude"].values.tolist() if "altitude" in ds.coords and level == "L2" else None,
            "list_labels": labels, "units": units, "attrs": dict(ds.attrs),
            "note": "L3 backscatter is identical to L2's and is not stored" if level == "L3" else None,
        }
        return pa.table(cols).replace_schema_metadata({_META_KEY: json.dumps(meta, default=_jsonable).encode()})

    # ---- 4. parse (parquet file -> DataFrame) ------------------------------------
    @classmethod
    def _parse_one(cls, job: Job, path: Path) -> pd.DataFrame:
        """Read one parquet file into a DataFrame indexed by UTC ``time``; labels and attributes are in ``df.attrs``."""
        table = pq.read_table(path)
        meta = json.loads((table.schema.metadata or {}).get(_META_KEY, b"{}"))
        df = table.to_pandas().set_index("time")
        df.attrs.update(meta)
        return df

    # ---- fetch + parse for one job ---------------------------------------------
    @classmethod
    def _load_one(cls, job: Job, *, skip_missing: bool = False, use_cache: bool = True, refresh: bool = False,
                  max_age: Union[str, timedelta, None] = "auto", cache_dir: Optional[PathLike] = None,
                  pause: float = 0.5, session_handler: Optional[SessionHandler] = None) -> Optional[pd.DataFrame]:
        tmp = None
        try:
            if use_cache:
                dest_dir = cls._cache_dir(cache_dir)
            else:
                tmp = tempfile.mkdtemp(prefix="atmoz_ucn_")
                dest_dir = Path(tmp)
            try:
                path = cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh or not use_cache, max_age=max_age,
                                      pause=pause, session_handler=session_handler)
            except NoDataError:
                if skip_missing:
                    return None
                raise
            return cls._parse_one(job, path)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    # ---- 5. assemble -------------------------------------------------------------
    @staticmethod
    def _assemble(results: Dict[str, Optional[pd.DataFrame]], unlisted: Optional[List[str]] = None) -> atmoz_dataset:
        found = {name: df for name, df in results.items() if df is not None}
        if not found:
            raise NoDataError(f"UCN has none of the {len(results) + len(unlisted or [])} requested file(s).")
        missing = [n for n, df in results.items() if df is None] + list(unlisted or [])
        meta = {"endpoint": "ucn", "files": list(found), "missing": missing}
        return atmoz_dataset(data=found, metadata=meta, datatype="lidar")

    # ---- availability, size estimates ------------------------------------------------
    @classmethod
    def _listed_days(cls, ceilometer: int, *, dest_dir: Path, refresh: bool = False, pause: float = 0.5,
                     session_handler: Optional[SessionHandler] = None) -> Optional[set]:
        """Days the viewer lists for a ceilometer (one 40 KB page, cached 6 h), or ``None`` if it can't be read."""
        cache = dest_dir / f"availability_{ceilometer}.json"
        if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < _AVAILABILITY_MAX_AGE.total_seconds():
            try:
                return set(json.loads(cache.read_text()))
            except ValueError:
                pass
        _pace(pause)
        try:
            html = (session_handler or default_session_handler()).session().get(
                f"{_BASE_URL}/?" + urlencode({"ceilometer": ceilometer, "date": f"{datetime.now(timezone.utc):%Y-%m-%d}"})
            ).text
            found = re.search(r"availableDatesJson\s*=\s*(\[.*?\])\s*;", html, re.S)
            days = json.loads(found.group(1)) if found else None
        except (requests.RequestException, ValueError):
            days = None
        if not days:
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(days)))
        return set(days)

    @classmethod
    def available(cls,
                  ceilometer: Union[int, str, List[Union[int, str]]],
                  start_date: Union[str, date, datetime],
                  end_date: Union[str, date, datetime, None] = None,
                  *,
                  level: Union[str, List[str]] = "L2",
                  cache_dir: Optional[PathLike] = None,
                  pause: float = 0.5,
                  session_handler: Optional[SessionHandler] = None) -> pd.DataFrame:
        """What a request would fetch, without downloading any data: one row per (ceilometer, day, level).

        Columns: ``ceilometer``, ``date``, ``level``, ``name``, ``listed`` (does the UCN viewer list that day?
        ``None`` if its list could not be read), ``cached`` (already on disk, so free), ``est_download_mb``
        and ``est_memory_mb`` (zero when cached / for download). Costs one 40 KB page per ceilometer (cached 6 h).
        """
        jobs = cls._validate(ceilometer, start_date, end_date, level)
        dest_dir = cls._cache_dir(cache_dir)
        listed = {c: cls._listed_days(c, dest_dir=dest_dir, pause=pause, session_handler=session_handler)
                  for c in sorted({j["ceilometer"] for j in jobs})}
        rows = []
        for j in jobs:
            days = listed[j["ceilometer"]]
            cached = (dest_dir / f"{cls._name(j)}.parquet").exists()
            rows.append({
                "ceilometer": j["ceilometer"], "date": j["date"], "level": j["level"], "name": cls._name(j),
                "listed": None if days is None else f"{j['date']:%Y-%m-%d}" in days, "cached": cached,
                "est_download_mb": 0.0 if cached else EST_DOWNLOAD_MB[j["level"]],
                "est_memory_mb": EST_MEMORY_MB[j["level"]],
            })
        return pd.DataFrame(rows)

    @classmethod
    def _check_size(cls, jobs: List[Dict[str, Any]], dest_dir: Path, *, refresh: bool, allow_large: bool,
                    memory: bool = True) -> None:
        todo = [j for j in jobs if refresh or not (dest_dir / f"{cls._name(j)}.parquet").exists()]
        dl = sum(EST_DOWNLOAD_MB[j["level"]] for j in todo)
        mem = sum(EST_MEMORY_MB[j["level"]] for j in jobs) if memory else 0.0
        if allow_large or (dl <= LARGE_DOWNLOAD_MB and mem <= LARGE_MEMORY_MB):
            return
        raise RequestTooLargeError(
            f"This request is {len(jobs)} file(s): about {dl / 1000:.1f} GB to download from UCN and "
            f"{mem / 1000:.1f} GB in memory (limits {LARGE_DOWNLOAD_MB / 1000:.0f} GB / {LARGE_MEMORY_MB / 1000:.0f} GB). "
            "Use UCN.available(...) to preview it, split it up, use UCN.fetch(...), or pass allow_large=True."
        )

    # ---- public --------------------------------------------------------------------
    @classmethod
    def download(cls,
                 ceilometer: Union[int, str, List[Union[int, str]]],
                 start_date: Union[str, date, datetime],
                 end_date: Union[str, date, datetime, None] = None,
                 *,
                 level: Union[str, List[str]] = "L2",
                 skip_missing: bool = False,
                 allow_large: bool = False,
                 max_workers: int = 2,
                 pause: float = 0.5,
                 use_cache: bool = True,
                 refresh: bool = False,
                 max_age: Union[str, timedelta, None] = "auto",
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False,
                 session_handler: Optional[SessionHandler] = None) -> atmoz_dataset:
        """Download (once, then cached) and read every (ceilometer, day, level) file as a DataFrame.

        Parameters
        ----------
        ceilometer:
            One or more ceilometer numbers (see :meth:`sites`), e.g. ``9`` for McMillan Reservoir, DC.
        start_date, end_date:
            UTC days, both inclusive. ``end_date`` defaults to ``start_date``.
        level:
            ``"L2"`` (default, 10-minute backscatter + retrievals), ``"L3"`` (hourly statistics) or a list.
        skip_missing:
            A day with no file raises by default. With ``skip_missing=True`` those days are left out and listed in
            ``metadata["missing"]``; days the UCN viewer does not list are skipped without any request.
        allow_large:
            Requests of more than about 1 GB to download or 2 GB in memory raise
            :class:`~atmoz.exceptions.RequestTooLargeError` unless this is ``True``. Preview with :meth:`available`.
        max_workers, pause:
            Concurrent downloads (default 2, kept low on purpose) and the minimum seconds between request starts.
        use_cache, refresh, max_age, cache_dir:
            File-cache controls. ``max_age="auto"`` re-downloads the newest two days hourly (still being processed)
            and treats older days as final. ``refresh=True`` forces a download.
        silent:
            No progress bar.

        Returns
        -------
        atmoz_dataset
            ``datatype="lidar"``; ``.data == {"CEILO-9_20261101_L2": DataFrame, ...}``. Each DataFrame is indexed by
            UTC ``time``; its ``.attrs`` hold the altitudes, list labels, units and the file's attributes.

        Raises
        ------
        DownloadError
            If any file fails or is missing (without ``skip_missing``). No partial result is returned.
        RequestTooLargeError
            See ``allow_large``.
        """
        jobs = cls._validate(ceilometer, start_date, end_date, level)
        dest_dir = cls._cache_dir(cache_dir)
        cls._check_size(jobs, dest_dir, refresh=refresh or not use_cache, allow_large=allow_large)
        unlisted: List[str] = []
        if skip_missing:
            def is_cached(j):
                return not (refresh or not use_cache) and (dest_dir / f"{cls._name(j)}.parquet").exists()
            # only days that would need a download are checked against the viewer's list (no request if all are cached)
            listed = {c: cls._listed_days(c, dest_dir=dest_dir, pause=pause, session_handler=session_handler)
                      for c in sorted({j["ceilometer"] for j in jobs if not is_cached(j)})}
            keep = []
            for j in jobs:
                days = listed.get(j["ceilometer"])
                if not is_cached(j) and days is not None and f"{j['date']:%Y-%m-%d}" not in days:
                    unlisted.append(cls._name(j))
                else:
                    keep.append(j)
            jobs = keep
        results = run_jobs(
            lambda job: cls._load_one(job, skip_missing=skip_missing, use_cache=use_cache, refresh=refresh,
                                      max_age=max_age, cache_dir=cache_dir, pause=pause, session_handler=session_handler),
            jobs, key=cls._name, describe=cls._name, noun="download", desc="Downloading UCN files",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        ) if jobs else {}
        return cls._assemble(results, unlisted)

    @classmethod
    def fetch(cls,
              ceilometer: Union[int, str, List[Union[int, str]]],
              start_date: Union[str, date, datetime],
              end_date: Union[str, date, datetime, None] = None,
              *,
              level: Union[str, List[str]] = "L2",
              allow_large: bool = False,
              max_workers: int = 2,
              pause: float = 0.5,
              refresh: bool = False,
              max_age: Union[str, timedelta, None] = "auto",
              cache_dir: Optional[PathLike] = None,
              silent: bool = False,
              show_traceback: bool = False,
              session_handler: Optional[SessionHandler] = None) -> Dict[str, Path]:
        """Download only (cached) and return ``{name: path}`` of the compact parquet files. Read them with
        :func:`pandas.read_parquet` (list columns come back as arrays) or :meth:`UCN._parse_one`."""
        jobs = cls._validate(ceilometer, start_date, end_date, level)
        dest_dir = cls._cache_dir(cache_dir)
        cls._check_size(jobs, dest_dir, refresh=refresh, allow_large=allow_large, memory=False)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return run_jobs(
            lambda job: cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh, max_age=max_age, pause=pause,
                                       session_handler=session_handler),
            jobs, key=cls._name, describe=cls._name, noun="download", desc="Downloading UCN files",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )

    @staticmethod
    def sites(*, session_handler: Optional[SessionHandler] = None) -> pd.DataFrame:
        """The network's ceilometers as a DataFrame with columns ``ceilometer`` (int) and ``site`` (text)."""
        session = (session_handler or default_session_handler()).session()
        html = session.get(f"{_BASE_URL}/").text
        rows = re.findall(r'<option value="(\d+)"[^>]*>\s*Ceilometer\s+\d+\s*-\s*([^<]+?)\s*</option>', html)
        if not rows:
            raise NoDataError("Could not read the ceilometer list from the UCN viewer page.")
        return pd.DataFrame({"ceilometer": [int(n) for n, _ in rows], "site": [s for _, s in rows]})

    # ---- helpers for the returned frames ---------------------------------------------
    @staticmethod
    def stack(ds: atmoz_dataset, ceilometer: int, level: str = "L2") -> pd.DataFrame:
        """All days of one ceilometer and level in *ds*, joined into one time-sorted DataFrame."""
        level = _LEVELS[str(level).strip().lower()]
        parts = [df for name, df in sorted(ds.data.items()) if df.attrs.get("ceilometer") == ceilometer
                 and df.attrs.get("level") == level]
        if not parts:
            raise KeyError(f"No L{level[1:]} data for ceilometer {ceilometer} in this dataset.")
        out = pd.concat(parts).sort_index()
        out.attrs.update({k: v for k, v in parts[0].attrs.items() if k not in ("date", "attrs")})
        out.attrs["dates"] = [p.attrs.get("date") for p in parts]
        return out

    @staticmethod
    def backscatter(df: pd.DataFrame):
        """The ``attenuated_backscatter`` list column as an ``xarray.DataArray`` with ``time`` and ``altitude`` (m)."""
        import xarray as xr

        arr = np.stack(df["attenuated_backscatter"].to_numpy())
        return xr.DataArray(arr, dims=("time", "altitude"),
                            coords={"time": df.index.values, "altitude": np.asarray(df.attrs["altitude_m"])},
                            name="attenuated_backscatter", attrs={"units": df.attrs.get("units", {}).get("attenuated_backscatter", "")})
