# -*- coding: utf-8 -*-
"""
UCN: the Unified Ceilometer Network (Hampton University), https://ucn.cas.hamptonu.edu/viewer/

Ceilometers are small ground-based lidars that measure attenuated backscatter (and so cloud base and
boundary-layer height) in 10 s profiles, 10 m resolution, up to 15.4 km.

* One NetCDF file per ceilometer per UTC day, in three levels:

  ``L0b``  standardized backscatter profiles (``time`` x ``altitude``) + instrument housekeeping.
  ``L2``   10-minute boundary-layer / cloud retrievals (``nsl_ml_height``, ``rl_height``, ``cloud_layers`` ...).
  ``L3``   hourly boundary-layer statistics.

* Follows the package-wide process (see ``docs/conventions.md``):
  ``_validate -> _url -> _fetch_one -> _parse_one -> _assemble``, with ``download()`` and ``fetch()`` as the public
  entry points. Requires ``pip install "atmoz[lidar]"``.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import pandas as pd
import requests
import xarray as xr

from atmoz.exceptions import NoDataError
from atmoz.resources.atmoz_dataclasses import atmoz_dataset
from atmoz.resources.cache import _meta_path
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.cache import download_to_cache
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.utilities import Job, PathLike, as_list, default_session_handler, run_jobs

_BASE_URL = "https://ucn.cas.hamptonu.edu/viewer"
_LEVELS = {"l0b": "L0b", "l2": "L2", "l3": "L3"}
_NETCDF_MAGIC = (b"\x89HDF\r\n\x1a\n", b"CDF\x01", b"CDF\x02")


def _is_netcdf(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(8)
    return any(head.startswith(magic) for magic in _NETCDF_MAGIC)


def _to_date(value: Union[str, date, datetime, pd.Timestamp]) -> date:
    return pd.Timestamp(value).date()


class UCN:
    """Unified Ceilometer Network files: ``UCN.download(ceilometer=9, start_date="2026-11-14")``."""

    levels = tuple(_LEVELS.values())

    # ---- 1. validate (request -> jobs) -------------------------------------------
    @classmethod
    def _validate(cls,
                  ceilometer: Union[int, str, List[Union[int, str]]],
                  start_date: Union[str, date, datetime],
                  end_date: Union[str, date, datetime, None] = None,
                  level: Union[str, List[str]] = "L0b") -> List[Dict[str, Any]]:
        """Expand ceilometer x day x level into jobs ``{"ceilometer", "date", "level"}`` (no duplicates).

        ``end_date`` defaults to ``start_date`` (one day, inclusive). Unknown levels raise ``ValueError``.
        """
        ceilos = [int(c) for c in as_list(ceilometer)]
        levels = []
        for lv in as_list(level):
            try:
                levels.append(_LEVELS[str(lv).strip().lower()])
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
        """Stable cache / result name, e.g. ``CEILO-9_20261114_L0b.nc`` (the server's own names change as files grow)."""
        return f"CEILO-{job['ceilometer']}_{job['date']:%Y%m%d}_{job['level']}.nc"

    # ---- 2. url ----------------------------------------------------------------
    @staticmethod
    def _url(job: Job) -> str:
        day = f"{job['date']:%Y-%m-%d}"
        if job["level"] == "L0b":
            return f"{_BASE_URL}/download_file.php?" + urlencode({"ceilometer": job["ceilometer"], "date": day})
        return f"{_BASE_URL}/pbl_plots.php?" + urlencode({
            "action": "netcdf_download", "ceilometer": job["ceilometer"], "date": day, "level": job["level"][1:],
        })

    @staticmethod
    def _default_max_age(job: Job) -> Optional[timedelta]:
        # The newest days are still being written / processed; older days are frozen.
        recent = job["date"] >= datetime.now(timezone.utc).date() - timedelta(days=2)
        return timedelta(hours=1) if recent else None

    # ---- 3. fetch (network -> file on disk) ------------------------------------
    @classmethod
    def _fetch_one(cls, job: Job, *, dest_dir: Path, refresh: bool = False,
                   max_age: Union[str, timedelta, None] = "auto",
                   session_handler: Optional[SessionHandler] = None) -> Path:
        """Stream one NetCDF into *dest_dir* (skipped if the cached copy is current). Returns its path.

        The server answers "no file" with an ordinary HTTP 200 and an HTML page, so the body is checked: a
        non-NetCDF answer is deleted (never cached) and raises :class:`~atmoz.exceptions.NoDataError`.
        """
        age = cls._default_max_age(job) if max_age == "auto" else max_age
        dest = dest_dir / cls._name(job)
        try:
            path = download_to_cache(cls._url(job), dest, (session_handler or default_session_handler()).session(),
                                     max_age=age, refresh=refresh)
        except requests.HTTPError as exc:  # L2 / L3 answer a missing day with a plain 404
            if getattr(exc.response, "status_code", None) == 404 or str(exc).startswith("404"):
                raise NoDataError(f"No {job['level']} file for ceilometer {job['ceilometer']} on {job['date']} (404).") from exc
            raise
        if not _is_netcdf(path):
            note = re.sub(r"<[^>]+>", " ", path.read_text(errors="replace"))
            note = " ".join(note.split())[:120] or "empty response"
            path.unlink(missing_ok=True)
            _meta_path(path).unlink(missing_ok=True)
            raise NoDataError(f"No {job['level']} file for ceilometer {job['ceilometer']} on {job['date']} ({note}).")
        return path

    # ---- 4. parse (file on disk -> xarray.Dataset) -----------------------------
    @classmethod
    def _parse_one(cls, job: Job, path: Path) -> xr.Dataset:
        """Open one file fully into memory (the file handle is released) with ``time`` and ``altitude`` coordinates.

        L0b files carry raw ``time_UTC`` (epoch seconds) and ``elevation_m`` arrays; they become the same
        ``time`` / ``altitude`` coordinates that L2 and L3 already use. Variable names are left as published.
        """
        with xr.open_dataset(path) as raw:
            ds = raw.load()
        if "time_dim" in ds.dims:  # L0b
            ds = ds.rename({"time_dim": "time", "range_dim": "altitude"})
            ds = ds.assign_coords(
                time=("time", pd.to_datetime(ds["time_UTC"].values, unit="s")),
                altitude=("altitude", ds["elevation_m"].values, {"units": "m"}),
            ).drop_vars(["time_UTC", "elevation_m"])
        ds.attrs.update(ucn_ceilometer=job["ceilometer"], ucn_level=job["level"], ucn_source=cls._url(job))
        return ds

    # ---- fetch + parse for one job ---------------------------------------------
    @classmethod
    def _load_one(cls, job: Job, *, skip_missing: bool = False, use_cache: bool = True, refresh: bool = False,
                  max_age: Union[str, timedelta, None] = "auto", cache_dir: Optional[PathLike] = None,
                  session_handler: Optional[SessionHandler] = None) -> Optional[xr.Dataset]:
        tmp = None
        try:
            if use_cache:
                dest_dir = Path(cache_dir) if cache_dir else _cache_root("ucn")
            else:
                tmp = tempfile.mkdtemp(prefix="atmoz_ucn_")
                dest_dir = Path(tmp)
            try:
                path = cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh or not use_cache,
                                      max_age=max_age, session_handler=session_handler)
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
    def _assemble(results: Dict[str, Optional[xr.Dataset]]) -> atmoz_dataset:
        found = {name: ds for name, ds in results.items() if ds is not None}
        if not found:
            raise NoDataError(f"UCN has none of the {len(results)} requested file(s).")
        meta = {"endpoint": "ucn", "files": list(found), "missing": [n for n, ds in results.items() if ds is None]}
        return atmoz_dataset(data=found, metadata=meta, datatype="lidar")

    # ---- public --------------------------------------------------------------------
    @classmethod
    def download(cls,
                 ceilometer: Union[int, str, List[Union[int, str]]],
                 start_date: Union[str, date, datetime],
                 end_date: Union[str, date, datetime, None] = None,
                 *,
                 level: Union[str, List[str]] = "L0b",
                 skip_missing: bool = False,
                 max_workers: int = 4,
                 use_cache: bool = True,
                 refresh: bool = False,
                 max_age: Union[str, timedelta, None] = "auto",
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False,
                 session_handler: Optional[SessionHandler] = None) -> atmoz_dataset:
        """Download and open every (ceilometer, day, level) file.

        Parameters
        ----------
        ceilometer:
            One or more ceilometer numbers (see :meth:`sites`), e.g. ``9`` for McMillan Reservoir, DC.
        start_date, end_date:
            UTC days, both inclusive. ``end_date`` defaults to ``start_date``.
        level:
            ``"L0b"`` (default), ``"L2"``, ``"L3"``, or a list of them.
        skip_missing:
            A day with no file (a gap in the network, or L2/L3 not processed yet) raises by default. With
            ``skip_missing=True`` those are left out and listed in ``metadata["missing"]``.
        max_workers:
            Download threads (default 4).
        use_cache, refresh, max_age, cache_dir:
            File-cache controls. ``max_age="auto"`` re-downloads the last 2 days hourly (they are still being
            written) and treats older days as frozen. ``refresh=True`` forces a fresh download.
        silent:
            No progress bar.

        Returns
        -------
        atmoz_dataset
            ``datatype="lidar"``; ``.data == {"CEILO-9_20261114_L0b.nc": xarray.Dataset, ...}``.

        Raises
        ------
        DownloadError
            If any file fails or is missing (without ``skip_missing``). No partial result is returned.
        """
        jobs = cls._validate(ceilometer, start_date, end_date, level)
        results = run_jobs(
            lambda job: cls._load_one(job, skip_missing=skip_missing, use_cache=use_cache, refresh=refresh,
                                      max_age=max_age, cache_dir=cache_dir, session_handler=session_handler),
            jobs, key=cls._name, describe=cls._name, noun="download", desc="Downloading UCN files",
            max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
        return cls._assemble(results)

    @classmethod
    def fetch(cls,
              ceilometer: Union[int, str, List[Union[int, str]]],
              start_date: Union[str, date, datetime],
              end_date: Union[str, date, datetime, None] = None,
              *,
              level: Union[str, List[str]] = "L0b",
              max_workers: int = 4,
              refresh: bool = False,
              max_age: Union[str, timedelta, None] = "auto",
              cache_dir: Optional[PathLike] = None,
              silent: bool = False,
              show_traceback: bool = False,
              session_handler: Optional[SessionHandler] = None) -> Dict[str, Path]:
        """Download the NetCDF files only (cached) and return ``{name: path}``. Open them with ``xarray`` yourself."""
        jobs = cls._validate(ceilometer, start_date, end_date, level)
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("ucn")
        dest_dir.mkdir(parents=True, exist_ok=True)
        return run_jobs(
            lambda job: cls._fetch_one(job, dest_dir=dest_dir, refresh=refresh, max_age=max_age,
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
