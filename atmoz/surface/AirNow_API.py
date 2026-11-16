# -*- coding: utf-8 -*-
"""
The AirNow web API: https://docs.airnowapi.org/

* Requested one window (default: one day) at a time on a thread pool. A request for the whole
  country, all pollutants and many days in one go gets blocked by AirNow, so you choose the
  window size (``chunk=``), the pollutants and the box.
* :class:`AirNowAPI` follows the package-wide process (see ``docs/conventions.md``):
  ``_validate -> _url -> _fetch_one -> _parse_one -> _assemble``, with ``download()`` and ``fetch()``
  as the public entry points.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import pandas as pd
import requests

from atmoz.exceptions import APIError
from atmoz.resources import credentials
from atmoz.resources.atmoz_dataclasses import atmoz_dataset
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.urls import build_url, redact
from atmoz.resources.utilities import Job, PathLike, default_session_handler, run_jobs

_BASE_URL = "https://www.airnowapi.org/aq/data"

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


class AirNowAPI:
    """
    Hourly observations from the AirNow API.

    Public entry points:

    * :meth:`download` -> :class:`~atmoz.atmoz_dataset` (``.data`` is ``{name: DataFrame}`` (one frame), ``.metadata["sites"]`` a per-site table)
    * :meth:`fetch`    -> ``{filename: path}`` (the whole query saved as one Parquet file)

    Notes:
        - API key from the ``api_key`` argument, the ``ATMOZ_AIRNOW_API_KEY`` environment
          variable, or a ``.atmoz.env`` file (save it once with :meth:`set_api_key`).
        - Finished windows older than ``settle_hours`` are cached on disk.
        - All-or-nothing: if any window fails, :class:`~atmoz.exceptions.DownloadError` is
          raised and nothing is returned. Completed windows stay cached.
        - The API key never appears in exceptions or reprs.

    Attributes:
        base_url (str): The AirNow endpoint.
        window_end_inclusive (bool): Whether AirNow returns the ``endDate`` hour. ``True``
            (the default) makes request windows end one hour before the next begins.
        settle_hours (int): Windows that end less than this many hours ago are never cached,
            because AirNow may still revise recent observations.
    """

    base_url = _BASE_URL
    window_end_inclusive = True
    settle_hours = 72

    def __init__(self, api_key: Optional[str] = None, *, session_handler: Optional[SessionHandler] = None) -> None:
        """
        Args:
            api_key: AirNow API key. If ``None`` it is looked up lazily on first use.
            session_handler: HTTP session pool (default: the package-wide shared one).
        """
        self._api_key = api_key
        self._session_handler = session_handler or default_session_handler()

    def __repr__(self) -> str:
        return f"AirNowAPI(api_key={'***' if self._api_key else None})"

    # ---- credentials -------------------------------------------------------
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

    # ---- 1. validate -----------------------------------------------------------
    def _validate(self, start_date, end_date, chunk, params: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Return ``(jobs, query)``: one ``{"startDate", "endDate"}`` job per window, and the shared query fields."""
        if (start_date is None) != (end_date is None):
            raise ValueError("Give both start_date and end_date, or neither (default: the last 24 hours).")
        base = AirNowParams(**params)
        start = start_date if start_date is not None else base.startDate
        end = end_date if end_date is not None else base.endDate
        windows = make_windows(start, end, chunk, end_inclusive=self.window_end_inclusive)
        query = asdict(base)
        query.pop("startDate"), query.pop("endDate")
        return [{"startDate": s, "endDate": e} for s, e in windows], query

    # ---- 2. url ----------------------------------------------------------------------
    def _url(self, job: Job, *, with_key: bool = False) -> str:
        """The request URL for a job (``AirNowParams`` fields). The key is only included on request."""
        query = asdict(AirNowParams(**job))
        if with_key:
            query["API_KEY"] = self.api_key
        return build_url(self.base_url, params=query)

    # ---- 3. fetch (network -> raw JSON) -----------------------------------------------
    def _fetch_one(self, job: Job) -> list:
        """Send one request and return the decoded JSON payload. Errors never contain the key."""
        key = self.api_key
        try:
            response = self._session_handler.session().get(self._url(job, with_key=True))
        except requests.RequestException as exc:
            # requests puts the full URL (with the key) in its messages, so re-raise a clean one.
            raise APIError(f"AirNow request failed: {type(exc).__name__}: {redact(exc, key)}") from None
        if response.status_code != 200:
            raise APIError(
                f"AirNow API returned HTTP {response.status_code}: {redact(response.text[:500], key)}",
                status_code=response.status_code,
            )
        payload = response.json()
        if isinstance(payload, dict):  # AirNow reports some errors as a JSON object with HTTP 200
            raise APIError(f"AirNow API error: {redact(json.dumps(payload)[:500], key)}")
        return payload

    # ---- 4. parse (raw JSON -> DataFrame) ---------------------------------------------
    @staticmethod
    def _parse_one(job: Job, payload: list) -> pd.DataFrame:
        return pd.DataFrame(payload)

    # ---- fetch + parse for one job, with the window cache -----------------------------
    def _cache_path(self, job: Job) -> Path:
        digest = hashlib.sha256(json.dumps(dict(job), sort_keys=True, default=str).encode()).hexdigest()[:24]
        return _cache_root("airnow_api") / f"{digest}.parquet"

    def _is_settled(self, end: str) -> bool:
        end_ts = pd.Timestamp(datetime.strptime(end, _HOUR_FMT))
        return end_ts < pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=self.settle_hours)

    def _load_one(self, job: Job, *, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
        cache_path = None
        if use_cache and self._is_settled(job["endDate"]):
            cache_path = self._cache_path(job)
            if cache_path.exists() and not refresh:
                return pd.read_parquet(cache_path)
        df = self._parse_one(job, self._fetch_one(job))
        if cache_path is not None and not df.empty:
            tmp = cache_path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(cache_path)
        return df

    @staticmethod
    def _metadata(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and formats metadata from a given DataFrame by removing data columns, deduplicating entries, and aggregating parameter information.

        Args:
            df (pd.DataFrame): Input DataFrame containing air quality data with columns such as "UTC", "Value", "RawConcentration", "AQI", "Category", "Parameter", "Unit", and "FullAQSCode".

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

    # ---- 5. assemble -----------------------------------------------------------------------
    def _assemble(self, df: pd.DataFrame, query: dict) -> atmoz_dataset:
        stamp = lambda s: s.replace(":", "")
        name = f"airnow_{stamp(query['startDate'])}_{stamp(query['endDate'])}"
        meta = {"endpoint": "airnow", "files": [name], "rows": {name: len(df)},
                "start": query["startDate"], "end": query["endDate"], "sites": self._metadata(df)}
        return atmoz_dataset(data={name: df}, metadata=meta, datatype="surface")

    def _collect(self, start_date, end_date, params, *, chunk, max_workers, use_cache, refresh, silent, show_traceback):
        """validate -> load every window on threads -> one DataFrame (+ the query that produced it)."""
        jobs, query = self._validate(start_date, end_date, chunk, params)
        self.api_key  # fail fast on a missing key, before any threads start
        frames = run_jobs(
            lambda job: self._load_one({**query, **job}, use_cache=use_cache, refresh=refresh),
            jobs, key=lambda j: j["startDate"], describe=lambda j: f"{j['startDate']} -> {j['endDate']}",
            noun="AirNow request", desc="Downloading AirNow data", max_workers=max_workers,
            silent=silent, show_traceback=show_traceback, secrets=[self._api_key],
        )
        df = pd.concat(list(frames.values()), ignore_index=True) if frames else pd.DataFrame()
        df = df.drop_duplicates(ignore_index=True)  # safety net if AirNow's endDate semantics differ
        if df.empty:
            raise ValueError("AirNow returned no observations for this request.")
        return df, {**query, "startDate": jobs[0]["startDate"], "endDate": jobs[-1]["endDate"]}

    # ---- public --------------------------------------------------------------------------------
    @classmethod
    def download(cls,
                 start_date: Optional[Union[str, date, datetime]] = None,
                 end_date: Optional[Union[str, date, datetime]] = None,
                 *,
                 api_key: Optional[str] = None,
                 max_workers: int = 3,
                 chunk: Union[str, timedelta] = "1D",
                 use_cache: bool = True,
                 refresh: bool = False,
                 silent: bool = False,
                 show_traceback: bool = False,
                 session_handler: Optional[SessionHandler] = None,
                 **params) -> atmoz_dataset:
        """
        Fetch observations for a date range and return them as an ``atmoz_dataset``.

        Parameters:
            start_date, end_date: Inclusive range. A bare date (``"2025-08-07"``) as *end_date*
                means "through 23:00 UTC that day"; pass a datetime or ``"2025-08-07T12"`` for an
                exact hour. ``start_date == end_date`` returns that single day. Omit both for
                the last 24 hours. Times are UTC.
            api_key: AirNow key (default: env var / ``.atmoz.env``, see :meth:`set_api_key`).
            max_workers: Concurrent requests (default 3, be kind to the API).
            chunk: Size of each request window, any ``pd.Timedelta`` string (default ``"1D"``).
                Use smaller windows for a wide box and many pollutants.
            use_cache, refresh: Cache controls (see class docs).
            silent: No progress bar.
            **params: :class:`AirNowParams` fields: ``parameters``, ``BBOX``, ``dataType``, ...

        Raises:
            DownloadError: if any window fails (no partial result).
            MissingCredentialError: if no API key can be found.
        """
        api = cls(api_key, session_handler=session_handler)
        df, query = api._collect(start_date, end_date, params, chunk=chunk, max_workers=max_workers,
                                 use_cache=use_cache, refresh=refresh, silent=silent, show_traceback=show_traceback)
        return api._assemble(df, query)

    @classmethod
    def fetch(cls,
              start_date: Optional[Union[str, date, datetime]] = None,
              end_date: Optional[Union[str, date, datetime]] = None,
              *,
              output_dir: Optional[PathLike] = None,
              api_key: Optional[str] = None,
              max_workers: int = 3,
              chunk: Union[str, timedelta] = "1D",
              use_cache: bool = True,
              refresh: bool = False,
              silent: bool = False,
              show_traceback: bool = False,
              session_handler: Optional[SessionHandler] = None,
              **params) -> Dict[str, Path]:
        """
        Same arguments as :meth:`download`, but save the result as one Parquet file and return
        ``{filename: path}`` (``output_dir`` defaults to the current directory).

        The query is stored in the file's ``attrs["query_params"]`` (the API key is not).
        """
        api = cls(api_key, session_handler=session_handler)
        df, query = api._collect(start_date, end_date, params, chunk=chunk, max_workers=max_workers,
                                 use_cache=use_cache, refresh=refresh, silent=silent, show_traceback=show_traceback)
        df.attrs = {"query_params": query}
        stamp = lambda s: s.replace(":", "")
        filename = f"airnow_api_query_{stamp(query['startDate'])}_{stamp(query['endDate'])}.parquet"
        folder = Path(output_dir) if output_dir else Path(".")
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(folder / filename, index=False)
        return {filename: folder / filename}
