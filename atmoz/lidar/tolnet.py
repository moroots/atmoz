"""NASA TOLNet (Tropospheric Ozone Lidar Network): query, download and plot.

One module for everything TOLNet:

* the REST API client (:class:`TOLNET`): file catalogue, ``download()`` (parsed) and ``fetch()`` (raw files),
* lazy HDF reading of fetched GEOMS files (:meth:`TOLNET.open_hdf`),
* time-altitude curtain plots (:meth:`TOLNET.plot_curtain`, which uses :func:`atmoz.plotting.curtain`).

Follows the package conventions (``docs/conventions.md``): ``download()`` returns an ``atmoz_dataset``,
``fetch()`` returns ``{name: Path}``, downloads are cached and all-or-nothing.
"""

from __future__ import annotations

import json
import threading
import warnings
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atmoz.exceptions import NoDataError
from atmoz.resources.atmoz_dataclasses import atmoz_dataset
from atmoz.resources.cache import cache_dir as _cache_root
from atmoz.resources.cache import download_to_cache
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.urls import build_url
from atmoz.resources.useful_functions import get_asset
from atmoz.resources.utilities import PathLike, default_session_handler, run_jobs

_FILE_KINDS = {"json": ("data", "json"), "hdf": ("data", "download")}   # file_type -> API path
GroupKey = Tuple[str, str, str]   # (instrument group, processing type, "latxlon")


# --------------------------------------------------------------------------------------------------------------------------------- #
# Query Model
# --------------------------------------------------------------------------------------------------------------------------------- #
class TOLNET_DATA_QUERY(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Fields that accept str, int, or list
    instrument_group: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="instrument_group_name", default=None)
    product_type: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="product_type_name", default=None)
    processing_type: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="processing_type_name", default=None)
    file_type: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="file_type_name", default=None)

    # Boolean Fields
    near_real_time: Optional[Literal["true", "false"]] = Field(alias="near_real_time", default=None)

    # Date Fields
    min_date: Optional[str] = Field(alias="min_date", default=None)
    max_date: Optional[str] = Field(alias="max_date", default=None)
    min_upload_date: Optional[str] = Field(alias="min_upload_date", default=None)
    max_upload_date: Optional[str] = Field(alias="max_upload_date", default=None)

    # Geospatial Fields
    latitude: Optional[float] = Field(gt=-90, lt=90, description="Latitude must be between -90 and 90 degrees.", default=None)
    longitude: Optional[float] = Field(gt=-180, lt=180, description="Longitude must be between -180 and 180 degrees.", default=None)
    radius: Optional[float] = Field(gt=0, description="Radius must be greater than 0.", default=None)

    # Bounding Box Fields
    minLatitude: Optional[float] = Field(gt=-90, lt=90, description="Minimum latitude must be between -90 and 90 degrees.", default=None)
    maxLatitude: Optional[float] = Field(gt=-90, lt=90, description="Maximum latitude must be between -90 and 90 degrees.", default=None)
    minLongitude: Optional[float] = Field(gt=-180, lt=180, description="Minimum longitude must be between -180 and 180 degrees.", default=None)
    maxLongitude: Optional[float] = Field(gt=-180, lt=180, description="Maximum longitude must be between -180 and 180 degrees.", default=None)

    # Sorting Fields
    order: Optional[
        Literal[
            "data_date",
            "upload_date",
            "instrument_group_name",
            "product_type_name",
            "processing_type_name",
            "file_type_name",
            "file_name",
            ]
        ] = Field(
            default="data_date",
            description="Valid values are data_date, upload_date, instrument_group_name, product_type_name, processing_type_name, file_type_name, or file_name",
            )

    order_direction: Optional[
        Literal["asc", "desc"]
        ] = Field(
            default="desc",
            description="Valid values are asc or desc",
            )

    @model_validator(mode="after")
    def valid_bbox(self):
        bbox_fields = [self.minLatitude, self.maxLatitude, self.minLongitude, self.maxLongitude]
        if all(field is not None for field in bbox_fields):
            for f in ["latitude", "longitude", "radius"]:
                if getattr(self, f) is not None:
                    warnings.warn(f"Do not use {f} and bbox at same time. Ignoring {f} in favor of bbox.")
                    setattr(self, f, None)
            if not (self.minLatitude < self.maxLatitude):
                raise ValueError("minLatitude must be less than maxLatitude.")
            if not (self.minLongitude < self.maxLongitude):
                raise ValueError("minLongitude must be less than maxLongitude.")
        elif any(field is not None for field in bbox_fields):
            raise ValueError("If any of the bbox fields (minLatitude, maxLatitude, minLongitude, maxLongitude) are provided, all must be provided.")
        return self

    @model_validator(mode="after")
    def valid_dates(self):
        date_fields = ["min_date", "max_date", "min_upload_date", "max_upload_date"]
        for field in date_fields:
            val = getattr(self, field, None)
            if val is not None:
                try:
                    datetime.strptime(val, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"{field} must be in YYYY-MM-DD format.")
        if all(getattr(self, field, None) is not None for field in date_fields[:2]):
            if not (self.min_date <= self.max_date):
                raise ValueError("min_date must be less than or equal to max_date.")
        if all(getattr(self, field, None) is not None for field in date_fields[2:]):
            if not (self.min_upload_date <= self.max_upload_date):
                raise ValueError("min_upload_date must be less than or equal to max_upload_date.")
        return self

    @field_validator("instrument_group", "product_type", "processing_type", "file_type", mode="before")
    @classmethod
    def _to_csv_string(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        if isinstance(v, (int, str)):
            return str(v)
        raise TypeError("Expected int, str, or List[int, str]")

    @field_validator("near_real_time", mode="before")
    @classmethod
    def _normalize_near_real_time(cls, v):
        """Accept bool, int (0/1), or strings ('true'/'false'/'1'/'0') and store 'true'/'false'."""
        if v is None:
            return None
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            if v in (0, 1):
                return "true" if v == 1 else "false"
            raise ValueError("near_real_time int must be 0 or 1")
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1"}:
                return "true"
            if s in {"false", "0"}:
                return "false"
            raise ValueError('near_real_time str must be "true", "false", "0", or "1"')
        raise TypeError("near_real_time must be bool, int(0/1), or str(true/false/0/1)")


# --------------------------------------------------------------------------------------------------------------------------------- #
# TOLNET Class
# --------------------------------------------------------------------------------------------------------------------------------- #
class TOLNET:
    """Client for the TOLNet REST API (https://tolnet.larc.nasa.gov/api).

    >>> ds = TOLNET.download("2025-07-28", "2025-08-01", product_type=[4], processing_type=[1])
    >>> groups = TOLNET.group(ds)                  # {(instrument, processing, "latxlon"): {date: DataFrame}}
    >>> TOLNET.plot_curtain(groups[next(iter(groups))])

    * :meth:`download` -> :class:`~atmoz.atmoz_dataset` (``.data == {file_name: DataFrame}``; ``.metadata`` has the
      per-file TOLNet metadata, the file list and the instrument groups). JSON needs no login.
    * :meth:`fetch`    -> ``{file_name: path}`` (raw ``json`` or ``hdf`` files; HDF needs NASA EarthData credentials).

    The catalogue helpers (:attr:`products`, :attr:`file_types`, :attr:`instrument_groups`, :attr:`processing_types`,
    :meth:`calendar`, :meth:`search_by_file_name`) work on an instance: ``TOLNET().products``.
    """

    base_url = "https://tolnet.larc.nasa.gov/api"
    file_list_dtypes = {
        "row": "int16", "count": "int16", "id": "int32", "file_name": "str", "file_server_location": "str",
        "author": "str", "instrument_group_id": "int16", "product_type_id": "int16", "file_type_id": "int16",
        "start_data_date": "datetime64[ns]", "end_data_date": "datetime64[ns]", "upload_date": "datetime64[ns]",
        "public": "bool", "instrument_group_name": "str", "folder_name": "str", "current_pi": "str", "doi": "str",
        "citation_url": "str", "product_type_name": "str", "processing_type_name": "str", "file_type_name": "str",
        "revision": "int16", "near_real_time": "str", "file_size": "int64", "latitude": "float64",
        "longitude": "float64", "altitude": "float64", "isAccessible": "bool",
    }

    def __init__(self, session_handler: Optional[SessionHandler] = None):
        self._handler = session_handler or default_session_handler()

    # ---- catalogue ---------------------------------------------------------------------------------------
    def _get(self, *path: Any, params: Optional[dict] = None, timeout: float = 30) -> requests.Response:
        response = self._handler.session().get(build_url(self.base_url, *path), params=params, timeout=timeout)
        response.raise_for_status()
        return response

    def _table(self, *path: str) -> pd.DataFrame:
        return pd.DataFrame(self._get(*path).json()).sort_values(by=["id"]).set_index("id", drop=True)

    @cached_property
    def api_schema(self) -> dict:
        """The API's OpenAPI description (parsed YAML)."""
        return yaml.safe_load(self._get("openapi.yml").text)

    @cached_property
    def products(self) -> pd.DataFrame:
        """All product types."""
        return self._table("data", "product_types")

    @cached_property
    def file_types(self) -> pd.DataFrame:
        """All file types."""
        return self._table("data", "file_types")

    @cached_property
    def instrument_groups(self) -> pd.DataFrame:
        """All instrument groups."""
        return self._table("instruments", "groups")

    @cached_property
    def processing_types(self) -> pd.DataFrame:
        """All processing types."""
        return self._table("data", "processing_types")

    def _empty_files(self) -> pd.DataFrame:
        return pd.DataFrame(columns=list(self.file_list_dtypes)).astype(self.file_list_dtypes)

    def _get_files_list(self, **params) -> pd.DataFrame:
        """The (paginated) TOLNet file list for a query. Accepts every :class:`TOLNET_DATA_QUERY` field."""
        query = TOLNET_DATA_QUERY(**params).model_dump(exclude_none=True)
        frames, page = [], 1
        while True:
            response = self._handler.session().get(build_url(self.base_url, "data", page), params=query, timeout=30)
            if response.status_code != 200:
                break
            payload = response.json()
            if not payload:
                break
            frames.append(pd.DataFrame(payload))
            page += 1
        if not frames:
            return self._empty_files()
        df = pd.concat(frames, ignore_index=True)
        for col in ("start_data_date", "end_data_date", "upload_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df.reindex(columns=list(self.file_list_dtypes)).astype(self.file_list_dtypes)

    def search_by_file_name(self, file_name: str,
                            match_type: Literal["exact_match", "begins_with", "ends_with"] = "exact_match") -> pd.DataFrame:
        """Find files by name (``GET /data/search_by_file_name``)."""
        data = self._get("data", "search_by_file_name", params={"file_name": file_name, "match_type": match_type}).json()
        if not data:
            return self._empty_files()
        return pd.DataFrame(data).reindex(columns=list(self.file_list_dtypes)).astype(self.file_list_dtypes)

    def calendar(self, instrument_group: Union[str, int], file_type: Union[str, int],
                 product_type: Optional[Union[str, int]] = None,
                 processing_type: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """Chronological listing of what exists (``GET /data/calendar``)."""
        params: Dict[str, str] = {"instrument_group": str(instrument_group), "file_type": str(file_type)}
        if product_type is not None:
            params["product_type"] = str(product_type)
        if processing_type is not None:
            params["processing_type"] = str(processing_type)
        data = self._get("data", "calendar", params=params).json()
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ---- process: validate -> url -> fetch_one -> parse_one -> assemble -----------------------------------
    def _validate(self, min_date: Optional[str], max_date: Optional[str], file_type: str, query: dict) -> List[dict]:
        """Check the request, ask the API which files match, and expand it into one job per file."""
        if file_type not in _FILE_KINDS:
            raise ValueError("file_type must be 'json' or 'hdf'")
        files = self._get_files_list(min_date=min_date, max_date=max_date, **query)
        if files.empty:
            raise NoDataError("TOLNet has no files for this request.")
        jobs = [{"id": int(r.id), "file_name": str(r.file_name), "file_type": file_type,
                 "nrt": str(r.near_real_time).lower() == "true"} for r in files.itertuples()]
        self._files = files
        return jobs

    def _url(self, job: dict) -> str:
        return build_url(self.base_url, *_FILE_KINDS[job["file_type"]], job["id"])

    @staticmethod
    def _filename(job: dict) -> str:
        return f"{job['id']}.{job['file_type']}"

    def _session_for(self, file_type: str):
        """JSON needs no login (shared pool). HDF needs NASA EarthData credentials (one session per thread)."""
        if file_type == "json":
            return self._handler.session()
        local = self.__dict__.setdefault("_earthdata", threading.local())
        if not hasattr(local, "session"):
            from atmoz.data_access.NASA import EarthData   # lazy: needs the `nasa` extra
            local.session = EarthData().auth.get_session()
        return local.session

    def _fetch_one(self, job: dict, *, dest_dir: Path, refresh: bool, use_cache: bool = True) -> Path:
        """One job -> a file on disk. Near-real-time files are re-checked hourly, final ones never change."""
        dest = dest_dir / self._filename(job)
        if not use_cache:
            refresh = True
        return download_to_cache(self._url(job), dest, self._session_for(job["file_type"]),
                                 max_age=timedelta(hours=1) if job["nrt"] else None, refresh=refresh)

    @staticmethod
    def _parse_one(path: PathLike) -> Tuple[dict, pd.DataFrame]:
        """One TOLNet JSON file -> (its metadata, a DataFrame indexed by UTC time with altitude columns)."""
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        df = pd.DataFrame(meta["value"]["data"], columns=meta["altitude"]["data"],
                          index=pd.to_datetime(meta["datetime"]["data"], utc=True))
        return meta, df.apply(pd.to_numeric, errors="coerce").sort_index()

    def _load_one(self, job: dict, **kw) -> Tuple[dict, pd.DataFrame]:
        return self._parse_one(self._fetch_one(job, **kw))

    @staticmethod
    def _group_key(meta: dict) -> GroupKey:
        info = meta["fileInfo"]
        return (info["instrument_group_name"], info["processing_type_name"],
                f"{meta['LATITUDE.INSTRUMENT']}x{meta['LONGITUDE.INSTRUMENT']}")

    def _assemble(self, loaded: Dict[str, Tuple[dict, pd.DataFrame]], files: pd.DataFrame) -> atmoz_dataset:
        data = {name: df for name, (_, df) in loaded.items()}
        info = {name: meta for name, (meta, _) in loaded.items()}
        groups: Dict[str, List[str]] = {}
        for name, meta in info.items():
            groups.setdefault("|".join(self._group_key(meta)), []).append(name)
        metadata = {"endpoint": "tolnet", "files": list(data), "rows": {n: len(d) for n, d in data.items()},
                    "info": info, "groups": groups, "file_list": files}
        return atmoz_dataset(data=data, metadata=metadata, datatype="lidar")

    # ---- public --------------------------------------------------------------------------------------------
    @classmethod
    def download(cls,
                 min_date: str,
                 max_date: str,
                 *,
                 max_workers: int = 4,
                 use_cache: bool = True,
                 refresh: bool = False,
                 cache_dir: Optional[PathLike] = None,
                 silent: bool = False,
                 show_traceback: bool = False,
                 session_handler: Optional[SessionHandler] = None,
                 **query) -> atmoz_dataset:
        """Download (once, then cached) and parse every matching TOLNet JSON file.

        Parameters
        ----------
        min_date, max_date:
            ``YYYY-MM-DD`` range of the data.
        **query:
            Any :class:`TOLNET_DATA_QUERY` field: ``product_type``, ``instrument_group``, ``processing_type``,
            ``near_real_time``, ``latitude``/``longitude``/``radius`` or a bounding box, ...
        max_workers, use_cache, refresh, cache_dir, silent, show_traceback, session_handler:
            The usual options (see ``docs/conventions.md``).

        Returns
        -------
        atmoz_dataset
            ``datatype="lidar"``; ``.data == {file_name: DataFrame}`` (UTC time index, altitude columns).
            ``.metadata``: ``info`` (TOLNet's metadata per file), ``groups`` (``"instrument|processing|latxlon"`` ->
            file names), ``file_list``. Use :meth:`group` to get frames per instrument and day.

        Raises
        ------
        NoDataError: nothing matches. DownloadError: any file fails (nothing is returned).
        """
        if "GEOS_CF" in query:
            raise TypeError("TOLNET.download(GEOS_CF=...) is gone. Use atmoz.models.GEOS_CF.download(...) at the "
                            "instrument's latitude/longitude (see ds.metadata['info']).")
        api = cls(session_handler)
        jobs = api._validate(min_date, max_date, "json", query)
        dest_dir = Path(cache_dir) if cache_dir else _cache_root("tolnet")
        loaded = run_jobs(
            lambda job: api._load_one(job, dest_dir=dest_dir, refresh=refresh, use_cache=use_cache),
            jobs, key=lambda j: j["file_name"], describe=lambda j: j["file_name"], noun="TOLNet file",
            desc="Downloading TOLNet data", max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )
        return api._assemble(loaded, api._files)

    @classmethod
    def fetch(cls,
              min_date: Optional[str] = None,
              max_date: Optional[str] = None,
              *,
              file_type: Literal["json", "hdf"] = "json",
              dest_dir: Optional[PathLike] = None,
              max_workers: int = 4,
              refresh: bool = False,
              silent: bool = False,
              show_traceback: bool = False,
              session_handler: Optional[SessionHandler] = None,
              **query) -> Dict[str, Path]:
        """Save the matching raw files to disk and return ``{file_name: path}`` (parse them yourself).

        ``file_type="hdf"`` fetches the GEOMS HDF files (needs NASA EarthData credentials, see
        ``docs/configuration.md``); open them with :meth:`open_hdf`. ``dest_dir`` defaults to the atmoz cache.
        """
        api = cls(session_handler)
        jobs = api._validate(min_date, max_date, file_type, query)
        folder = Path(dest_dir) if dest_dir else _cache_root("tolnet")
        folder.mkdir(parents=True, exist_ok=True)
        return run_jobs(
            lambda job: api._fetch_one(job, dest_dir=folder, refresh=refresh),
            jobs, key=lambda j: j["file_name"], describe=lambda j: j["file_name"], noun="TOLNet file",
            desc="Fetching TOLNet files", max_workers=max_workers, silent=silent, show_traceback=show_traceback,
        )

    # ---- helpers on results -----------------------------------------------------------------------------------
    @staticmethod
    def group(ds: atmoz_dataset) -> Dict[GroupKey, Dict[str, pd.DataFrame]]:
        """``{(instrument, processing, "latxlon"): {"YYYY-MM-DD": DataFrame}}``: the shape the curtain plot takes."""
        out: Dict[GroupKey, Dict[str, pd.DataFrame]] = {}
        for name, meta in ds.metadata["info"].items():
            key = TOLNET._group_key(meta)
            day = meta["fileInfo"]["start_data_date"].split(" ")[0]
            out.setdefault(key, {})[day] = ds.data[name]
        return {k: dict(sorted(v.items())) for k, v in out.items()}

    @staticmethod
    def open_hdf(filepaths) -> Dict[str, Any]:
        """Lazily open fetched HDF files: ``{filename: node}`` (``node["O3.MIXING.RATIO.VOLUME_DERIVED"][()]``)."""
        from atmoz.resources.hdf import open_hdf
        return open_hdf(filepaths)

    # ---- plotting (one implementation: atmoz.plotting.curtain) --------------------------------------------------
    @staticmethod
    def plot_curtain(data, **kwargs):
        """Ozone curtain of one DataFrame or a ``{date: DataFrame}`` dict. Returns ``(fig, ax)``.

        Same as ``atmoz.plot.curtain``; adds the TOLNet watermark unless you pass ``watermark=None``/a path.
        Pass ``show=True`` to call ``plt.show()`` (scripts).
        """
        from atmoz.plotting.curtain import curtain
        show = kwargs.pop("show", False)
        kwargs.setdefault("watermark", get_asset("Watermark_TOLNet.png"))
        fig, ax = curtain(data, **kwargs)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig, ax

    @staticmethod
    def plot_curtains(ds: atmoz_dataset, **kwargs) -> Dict[GroupKey, tuple]:
        """One curtain per (instrument, processing type, site) in *ds*: ``{group key: (fig, ax)}``."""
        out = {}
        for key, frames in TOLNET.group(ds).items():
            out[key] = TOLNET.plot_curtain(frames, **{"title": " / ".join(key), **kwargs})
        return out


if __name__ == "__main__":
    ds = TOLNET.download("2025-07-28", "2025-08-01", product_type=[4])
    groups = TOLNET.group(ds)
    TOLNET.plot_curtain(groups[next(iter(groups))], show=True)
