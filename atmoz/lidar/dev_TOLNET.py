# -*- coding: utf-8 -*-
"""
Created on 2026-06-10

@author: Maurice Roots

A Module for Downloading and Plotting TOLNet Data
"""

#%%

# Math & Data
import numpy as np
import pandas as pd

# Housekeeping
import threading
import warnings
import yaml
import requests
from datetime import datetime
from dateutil import tz
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Union

# Validation
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Plotting
import matplotlib.pyplot as plt

# Multi-Threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal
from atmoz.resources import plot_utilities, useful_functions, colorbars, default_plot_params
from atmoz import models
from atmoz.data_access.NASA import EarthData


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
    def __init__(self):
        self.base_url = r"https://tolnet.larc.nasa.gov/api"
        self.file_list_dtypes = {
            "row": "int16",
            "count": "int16",
            "id": "int16",
            "file_name": "str",
            "file_server_location": "str",
            "author": "str",
            "instrument_group_id": "int16",
            "product_type_id": "int16",
            "file_type_id": "int16",
            "start_data_date": "datetime64[ns]",
            "end_data_date": "datetime64[ns]",
            "upload_date": "datetime64[ns]",
            "public": "bool",
            "instrument_group_name": "str",
            "folder_name": "str",
            "current_pi": "str",
            "doi": "str",
            "citation_url": "str",
            "product_type_name": "str",
            "processing_type_name": "str",
            "file_type_name": "str",
            "revision": "int16",
            "near_real_time": "str",
            "file_size": "int64",
            "latitude": "float64",
            "longitude": "float64",
            "altitude": "float64",
            "isAccessible": "bool",
            }

        self.plot_theme = default_plot_params.curtain_plot_theme
        self.plot_params = default_plot_params.tolnet_plot_params
        self.watermark = useful_functions.get_asset("Watermark_TOLNet.png")
        self.nrt = True
        self.data = {}
        self.meta_data = {}
        self._errors = []
        return

    @cached_property
    def api_schema(self):
        """Returns a yaml object containing api_schema information."""
        response = requests.get(self.base_url + "/openapi.yml")
        response.raise_for_status()
        return yaml.safe_load(response.text)

    @cached_property
    def products(self) -> pd.DataFrame:
        """Returns a DataFrame of all product types."""
        response = requests.get(self.base_url + r"/data/product_types")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    @cached_property
    def file_types(self) -> pd.DataFrame:
        """Returns a DataFrame of all file types."""
        response = requests.get(self.base_url + r"/data/file_types")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    @cached_property
    def instrument_groups(self) -> pd.DataFrame:
        """Returns a DataFrame of all instrument groups."""
        response = requests.get(self.base_url + r"/instruments/groups")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    @cached_property
    def processing_types(self) -> pd.DataFrame:
        """Returns a DataFrame of all processing types."""
        response = requests.get(self.base_url + r"/data/processing_types")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    def _get_files_list(self, **params) -> pd.DataFrame:
        """
        Fetch paginated TOLNet file list. Accepts all TOLNET_DATA_QUERY fields as kwargs.
        """
        query = TOLNET_DATA_QUERY(**params)
        query_dict = query.model_dump(exclude_none=True)

        session = requests.Session()
        data_frames = []
        page = 1

        while True:
            url = f"{self.base_url}/data/{page}"
            response = session.get(url, params=query_dict, timeout=10)
            if response.status_code != 200:
                break
            json_data = response.json()
            if not json_data:
                break
            data_frames.append(pd.DataFrame(json_data))
            page += 1

        if not data_frames:
            return pd.DataFrame().astype(self.file_list_dtypes)

        df = pd.concat(data_frames, ignore_index=True)
        for col in ["start_data_date", "end_data_date", "upload_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df.astype(self.file_list_dtypes)

    def search_by_file_name(
        self,
        file_name: str,
        match_type: Literal["exact_match", "begins_with", "ends_with"] = "exact_match",
        ) -> pd.DataFrame:
        """
        Search for files by name via GET /data/search_by_file_name.

        Parameters
        ----------
        file_name : str
            The file name (or partial name) to search for.
        match_type : str
            "exact_match" (default), "begins_with", or "ends_with".
        """
        response = requests.get(
            f"{self.base_url}/data/search_by_file_name",
            params={"file_name": file_name, "match_type": match_type},
            timeout=10,
            )
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame().astype(self.file_list_dtypes)
        return pd.DataFrame(data).astype(self.file_list_dtypes)

    def calendar(
        self,
        instrument_group: Union[str, int],
        file_type: Union[str, int],
        product_type: Optional[Union[str, int]] = None,
        processing_type: Optional[Union[str, int]] = None,
        ) -> pd.DataFrame:
        """
        Return a chronological listing via GET /data/calendar.

        Parameters
        ----------
        instrument_group : str or int
            Instrument group name or ID (required).
        file_type : str or int
            File type name or ID (required).
        product_type : str or int, optional
        processing_type : str or int, optional
        """
        params: dict = {"instrument_group": str(instrument_group), "file_type": str(file_type)}
        if product_type is not None:
            params["product_type"] = str(product_type)
        if processing_type is not None:
            params["processing_type"] = str(processing_type)
        response = requests.get(f"{self.base_url}/data/calendar", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def _add_timezone(self, time):
        return [utc.replace(tzinfo=tz.gettz("UTC")) for utc in time]

    def change_timezone(self, timezone: str):
        to_zone = tz.gettz(timezone)
        for key in self.data.keys():
            for date in self.data[key].keys():
                time = self.data[key][date].index.to_list()
                self.data[key][date].index = [t.astimezone(to_zone) for t in time]
        return self

    def _json_to_dict(self, file_id: int) -> dict:
        try:
            url = f"{self.base_url}/data/json/{file_id}"
            return requests.get(url).json()
        except Exception:
            self._errors.append(f"Error pulling file id {file_id}")

    def _unpack_data(self, meta_data: dict) -> pd.DataFrame:
        try:
            df = pd.DataFrame(
                meta_data["value"]["data"],
                columns=meta_data["altitude"]["data"],
                index=pd.to_datetime(meta_data["datetime"]["data"]),
                )
            df = df.apply(pd.to_numeric, errors="coerce")
            df.sort_index(inplace=True)
        except Exception as e:
            self._errors.append(f"_unpack_data: {e}")
            return
        return df

    def import_data(self, min_date: str, max_date: str, **kwargs):
        """
        Download TOLNet JSON data into memory and store in self.data.

        Parameters
        ----------
        min_date : str
            Start date in YYYY-MM-DD format.
        max_date : str
            End date in YYYY-MM-DD format.
        **kwargs
            Any TOLNET_DATA_QUERY field (product_type, instrument_group, etc.).
            Pass GEOS_CF=True to also fetch co-located GEOS-CF model data.
        """
        geos_cf = kwargs.pop("GEOS_CF", False)

        def process_file(file_name, file_id):
            meta_data = self._json_to_dict(file_id)
            data = self._unpack_data(meta_data)
            data.index = self._add_timezone(data.index.to_list())
            return file_name, meta_data, data

        self.files = self._get_files_list(min_date=min_date, max_date=max_date, **kwargs)
        self.request_dates = (min_date, max_date)
        self.meta_data = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {
                executor.submit(process_file, file_name, file_id): file_name
                for file_name, file_id in zip(self.files["file_name"], self.files["id"])
                }

            for future in tqdm(
                as_completed(future_to_file),
                total=len(future_to_file),
                desc="Downloading TOLNet Data",
                ncols=100,
                ):
                file_name = future_to_file[future]
                try:
                    file_name, meta_data, data = future.result()

                    lat_lon = (
                        str(meta_data["LATITUDE.INSTRUMENT"])
                        + "x"
                        + str(meta_data["LONGITUDE.INSTRUMENT"])
                        )
                    date = meta_data["fileInfo"]["start_data_date"].split(" ")[0]
                    key = (
                        meta_data["fileInfo"]["instrument_group_name"],
                        meta_data["fileInfo"]["processing_type_name"],
                        lat_lon,
                        )

                    if key not in self.data:
                        self.data[key] = {}
                        self.meta_data[key] = {}

                    self.data[key][date] = data
                    self.meta_data[key][file_name] = meta_data

                except Exception as e:
                    self._errors.append(f"Error processing file {file_name}: {e}")

        if geos_cf:
            geos_cf_model = models.geos_cf()
            for key in list(self.data.keys()):
                geos_cf_model.get_geos_data_multithreaded(
                    key[2],
                    self.request_dates[0],
                    self.request_dates[1],
                    )
            for geos_key, geos_dates in geos_cf_model.data.items():
                self.data[geos_key] = geos_dates

        return self

    def _download(self, file_id: Union[int, str], url_path: str):
        url = f"{self.base_url}{url_path}/{file_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return file_id, response.content
        print(f"Failed to download file {file_id}. Status code: {response.status_code}")
        return file_id, None

    def download(self, dest_dir: str, file_type: str = "json", threaded: bool = True, max_workers: int = 8, **params):
        """
        Query the file list and download matching files to dest_dir.

        Parameters
        ----------
        dest_dir : str
            Directory to save downloaded files.
        file_type : str
            "json" (default) or "hdf".
        threaded : bool
            Use parallel download (default True).
        max_workers : int
            Thread count for parallel download (default 8).
        **params
            Any TOLNET_DATA_QUERY fields to filter the file list.
        """
        files_list = self._get_files_list(**params)
        if files_list.empty:
            print("No files found for the given query parameters.")
            return []
        if threaded:
            return self.download_files_threaded(files_list, dest_dir, file_type=file_type, max_workers=max_workers)
        else:
            return self.download_files(files_list, dest_dir, file_type=file_type)

    def download_files(self, files_list: pd.DataFrame, dest_dir: str, file_type: str = "json"):
        """Sequential authenticated download of TOLNet files to disk."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        session = EarthData().auth.get_session()

        for idx, file_id in enumerate(files_list.id, start=1):
            if file_type == "json":
                url = f"{self.base_url}/data/json/{file_id}"
                ext = "json"
            elif file_type == "hdf":
                url = f"{self.base_url}/data/download/{file_id}"
                ext = "hdf"
            else:
                raise ValueError("file_type must be 'json' or 'hdf'")

            dest_path = dest_dir / f"{file_id}.{ext}"
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                dest_path.write_bytes(r.content)
            except requests.RequestException as e:
                print(f"[{idx}/{len(files_list)}] Failed {file_id}: {e}")
            else:
                print(f"[{idx}/{len(files_list)}] Downloaded {file_id}")

    def download_files_threaded(
        self,
        files_list: pd.DataFrame,
        dest_dir: str,
        file_type: str = "json",
        max_workers: int = 8,
        ) -> list:
        """Parallel authenticated download of TOLNet files to disk."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        _local = threading.local()

        def _get_session():
            if not hasattr(_local, "session"):
                _local.session = EarthData().auth.get_session()
            return _local.session

        def download_one(file_id):
            if file_type == "json":
                url = f"{self.base_url}/data/json/{file_id}"
                ext = "json"
            elif file_type == "hdf":
                url = f"{self.base_url}/data/download/{file_id}"
                ext = "hdf"
            else:
                raise ValueError("file_type must be 'json' or 'hdf'")
            dest_path = dest_dir / f"{file_id}.{ext}"
            try:
                r = _get_session().get(url, timeout=30)
                r.raise_for_status()
                dest_path.write_bytes(r.content)
            except requests.RequestException as e:
                return file_id, False, str(e)
            return file_id, True, None

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, fid): fid for fid in files_list.id}
            for future in as_completed(futures):
                fid, success, error = future.result()
                if success:
                    print(f"Downloaded {fid}")
                else:
                    print(f"Failed {fid}: {error}")
                results.append((fid, success, error))

        return results

    def plot_curtain(self, data: dict, **kwargs):
        """
        Render a time-altitude curtain plot of ozone mixing ratio.

        Parameters
        ----------
        data : dict
            Keyed by date string (YYYY-MM-DD); values are DataFrames with
            datetime index and altitude columns.
        **kwargs
            Overrides for self.plot_params. Also accepts:
              xlims : "auto" | [min_date_str, max_date_str]
              time_resolution : "auto" | pandas offset string (e.g. "30min")
              use_contourf : bool (default False, uses pcolormesh)
        """
        params = useful_functions.merge_dicts(self.plot_params, kwargs)
        xlims = params.pop("xlims", "auto")
        time_resolution = params.pop("time_resolution", "auto")

        cmap, norm = colorbars.tolnet_ozone()

        with plt.rc_context(self.plot_theme):
            fig, ax = plt.subplots()

            dates = sorted(list(data.keys()))

            if xlims == "auto":
                pass
            elif isinstance(xlims, list):
                xlims = pd.to_datetime(xlims, utc=True)
                xlims = [xlims.min(), xlims.max()]
                dates_dt = pd.to_datetime(dates, utc=True)
                dates = [
                    str(x.strftime("%Y-%m-%d"))
                    for x in dates_dt[
                        (dates_dt >= xlims[0]) & (dates_dt <= xlims[1])
                        ]
                    ]

            im = None
            for date in dates:
                df = data[date].copy() if xlims == "auto" else data[date].copy()[xlims[0]:xlims[1]]

                if df.empty:
                    continue

                if time_resolution == "auto":
                    resolution = np.min([df.index[i] - df.index[i - 1] for i in range(1, len(df))])
                    df = df.resample(f"{resolution.seconds}s").mean()
                else:
                    df = df.resample(f"{time_resolution}").mean()

                X, Y, Z = df.index, df.columns, df.to_numpy().T

                if params.get("use_contourf", False):
                    levels = norm.boundaries
                    im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
                else:
                    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="nearest", alpha=1)

            if im is not None:
                params["fig.colorbar"]["mappable"] = im

            plot_utilities.apply_datetime_axis(ax)

            if self.watermark:
                plot_utilities.apply_watermark(fig, self.watermark)

            if self.nrt is True:
                plot_utilities.apply_near_real_time(ax)

            plot_utilities.apply_plot_params(fig, ax, **params)

            plt.show()
        return


if __name__ == "__main__":
    tolnet = TOLNET()

    date_start = "2025-07-28"
    date_end = "2025-08-01"

    tolnet.import_data(
        min_date=date_start,
        max_date=date_end,
        product_type=[4],
        GEOS_CF=False,
        )

    tolnet.plot_curtain(
        tolnet.data[("NASA JPL SMOL-2", "Centrally Processed (GLASS)", "39.24x-76.363")]
        )
