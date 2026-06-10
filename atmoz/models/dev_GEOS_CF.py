# -*- coding: utf-8 -*-
"""
Created on 2026-06-10

@author: Maurice Roots

A Module for Working with NASA GEOS-CF Data via the CFAPI.
Single-site point queries returning xarray Datasets.
"""

#%%

# Math & Data
import numpy as np
import pandas as pd
import xarray as xr

# Housekeeping
import requests
from datetime import datetime
from typing import Dict, List, Literal, Optional

# Validation
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Multi-Threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------------------------------------------------------------------------------------------------------------- #
# Collection Catalog
# --------------------------------------------------------------------------------------------------------------------------------- #
_COLLECTION_CATALOG: Dict[str, dict] = {
    "aqc": {
        "variables": ["O3", "NO2", "CO", "SO2", "PM25_RH35_GCC"],
        "level_type": None,
        "description": "Surface air quality (O3, NO2, CO, SO2, PM2.5)",
    },
    "chm": {
        "variables": ["O3", "NO2", "CO", "SO2", "OH", "HNO3", "CH2O", "PAN"],
        "level_type": "v72",
        "description": "Chemistry vertical profiles, 72 model levels",
    },
    "met": {
        "variables": ["T", "U", "V", "Q", "ZL", "ZPBL", "PS"],
        "level_type": "v72",
        "description": "Meteorology vertical profiles, 72 model levels",
    },
}


# --------------------------------------------------------------------------------------------------------------------------------- #
# Query Model
# --------------------------------------------------------------------------------------------------------------------------------- #
class GEOS_CF_QUERY(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(gt=-90, lt=90)
    lon: float = Field(gt=-180, lt=180)
    start_date: str
    end_date: str
    collection: Literal["aqc", "chm", "met"] = Field(default="aqc")
    mode: Literal["assim", "fcast"] = Field(default="assim")
    variables: Optional[List[str]] = Field(default=None)
    level_type: Optional[Literal["v72", "v36", "p23", "x1"]] = Field(default=None)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: str) -> str:
        """Accept YYYY-MM-DD or YYYYMMDD; store as YYYYMMDD."""
        v = str(v).strip()
        if len(v) == 10 and "-" in v:
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v.replace("-", "")
            except ValueError:
                pass
        elif len(v) == 8:
            try:
                datetime.strptime(v, "%Y%m%d")
                return v
            except ValueError:
                pass
        raise ValueError(f"'{v}' must be YYYY-MM-DD or YYYYMMDD.")

    @model_validator(mode="after")
    def _valid_date_order(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        return self

    @model_validator(mode="after")
    def _fill_collection_defaults(self):
        defaults = _COLLECTION_CATALOG.get(self.collection, {})
        if self.level_type is None:
            self.level_type = defaults.get("level_type")
        if self.variables is None:
            self.variables = list(defaults.get("variables", []))
        return self


# --------------------------------------------------------------------------------------------------------------------------------- #
# GEOS_CF Class
# --------------------------------------------------------------------------------------------------------------------------------- #
class GEOS_CF:
    """
    Single-site client for the NASA GEOS-CF CFAPI.

    Fetches AQC (surface), CHM (chemistry profiles), and MET (met profiles)
    for a single lat/lon point and stores results as xr.Dataset objects.

    Data keys: (lat_lon_str, mode, collection)
      e.g. ("39.24x-76.363", "assim", "aqc")
    """

    def __init__(self):
        self.base_url = r"https://fluid.nccs.nasa.gov/cfapi"
        self.data: Dict[tuple, xr.Dataset] = {}
        self._errors: List[str] = []
        return

    @property
    def catalog(self) -> dict:
        """Available collections with their default variables and level types."""
        return _COLLECTION_CATALOG

    # ---------------------------------------------------------------- #
    # Internal: URL and fetch
    # ---------------------------------------------------------------- #
    def _url(
        self,
        mode: str,
        collection: str,
        level_type: Optional[str],
        variable: str,
        lat_lon: str,
        start: str,
        end: str,
    ) -> str:
        # Surface collections: {base}/{collection}/{mode}/{variable}/{lat_lon}/{start}/{end}
        # Profile collections: {base}/{mode}/{collection}/{level_type}/{variable}/{lat_lon}/{start}/{end}
        if level_type:
            return f"{self.base_url}/{mode}/{collection}/{level_type}/{variable}/{lat_lon}/{start}/{end}"
        return f"{self.base_url}/{collection}/{mode}/{variable}/{lat_lon}/{start}/{end}"

    def _fetch_single(
        self,
        mode: str,
        collection: str,
        level_type: Optional[str],
        variable: str,
        lat_lon: str,
        start: str,
        end: str,
        session: requests.Session,
    ) -> dict:
        url = self._url(mode, collection, level_type, variable, lat_lon, start, end)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _to_dataarray(
        self,
        response_json: dict,
        variable: str,
        has_levels: bool,
    ) -> xr.DataArray:
        times = pd.to_datetime(
            response_json["time"], utc=True, format="%Y-%m-%dT%H:%M:%S"
        )
        values_dict = response_json["values"]

        # Case-insensitive key lookup
        var_key = next((k for k in values_dict if k.upper() == variable.upper()), None)
        if var_key is None:
            raise KeyError(f"'{variable}' not in response. Got: {list(values_dict.keys())}")

        arr = np.array(values_dict[var_key], dtype=float)

        if has_levels and arr.ndim == 2:
            return xr.DataArray(
                arr,
                dims=["time", "level"],
                coords={"time": times, "level": np.arange(arr.shape[1])},
                name=variable,
            )

        return xr.DataArray(
            arr.ravel(),
            dims=["time"],
            coords={"time": times},
            name=variable,
        )

    # ---------------------------------------------------------------- #
    # Public: Fetch
    # ---------------------------------------------------------------- #
    def fetch(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        collection: str = "aqc",
        mode: str = "assim",
        variables: Optional[List[str]] = None,
        level_type: Optional[str] = None,
        max_workers: int = 4,
    ) -> "GEOS_CF":
        """
        Fetch GEOS-CF data for a single site.

        Parameters
        ----------
        lat, lon : float
            Site coordinates (decimal degrees).
        start_date, end_date : str
            Date range as YYYY-MM-DD or YYYYMMDD.
        collection : 'aqc' | 'chm' | 'met'
            Data product. Defaults to 'aqc' (surface air quality).
        mode : 'assim' | 'fcast'
            'assim' for replay/historical; 'fcast' for 5-day forecast.
        variables : list of str, optional
            Variables to fetch. Uses collection defaults if None.
        level_type : 'v72' | 'v36' | 'p23' | 'x1', optional
            Vertical level scheme. Inferred from collection if None.
        max_workers : int
            Parallel threads (one per variable).

        Returns
        -------
        self — result stored in self.data[(lat_lon, mode, collection)]
        """
        query = GEOS_CF_QUERY(
            lat=lat, lon=lon,
            start_date=start_date, end_date=end_date,
            collection=collection, mode=mode,
            variables=variables, level_type=level_type,
        )

        lat_lon = f"{query.lat}x{query.lon}"
        has_levels = query.level_type is not None
        session = requests.Session()
        data_arrays: Dict[str, xr.DataArray] = {}

        def _fetch_one(variable: str):
            response_json = self._fetch_single(
                query.mode, query.collection, query.level_type,
                variable, lat_lon,
                query.start_date, query.end_date,
                session,
            )
            return variable, self._to_dataarray(response_json, variable, has_levels)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, var): var for var in query.variables}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"GEOS-CF {query.collection}/{query.mode}",
                ncols=100,
            ):
                var = futures[future]
                try:
                    var_name, da = future.result()
                    data_arrays[var_name] = da
                except Exception as e:
                    self._errors.append(f"Failed {var}: {e}")

        ds = xr.Dataset(data_arrays)
        ds.attrs.update({
            "lat": query.lat,
            "lon": query.lon,
            "lat_lon": lat_lon,
            "collection": query.collection,
            "mode": query.mode,
            "start_date": start_date,
            "end_date": end_date,
        })

        self.data[(lat_lon, query.mode, query.collection)] = ds
        return self

    # ---------------------------------------------------------------- #
    # Public: Altitude Coordinates
    # ---------------------------------------------------------------- #
    def add_altitude_coords(
        self,
        key: tuple,
        met_key: Optional[tuple] = None,
    ) -> "GEOS_CF":
        """
        Attach ZL (layer heights, km) from a MET dataset as an altitude
        coordinate on a profile dataset (CHM or MET).

        Parameters
        ----------
        key : tuple
            Key of the profile dataset to update, e.g.
            ("39.24x-76.363", "assim", "chm").
        met_key : tuple, optional
            Key of the MET dataset containing ZL. Inferred from key if None
            (same lat_lon and mode, collection="met").
        """
        ds = self.data.get(key)
        if ds is None:
            raise KeyError(f"No dataset found for {key}.")

        if met_key is None:
            lat_lon, mode, _ = key
            met_key = (lat_lon, mode, "met")

        met_ds = self.data.get(met_key)
        if met_ds is None or "ZL" not in met_ds:
            print(f"No MET/ZL data at {met_key}. Fetch met collection first.")
            return self

        # ZL shape: (time, level) — assign as a non-dimension coordinate
        self.data[key] = ds.assign_coords(altitude=met_ds["ZL"])
        return self


if __name__ == "__main__":
    geos = GEOS_CF()

    lat, lon = 39.24, -76.363
    start, end = "2025-07-28", "2025-08-01"

    # Surface AQC
    geos.fetch(lat=lat, lon=lon, start_date=start, end_date=end,
               collection="aqc", mode="assim")

    # CHM profiles
    geos.fetch(lat=lat, lon=lon, start_date=start, end_date=end,
               collection="chm", mode="assim", variables=["O3", "NO2"])

    # MET (for altitude coords)
    geos.fetch(lat=lat, lon=lon, start_date=start, end_date=end,
               collection="met", mode="assim", variables=["ZL", "T"])

    # Attach altitude to CHM
    geos.add_altitude_coords(key=(f"{lat}x{lon}", "assim", "chm"))

    print(geos.data[(f"{lat}x{lon}", "assim", "aqc")])
    print(geos.data[(f"{lat}x{lon}", "assim", "chm")])
