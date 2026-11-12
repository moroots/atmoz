# -*- coding: utf-8 -*-
"""
Created on 2026-06-10

@author: Maurice Roots

A Module for Working with NASA GEOS-CF Data via OPeNDAP.
Single-site point queries returning xarray Datasets.

Data access: https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/
Units: gas species are in mol/mol — multiply by 1e9 to convert to ppb.
"""

#%%

# Math & Data
import numpy as np
import pandas as pd
import xarray as xr

# Housekeeping
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Optional

# Validation
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Progress
from tqdm import tqdm


# --------------------------------------------------------------------------------------------------------------------------------- #
# OPeNDAP Collection Catalog
# --------------------------------------------------------------------------------------------------------------------------------- #
_OPENDAP_BASE = "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf"

_COLLECTION_CATALOG: Dict[str, dict] = {
    "aqc": {
        "urls": {
            "assim": f"{_OPENDAP_BASE}/assim/aqc_tavg_1hr_g1440x721_v1",
            "fcast": f"{_OPENDAP_BASE}/fcast/aqc_tavg_1hr_g1440x721_v1",
        },
        "variables": ["o3", "no2", "co", "so2", "pm25_rh35_gcc"],
        "level_type": None,
        "units": "mol/mol (gas), µg/m³ (pm25)",
        "description": "Surface air quality — 1-hr average (2018–2026)",
    },
    "chm": {
        "urls": {
            "assim": f"{_OPENDAP_BASE}/assim/chm_tavg_1hr_g1440x721_v1",
        },
        "variables": ["o3", "no2", "co", "so2", "oh", "hno3"],
        "level_type": "lev",
        "units": "mol/mol",
        "description": "Chemistry profiles — 1-hr average, 72 model levels (2018–2026)",
    },
    "met": {
        "urls": {
            "assim": f"{_OPENDAP_BASE}/assim/met_tavg_1hr_g1440x721_x1",
        },
        "variables": ["t2m", "u10m", "v10m", "q2m", "zpbl", "ps"],
        "level_type": None,
        "units": "SI (K, m/s, kg/kg, Pa)",
        "description": "Meteorology surface — 1-hr average (2018–2026)",
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

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: str) -> str:
        """Accept YYYY-MM-DD or YYYYMMDD; store as YYYY-MM-DD for xarray slicing."""
        v = str(v).strip()
        if len(v) == 10 and "-" in v:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        elif len(v) == 8:
            dt = datetime.strptime(v, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        raise ValueError(f"'{v}' must be YYYY-MM-DD or YYYYMMDD.")

    @model_validator(mode="after")
    def _valid_date_order(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")
        return self

    @model_validator(mode="after")
    def _fill_collection_defaults(self):
        if self.variables is None:
            self.variables = list(_COLLECTION_CATALOG.get(self.collection, {}).get("variables", []))
        return self


# --------------------------------------------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------------------------------------------- #
def _monthly_date_chunks(start: str, end: str) -> List[tuple]:
    """Return (chunk_start, chunk_end) pairs that cover [start, end] in monthly steps."""
    periods = pd.period_range(start=start, end=end, freq="M")
    chunks = []
    for period in periods:
        cs = max(pd.Timestamp(start), period.start_time).strftime("%Y-%m-%d")
        ce = min(pd.Timestamp(end), period.end_time).strftime("%Y-%m-%d")
        chunks.append((cs, ce))
    if not chunks:
        chunks = [(start, end)]
    return chunks


# --------------------------------------------------------------------------------------------------------------------------------- #
# GEOS_CF Class
# --------------------------------------------------------------------------------------------------------------------------------- #
class GEOS_CF:
    """
    Single-site client for NASA GEOS-CF via OPeNDAP.

    Fetches AQC (surface), CHM (chemistry profiles), and MET (met profiles)
    at a single lat/lon point using server-side subsetting and stores results
    as xr.Dataset objects.

    Data keys: (lat_lon_str, mode, collection)
      e.g. ("39.0x-77.0", "assim", "aqc")

    Units: gas species are in mol/mol from OPeNDAP.
      → multiply by 1e9 to convert to ppb
      → multiply by 1e6 to convert to ppm
    """

    def __init__(self):
        self.data: Dict[tuple, xr.Dataset] = {}
        self._errors: List[str] = []
        return

    @property
    def catalog(self) -> dict:
        """Available collections, variables, and OPeNDAP URLs."""
        return _COLLECTION_CATALOG

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
        cache_dir: Optional[str] = None,
        force_download: bool = False,
    ) -> "GEOS_CF":
        """
        Fetch GEOS-CF data for a single site via OPeNDAP.

        Parameters
        ----------
        lat, lon : float
            Site coordinates (decimal degrees).
        start_date, end_date : str
            Date range as YYYY-MM-DD or YYYYMMDD.
        collection : 'aqc' | 'chm' | 'met'
            Data product. Defaults to 'aqc' (surface air quality).
        mode : 'assim' | 'fcast'
            'assim' for replay/historical; 'fcast' for forecast.
        variables : list of str, optional
            Variables to fetch (lowercase). Uses collection defaults if None.
        cache_dir : str, optional
            Directory for local NetCDF cache. Avoids re-downloading on
            repeated calls with identical parameters.
        force_download : bool
            Ignore cache and re-download even if a cached file exists.

        Returns
        -------
        self — result stored in self.data[(lat_lon, mode, collection)]

        Notes
        -----
        Gas-phase species (o3, no2, co, so2) are returned in mol/mol.
        Multiply by 1e9 to convert to ppb.
        """
        import hashlib, os, pathlib

        query = GEOS_CF_QUERY(
            lat=lat, lon=lon,
            start_date=start_date, end_date=end_date,
            collection=collection, mode=mode,
            variables=variables,
        )

        catalog_entry = _COLLECTION_CATALOG.get(query.collection, {})
        url = catalog_entry.get("urls", {}).get(query.mode)
        if url is None:
            raise ValueError(
                f"Mode '{query.mode}' is not available for collection '{query.collection}'."
                f" Available modes: {list(catalog_entry.get('urls', {}).keys())}"
            )

        vars_lower = [v.lower() for v in query.variables]
        lat_lon = f"{query.lat}x{query.lon}"

        # Cache lookup — snap lat/lon to the nearest 0.25° grid cell so that
        # sites within the same OPeNDAP grid cell share one cache file.
        _snap = lambda v: round(round(v / 0.25) * 0.25, 4)
        cache_path = None
        if cache_dir is not None:
            cache_key = hashlib.md5(
                f"{_snap(query.lat)}x{_snap(query.lon)}_{query.mode}_{query.collection}_{query.start_date}_{query.end_date}_{'_'.join(sorted(vars_lower))}".encode()
            ).hexdigest()[:12]
            cache_path = pathlib.Path(cache_dir) / f"geos_cf_{cache_key}.nc"
            if cache_path.exists() and not force_download:
                print(f"Loading from cache: {cache_path}")
                ds_loaded = xr.open_dataset(cache_path).load()
                self.data[(lat_lon, query.mode, query.collection)] = ds_loaded
                return self

        try:
            ds = xr.open_dataset(url, engine="netcdf4")

            # Server-side point selection (nearest grid cell)
            ds_site = ds.sel(lat=query.lat, lon=query.lon, method="nearest")

            # Variable selection — validate against full dataset before chunking
            available = [v for v in vars_lower if v in ds_site.data_vars]
            missing = set(vars_lower) - set(available)
            if missing:
                print(f"Warning: {missing} not found in {collection}. Available: {list(ds_site.data_vars)}")
            if not available:
                raise KeyError(f"None of {vars_lower} found in dataset.")

            # OPeNDAP servers cap per-request time slices; chunk into monthly
            # windows and fetch in parallel to stay within limits.
            chunks = _monthly_date_chunks(query.start_date, query.end_date)
            n_workers = min(4, len(chunks))
            print(
                f"Loading {collection}/{mode} from OPeNDAP "
                f"({len(available)} variable(s), {len(chunks)} monthly chunk(s), "
                f"{n_workers} workers)..."
            )

            def _fetch_chunk(args):
                cs, ce, chunk_url = args
                ds_c = xr.open_dataset(chunk_url, engine="netcdf4")
                ds_c = ds_c.sel(lat=query.lat, lon=query.lon, method="nearest")
                ds_c = ds_c.sel(time=slice(cs, ce))
                return cs, ds_c[available].load()

            chunk_args = [(cs, ce, url) for cs, ce in chunks]
            results = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_fetch_chunk, arg): arg[0] for arg in chunk_args}
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{collection}/{mode}"):
                    cs, ds_chunk = fut.result()
                    results[cs] = ds_chunk

            ordered = [results[cs] for cs, _ in chunks]
            ds_loaded = xr.concat(ordered, dim="time") if len(ordered) > 1 else ordered[0]

        except Exception as e:
            self._errors.append(f"fetch({collection}/{mode}): {e}")
            raise

        ds_loaded.attrs.update({
            "lat_requested": query.lat,
            "lon_requested": query.lon,
            "lat_actual": float(ds_loaded["lat"].values) if "lat" in ds_loaded.coords else query.lat,
            "lon_actual": float(ds_loaded["lon"].values) if "lon" in ds_loaded.coords else query.lon,
            "collection": query.collection,
            "mode": query.mode,
            "start_date": query.start_date,
            "end_date": query.end_date,
            "units_note": catalog_entry.get("units", "see GEOS-CF documentation"),
        })

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ds_loaded.to_netcdf(cache_path)
            print(f"Cached to: {cache_path}")

        self.data[(lat_lon, query.mode, query.collection)] = ds_loaded
        return self

    # ---------------------------------------------------------------- #
    # Public: Altitude Coordinates (for CHM profiles)
    # ---------------------------------------------------------------- #
    def add_altitude_coords(
        self,
        key: tuple,
        met_key: Optional[tuple] = None,
    ) -> "GEOS_CF":
        """
        Attach ZL (layer heights, km) from a MET dataset as an altitude
        coordinate on a CHM profile dataset.

        Parameters
        ----------
        key : tuple
            Key of the CHM dataset to update.
        met_key : tuple, optional
            Key of the MET dataset containing ZL. Inferred from key if None.
        """
        ds = self.data.get(key)
        if ds is None:
            raise KeyError(f"No dataset found for {key}.")

        if met_key is None:
            lat_lon, mode, _ = key
            met_key = (lat_lon, mode, "met")

        met_ds = self.data.get(met_key)
        if met_ds is None or "zl" not in met_ds:
            print(f"No MET/ZL data at {met_key}. Fetch met collection with 'zl' first.")
            return self

        self.data[key] = ds.assign_coords(altitude=met_ds["zl"])
        return self


if __name__ == "__main__":
    geos = GEOS_CF()

    lat, lon = 39.05, -76.87
    start, end = "2025-07-28", "2025-08-01"

    # Surface AQC — O3 only
    geos.fetch(lat=lat, lon=lon, start_date=start, end_date=end,
               collection="aqc", mode="assim", variables=["o3"])

    key = next(iter(geos.data))
    ds = geos.data[key]
    print(ds)

    # Convert mol/mol -> ppb
    o3_ppb = ds["o3"] * 1e9
    print(f"O3 range: {float(o3_ppb.min()):.1f} – {float(o3_ppb.max()):.1f} ppb")
