# -*- coding: utf-8 -*-
"""
Created on 2025-10-02 09:24:37

@author: Maurice Roots

Description:
     - Pulling Data from the AirNow API
"""
#%% 

#%%
# Importing Packages
from __future__ import annotations

from collections import namedtuple
from urllib.parse import urljoin, urlencode, urlparse, urlunparse
from datetime import datetime, timedelta, UTC
import keyring 
import requests 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from atmoz.resources import useful_functions as utils

from dataclasses import dataclass, field, fields, MISSING, asdict

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import io
 
from atmoz.resources.sessionHandler import SessionHandler
from atmoz.resources.parallelExecutor import ParallelExecutor, JobResult

from atmoz.resources.atmoz_dataclasses import atmoz_dataset

# ---------------------------------------------------------------------------
# Module-level session handler — one instance, shared across all threads.
# Each thread transparently gets its own requests.Session via threading.local.
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
_BASE_URL_AQS     = "https://aqs.epa.gov/aqsweb/airdata/"
_BASE_URL_AIRNOW  = "https://www.airnowapi.org/aq/data/"


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

class epa_pregen:
    """
    Simple class to efficiently handle downloading and processing the EPA PreGenerated Files.

    Downloads are executed in parallel using 'atmoz.resources.parallelExecutor' (mode ="threads"). 
    HTTP sessions are manaded by 'atmoz.resources.sessionHandler' to ensure thread safety and efficient connection pooling.
    """

    parameters = EPA_PARAMETERS
    base_url_aqs = _BASE_URL_AQS
           
    # O(1) lookup from Parameter Code -> Resolutions
    param_code_map = {
        v["code"]: v for v in EPA_PARAMETERS.values()
        }
    
    def __init__(self) -> None:
        pass

    # - Validation of Parameter <-> Reslution Combos - #
    @classmethod
    def _parameters_validator(cls, 
                               parameters: Union[str, List],
                               resolutions: Union[str, List],
                               years: Union[str, List],
                               **kwargs
                               ) -> List[Dict[str, Any]]:

        """
        Validation of Parameter <-> Resolution Combos for EPA PreGenerated Files.
        Depends on 'EPA_PARAMETERS' as a quikc lookup for valid combinations.

        Returns a list of job dicts ready to be passed directly to the parallel executor,
        with keys: "resolution", "parameter", "year". 
        """
        
        if not kwargs.get("silent", True):
            print("Validating parameter-resolution-year combinations...")

        if isinstance(parameters, str):
            parameters = [parameters]
        if isinstance(resolutions, str):
            resolutions = [resolutions]
        if isinstance(years, str):
            years = [years]
        
        seen: set[tuple] = set()
        combos: List[Dict[str, Any]] = []

        for y in years: 
            for p in parameters: 
                for r in resolutions: 

                    if not p.isdigit():
                        p = cls.parameters[p.lower()]["code"]
                    
                    # No Duplicates
                    key = (r, p, y)
                    if key in seen:
                        continue
                    seen.add(key)

                    if cls.param_code_map.get(p, {}).get(r, False):
                        combos.append({
                            "resolution": r,
                            "parameter": p,
                            "year": y
                            })
                    else:
                        if not kwargs.get("silent", True): 
                            print(f"Parameter: {p} does not have Resolution: {r}")
        
        if not kwargs.get("silent", True):
            print(f"Validation complete. {len(combos)} valid combinations found.")
            for combo in combos:
                print(f"  [VALID] {combo['parameter']} at {combo['resolution']} resolution for {combo['year']}")

        return combos

    # ------------------------------------------------------------------
    # Single-file download  (plain classmethod — called by the executor)
    # ------------------------------------------------------------------
    @classmethod
    def _download_single(cls, 
                         resolution: str,
                         parameter: str, 
                         year: int, 
                         session_handler = _session_handler,
                         **kwargs
                         ) -> pd.DataFrame:
        
        """Download one ZIP file and return a ``JobReturn`` dict.
 
        The HTTP session is obtained from the module-level ``SessionHandler``.
        Each worker thread automatically receives its own ``requests.Session``
        via ``threading.local``; no session is passed through the job dict
        (sessions are not serialisable and must not cross thread boundaries
        as shared objects).
 
        Returns
        -------
        dict
            ``{"key": filename, "value": DataFrame}`` — the ``JobReturn``
            contract expected by ``ParallelExecutor``.
        """

        if not kwargs.get("silent", True):
            print(f"Downloading {parameter} at {resolution} resolution for {year}...")
        
        dtypes = {
            "State Code": str,
            "County Code": str,
            "Site Num": str,
            "Parameter Code": str,
            "POC": str,
            "Latitude": str, 
            "Longitude": str, 
            "Datum": str, 
            "Parameter Name": str, 
            "Date Local": str, 
            "Time Loca": str, 
            "Date GMT": str, 
            "Time GMT": str, 
            "Sample Measurement": float, 
            "Units of Measure": str, 
            "MDL": str, 
            "Uncertainty": float, 
            "Qualifier": str, 
            "Method Type": str, 
            "Method Code": str, 
            "Method Name": str, 
            "State Name": str, 
            "County Name": str, 
            "Date of Last Change": str,
            }
        
        session = session_handler.session() 
        url = f"{cls.base_url_aqs}/{resolution}_{parameter}_{year}.zip"
        filename = f"{resolution}_{parameter}_{year}.csv"

        zip_file = utils.download_zip(url, session)
        with zip_file.open(filename) as f:
            df = pd.read_csv(f, low_memory=False, dtype=dtypes)

        return {"key": filename, "value": df}

    # ------------------------------------------------------------------
    # Parallel Downloader
    # ------------------------------------------------------------------
    @classmethod
    def _download(cls,
                  parameters: Union[str, List[str]],
                  resolutions: Union[str, List[str]],
                  years: Union[int, List[int]],
                  max_workers: int = 5,
                  show_traceback: bool = False,
                  **kwargs
                  ) -> Dict[str, pd.DataFrame]:
        
        """Download all valid (parameter, resolution, year) combinations in parallel.
 
        Parameters
        ----------
        parameters:
            One or more parameter names or numeric codes.
        resolutions:
            One or more resolution strings (e.g. ``"hourly"``, ``"daily"``).
        years:
            One or more calendar years.
        max_workers:
            Number of parallel download threads (default 5).
        show_traceback:
            If ``True``, print full tracebacks for failed downloads instead of
            a one-line summary.
 
        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of ``filename -> DataFrame`` for every successful download.
            Failed downloads are logged to stdout; call ``_download_result``
            variant if you need structured access to failures.
        """

        if not kwargs.get("silent", True):
            print(f"Starting parallel download with {max_workers} workers...")
            print(f"Parameters: {parameters}")
            print(f"Resolutions: {resolutions}")
            print(f"Years: {years}")

        combos = cls._parameters_validator(parameters, resolutions, years)
        if not combos:
            return {}
 
        # max_workers without touching module-level state.
        executor = ParallelExecutor(
            max_workers=max_workers,
            desc="Downloading EPA data",
            mode="thread",          
            show_traceback=show_traceback,
            )
        
        download_parallel = executor(cls._download_single)
 
        result: JobResult = download_parallel(combos)
 
        if result.failed:
            print(
                f"\n{len(result.failed)} download(s) failed. "
                "Retry the failed jobs or inspect result.failed for details."
            )
            for failure in result.failed:
                job_label = (
                    "{resolution}_{parameter}_{year}".format(**failure.job)
                )
                print(f"  [FAILED] {job_label}: {failure.exc}")
 
        return result.succeeded


# ---------------------------------------- #
# AirNow API Handler Class
# ---------------------------------------- #
@dataclass
class AirNowParams:
    startDate: str = field(default_factory=lambda: (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H"))
    endDate: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H"))
    parameters: list = field(default_factory=lambda: ["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"])
    BBOX: list = field(default_factory=lambda: ["-80.655479", "35.574398", "-72.086143", "41.415693"])
    dataType: str = field(default='B')
    format: str = field(default='application/json')
    verbose: str = field(default='1')
    monitorType: str = field(default='0')
    includerawconcentrations: str = field(default='1')

    def __post_init__(self) -> None:
        """
        Ensures that any field set to None is replaced with its default value.
        """
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                if f.default_factory is not MISSING:
                    setattr(self, f.name, f.default_factory())
                elif f.default is not MISSING:
                    setattr(self, f.name, f.default)
        return



class AirNow:
    """
    A class to interact with the AirNow API for fetching and processing air quality data.

    Features:
        - Handles authentication via API key (from argument or system keyring).
        - Builds API request URLs with flexible parameters (date range, pollutants, bounding box, etc.).
        - Fetches data from the AirNow API and handles HTTP responses.
        - Processes API responses into pandas DataFrames.
        - Organizes air quality data into nested dictionaries by pollutant and station.

    Attributes:
        api_key (str): API key for AirNow API access.
        base_url (str): Base endpoint for AirNow API.
        _queries_strings (list): Stores query strings for API requests.
        _data (list): Stores fetched data as pandas DataFrames.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize AirNow API handler.

        Args:
            api_key (str, optional): API key for AirNow API. If None, retrieves from keyring.
        """
        if api_key is None:
            try:
                self.api_key = keyring.get_password("EPA_AirNow", "API_KEY")
            except Exception as e:
                self.api_key = input("EPA AIRNOW API KEY: ")
        else:
            self.api_key = api_key

        self.base_url = "https://www.airnowapi.org/aq/data/"
        self._queries_strings: List[str] = []
        self._data: List[pd.DataFrame] = []
        return

    def _build_url(self, **kwargs) -> str:
        """
        Build a URL for querying AirNow surface data.

        Keyword Args:
            startDate (str): Start date/time in "%Y-%m-%dT%H" format (default: 24h ago UTC).
            endDate (str): End date/time in "%Y-%m-%dT%H" format (default: now UTC).
            parameters (list of str): Pollutants to include (default: all).
            BBOX (list of str): Bounding box [minLon, minLat, maxLon, maxLat] (default: preset region).
            dataType (str): Data type (default: 'B').
            format (str): Response format (default: 'application/json').
            verbose (str): Verbosity flag (default: "1").
            monitorType (str): Monitor type (default: "0").
            includerawconcentrations (str): Include raw concentrations (default: "1").
            API_KEY (str): API key (set automatically).

        Returns:
            str: Constructed URL with query parameters.
        """
        if kwargs.get("api_key", None):
            API_KEY = kwargs.get("api_key")
        else:
            API_KEY = self.api_key or input("EPA AIRNOW API KEY: ")

        startDate = kwargs.get("startDate", (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H"))
        endDate = kwargs.get("endDate", (datetime.now(UTC)).strftime("%Y-%m-%dT%H"))
        parameters = kwargs.get("parameters", ["OZONE", "PM25", "PM10", "CO", "NO2", "SO2"])
        BBOX = kwargs.get("BBOX", ["-80.655479", "35.574398", "-72.086143", "41.415693"])

        query_params = {
            'startDate': startDate,
            'endDate': endDate,
            'parameters': ','.join(parameters),
            'BBOX': ','.join(BBOX),
            'dataType': kwargs.get("dataType", 'B'),
            'format': kwargs.get("format", 'application/json'),
            'verbose': kwargs.get("verbose", "1"),
            'monitorType': kwargs.get("monitorType", "0"),
            'includerawconcentrations': "1",
            'API_KEY': API_KEY
            }

        self._queries_strings.append('&'.join(f"{key}={value}" for key, value in query_params.items()))
        return f"{self.base_url}?{self._queries_strings[-1]}"

    def _pull(self, **kwargs) -> requests.Response:
        """
        Fetch EPA AirNow data using the API key.

        If the API key is not set, prompts for input and stores it in the keyring.
        Builds the request URL and sends a GET request.

        Parameters:
            **kwargs: Arguments for building the API request URL.

        Returns:
            requests.Response: API response object.

        Raises:
            Exception: If the API request fails.
        """
        if not self.api_key:
            response = input("EPA AIRNOW API KEY: ")
            if response:
                self.api_key = response
                keyring.set_password("EPA_AirNow", "API_KEY", self.api_key)
            else:
                raise ValueError("An API key is required to access the AirNow API.")

        url = self._build_url(api_key=self.api_key, **kwargs)
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    def _process(self, response: requests.Response, **kwargs) -> None:
        """
        Process the HTTP response and store as a pandas DataFrame.

        Args:
            response (requests.Response): HTTP response with JSON data.
            **kwargs: Unused.

        Side Effects:
            Appends a DataFrame to self._data.
        """
        self._data.append(pd.DataFrame(response.json()))
        return

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

    def import_data(self,
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    **kwargs
                    ) -> Tuple[Dict[str, Dict[Any, pd.DataFrame]], pd.DataFrame]:
        """
        Imports data from the AirNow API within a specified date range and returns the dataset along with its metadata.

        Parameters:
            start_date (str or datetime, optional): The start date for data import. If provided with `end_date`, data will be fetched for each day in the range.
            end_date (str or datetime, optional): The end date for data import. Used with `start_date` to define the date range.
            **kwargs: Additional keyword arguments to be passed to the AirNowParams constructor and the internal data pulling method. 
                - silent (bool, optional): If True (default), displays a progress bar during data download.

        Returns:
            tuple:
                - dataset (dict): The processed dataset containing the imported data, nested as required.
                - metadata (pd.DataFrame): Metadata associated with the imported dataset.

        Notes:
            - If both `start_date` and `end_date` are not provided, data is imported for the default parameters specified in `kwargs`.
            - Uses tqdm for progress indication if `silent` is True.
        """
        silent = kwargs.pop("silent", True)

        list_of_params_objs: List[AirNowParams] = []
        if start_date and end_date: 
            date_range = [t.strftime("%Y-%m-%dT%H") for t in pd.date_range(start=start_date, end=end_date, freq="1d")]
            list_of_params_objs.extend([
                AirNowParams(
                    startDate=date_range[i-1], 
                    endDate=date_range[i], 
                    **kwargs
                    ) 
                for i in range(1, len(date_range))
                ])
        else: 
            list_of_params_objs.extend([AirNowParams(**kwargs)])

        iterator = tqdm(list_of_params_objs, desc="Downloading...", unit="req") if silent is True else list_of_params_objs

        list_of_dfs: List[pd.DataFrame] = [
            pd.DataFrame(
                self._pull(**asdict(obj)).json()
            )
            for obj in iterator
        ]
        
        data: pd.DataFrame = pd.concat(list_of_dfs) if len(list_of_dfs) > 1 else list_of_dfs[0]
        metadata: pd.DataFrame = self._metadata(data)
        # dataset: Dict[str, Dict[Any, pd.DataFrame]] = self._nest(data, metadata)
        return atmoz_dataset(data=data, metadata=metadata, datatype="surface")

    def download_data(self,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None,
                      output_dir: Optional[Path] = None,
                      max_workers: int = 3,
                      **kwargs
                      ) -> Path:
        """
        Imports data from the AirNow API within a specified date range and saves it to a Parquet file.

        Parameters:
            start_date (str or datetime, optional): The start date for data import. If provided with `end_date`, data will be fetched for each day in the range.
            end_date (str or datetime, optional): The end date for data import. Used with `start_date` to define the date range.
            **kwargs: Additional keyword arguments to be passed to the AirNowParams constructor and the internal data pulling method. 
                - silent (bool, optional): If True (default), displays a progress bar during data download.

        Returns:
            dict: A dictionary containing the imported data and metadata.

        Notes:
            - If both `start_date` and `end_date` are not provided, data is imported for the default parameters specified in `kwargs`.
            - Uses tqdm for progress indication if `silent` is True.
        """
        silent = kwargs.pop("silent", True)

        list_of_params_objs: List[AirNowParams] = []
        if start_date and end_date: 
            date_range = [t.strftime("%Y-%m-%dT%H") for t in pd.date_range(start=start_date, end=end_date, freq="1D")]
            list_of_params_objs.extend([
                AirNowParams(
                    startDate=date_range[i-1], 
                    endDate=date_range[i], 
                    **kwargs
                    ) 
                for i in range(1, len(date_range))
                ])
        else: 
            list_of_params_objs.extend([AirNowParams(**kwargs)])

        temp = []; futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for obj in list_of_params_objs:
                futures.append(executor.submit(self._pull, **asdict(obj)))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading AirNow data"):
                try:
                    df = pd.DataFrame(future.result().json())
                    temp.append(df)
                except Exception as e:
                    print(f"An error occurred: {e}")

        df = pd.concat(temp, ignore_index=True)
        full_query = AirNowParams(startDate=start_date, endDate=end_date, **kwargs)
        df.attrs = {"query_params": asdict(full_query)}
        filename = f"airnow_api_query_{start_date}_{end_date}.parquet"
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
        else: 
            filepath = Path(".") / filename
        
        df.to_parquet(filepath, index=False)
        return filepath

    @classmethod
    def download(cls, endpoint: str = "airnow", **kwargs) -> Path:
        if endpoint == "airnow":
            if not kwargs.get("api_key"):
                try:
                    kwargs["api_key"] = keyring.get_password("EPA_AirNow", "API_KEY")
                except Exception as e:
                    kwargs["api_key"] = input("EPA AIRNOW API KEY: ")
            return cls().import_data(**kwargs)
        elif endpoint == "aqs":
            return epa_pregen._download(**kwargs)
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}. Valid options are 'airnow' and 'aqs'.")
        


