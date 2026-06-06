# -*- coding: utf-8 -*-
"""
Created on 2025-10-02 09:24:37

@author: Maurice Roots

Description:
     - Pulling Data from the AirNow API
"""
#%% 

# Importing Packages

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

_BASE_URL_AQS     = "https://aqs.epa.gov/aqsweb/airdata"
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

class EPA_PREGEN:
    parameters = EPA_PARAMETERS
    base_url_aqs = _BASE_URL_AQS

    def __init__(self):
        pass
    
    @classmethod
    def _download_single(cls, 
                         resolution: str,
                         parameter: str, 
                         year: int, 
                         session: requests.Session
                         ) -> pd.DataFrame:
        try:
            url = f"{cls.base_url_aqs}/{resolution}_{parameter}_{year}.zip"
            zip_file = utils.download_zip(url, session)
            filename = f"{resolution}_{parameter}_{year}.csv"

            with zip_file.open(filename) as f:
                df = pd.read_csv(f, low_memory=False)

            return df, filename
        
        except Exception as e:
            print(f"Error downloading {parameter} at {resolution} resolution for year {year}: {e}")
            
    def download(parameters: Union[str, List],
                 resolutions: Union[str, List], 
                 years: Union[int, List],
                 output_dir: Optional[Path] = Path("./"),
                 save_as_parquet: bool = True,
                 **kwargs
                 ):

        session = requests.Session()
        
        #check that combinations of parameters are valid
        if isinstance(parameters, str):
            parameters = [parameters]
        if isinstance(resolutions, str):
            resolutions = [resolutions]
        if isinstance(years, int):
            years = [years]

        # Validate (parameter, resolution) combinations
        parameters = ["Ozone", "NO2", "44201"]
        resolutions = ["hourly", "daily", "annual"]
        years = ["2025", "2024"]

        param_codes = []
        res_combos = []
        for p in parameters: 
            if p.isdigit(): 
                param_codes.append(p)
            else:      
                param_codes.append(EPA_PARAMETERS[p.lower()]["code"])


        param_codes = np.unique(param_codes)

        # build O(1) lookup from AQS code -> parameter dict
        code_map = {v["code"]: v for v in EPA_PARAMETERS.values()}

        for p in param_codes: 
            for r in resolutions: 
                res_bool = code_map.get(p, {}).get(r, False)
                if res_bool:
                    res_combos.append(f"{r}_{p}")
                else: 
                    print(f"Parameter: {p} does not have Resolution: {r}")

        zip_filenames = [f"{r}_{y}.zip" for r in res_combos for y in years]


        return

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
            self.api_key = keyring.get_password("EPA_AirNow", "API_KEY")
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
            'API_KEY': self.api_key
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
                    date_start: Optional[Union[str, datetime]] = None,
                    date_end: Optional[Union[str, datetime]] = None,
                    **kwargs
                    ) -> Tuple[Dict[str, Dict[Any, pd.DataFrame]], pd.DataFrame]:
        """
        Imports data from the AirNow API within a specified date range and returns the dataset along with its metadata.

        Parameters:
            date_start (str or datetime, optional): The start date for data import. If provided with `date_end`, data will be fetched for each day in the range.
            date_end (str or datetime, optional): The end date for data import. Used with `date_start` to define the date range.
            **kwargs: Additional keyword arguments to be passed to the AirNowParams constructor and the internal data pulling method. 
                - silent (bool, optional): If True (default), displays a progress bar during data download.

        Returns:
            tuple:
                - dataset (dict): The processed dataset containing the imported data, nested as required.
                - metadata (pd.DataFrame): Metadata associated with the imported dataset.

        Notes:
            - If both `date_start` and `date_end` are not provided, data is imported for the default parameters specified in `kwargs`.
            - Uses tqdm for progress indication if `silent` is True.
        """
        silent = kwargs.pop("silent", True)

        list_of_params_objs: List[AirNowParams] = []
        if date_start and date_end: 
            date_range = [t.strftime("%Y-%m-%dT%H") for t in pd.date_range(start=date_start, end=date_end, freq="1d")]
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
        dataset: Dict[str, Dict[Any, pd.DataFrame]] = self._nest(data, metadata)
        return dataset, metadata

    def download_data(self,
                      date_start: Optional[Union[str, datetime]] = None,
                      date_end: Optional[Union[str, datetime]] = None,
                      output_dir: Optional[Path] = None,
                      max_workers: int = 3,
                      **kwargs
                      ) -> Path:
        """
        Imports data from the AirNow API within a specified date range and saves it to a Parquet file.

        Parameters:
            date_start (str or datetime, optional): The start date for data import. If provided with `date_end`, data will be fetched for each day in the range.
            date_end (str or datetime, optional): The end date for data import. Used with `date_start` to define the date range.
            **kwargs: Additional keyword arguments to be passed to the AirNowParams constructor and the internal data pulling method. 
                - silent (bool, optional): If True (default), displays a progress bar during data download.

        Returns:
            tuple:
                - dataset (dict): The processed dataset containing the imported data, nested as required.
                - metadata (pd.DataFrame): Metadata associated with the imported dataset.

        Notes:
            - If both `date_start` and `date_end` are not provided, data is imported for the default parameters specified in `kwargs`.
            - Uses tqdm for progress indication if `silent` is True.
        """
        silent = kwargs.pop("silent", True)

        list_of_params_objs: List[AirNowParams] = []
        if date_start and date_end: 
            date_range = [t.strftime("%Y-%m-%dT%H") for t in pd.date_range(start=date_start, end=date_end, freq="1D")]
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
        full_query = AirNowParams(startDate=date_start, endDate=date_end, **kwargs)
        df.attrs = {"query_params": asdict(full_query)}
        filename = f"airnow_api_query_{date_start}_{date_end}.parquet"
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
        else: 
            filepath = Path(".") / filename
        
        df.to_parquet(filepath, index=False)
        return filepath

#%%
if __name__ == "__main__": 
    airnow = AirNow() 
    dataset, metadata = airnow.import_data()

    print("Metadata:")
    print(metadata.head())
    print("\nDataset keys (pollutants):")
    print(dataset.keys())

# %%

def download_temp(parameters, resolutions, years, output_dir=None, save_as_parquet=True, **kwargs):

    session = requests.Session()

    # O(1) lookup from Parameter Code -> Resolutions
    code_map = {v["code"]: v for v in EPA_PARAMETERS.values()}

    params_combos = []
    for y in years: 
        for p in parameters: 
            for r in resolutions: 

                if not p.isdigit():
                    p = EPA_PARAMETERS[p.lower()]["code"]
                
                res_bool = code_map.get(p, {}).get(r, False)
                if res_bool:
                    params_combos.append({
                        "resolution": r,
                        "parameter": p,
                        "year": y,
                        "session": session,
                        })
                else: 
                    print(f"Parameter: {p} does not have Resolution: {r}")
    
    with ThreadPoolExecutor(max_workers=kwargs.get("max_workers", 5)) as executor:
        futures = [executor.submit(EPA_PREGEN._download_single, **combo) for combo in params_combos]
        results = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading data"):
            try:
                df, filename = future.result()
                results[filename] = df
            except Exception as e:
                print(f"An error occurred: {e}")

    return results

