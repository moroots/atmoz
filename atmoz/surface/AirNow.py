# -*- coding: utf-8 -*-
"""
Created on 2025-10-02 09:24:37

@author: Maurice Roots

Description:
     - Pulling Data from the AirNow API
"""

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

from atmoz.resources.useful_functions import merge_dicts

from dataclasses import dataclass, field, fields, MISSING, asdict

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

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                if f.default_factory is not MISSING:
                    setattr(self, f.name, f.default_factory())
                elif f.default is not MISSING:
                    setattr(self, f.name, f.default)
    
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

    Methods:
        _build_url(**kwargs): Constructs a query URL for the AirNow API.
        _pull(**kwargs): Fetches data from the AirNow API.
        _process(response, **kwargs): Converts API response to pandas DataFrame.
        _nest(df, metadata, **kwargs): Nests DataFrame by pollutant and station ID.
    """

    def __init__(self, api_key=None):
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
        self._queries_strings = []
        self._data = []
        return

    def _build_url(self, **kwargs):
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

        self._queries_strings.append( '&'.join(f"{key}={value}" for key, value in query_params.items()) ) 
        return f"{self.base_url}?{self._queries_strings[-1]}"

    def _pull(self, **kwargs): 
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
    
    def _process(self, response: requests.Response, **kwargs):  
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

    def _metadata(self, df: pd.DataFrame, **kwargs): 
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

    def _nest(self, df: pd.DataFrame, metadata: pd.DataFrame, **kwargs):
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
                    date_start=None, 
                    date_end=None,  
                    **kwargs
                    ):
        
        silent = kwargs.pop("silent", True)

        list_of_params_objs = []
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

        list_of_dfs = [
            pd.DataFrame(
                self._pull(**asdict(obj)).json()
            )
            for obj in iterator
        ]
        
        data = pd.concat(list_of_dfs) if len(list_of_dfs) > 1 else list_of_dfs[0]
        data.reset_index(drop=True, inplace=True)
        metadata = self._metadata(data)
        dataset = self._nest(data, metadata)
        return dataset, metadata


if __name__ == "__main__": 
    airnow = AirNow() 
    dataset, metadata = airnow.import_data()

    print("Metadata:")
    print(metadata.head())
    print("\nDataset keys (pollutants):")
    print(dataset.keys())