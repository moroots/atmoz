# -*- coding: utf-8 -*-
"""
Created on 2026-02-19 18:24:19

@author: Maurice Roots

Description:
     - A module for accessing and using TOLNet datasets. 
"""
#%% 

from collections import namedtuple
from functools import cached_property
from urllib.parse import urljoin, urlencode, urlparse, urlunparse

from datetime import datetime
import pandas as pd

from functools import cached_property, cache
from typing import Any, Dict, List, Literal, Optional, Type, Union
from pydantic import BaseModel, ValidationError, Field, field_validator, model_validator, ConfigDict
import requests
import yaml


components = namedtuple(
    typename ='components', 
    field_names = ['scheme', 'netloc', 'path', 'params', 'query', 'fragment']
    )

class TOLNET_DATA_QUERY(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # Fields that accept str, int, or list
    instrument_group: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="instrument_group_name", default=None)
    product_type: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="product_type_name", default=None)
    processing_type: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="processing_type_name", default=None)
    file_type: Optional[Union[str, int, List[Union[str, int]]]] = Field(alias="file_type_name", default=None)

    # Boolean Fields
    near_real_time: Optional[Literal["true", "false"]] = Field(alias="near_real_time", default="true")

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
            "file_name"
            ]
        ] = Field(
            default="data_date",
            description="Valid values are data_date, upload_date, instrument_group_name, product_type_name, processing_type_name, file_type_name, or file_name"
            )
    
    order_direction: Optional[
        Literal[
            "asc", 
            "desc"
            ]
        ] = Field(
            default="desc",
            description="Valid values are asc or desc"
            )
    
    @model_validator(mode="after")
    def valid_bbox(self):
        bbox_fields = [self.minLatitude, self.maxLatitude, self.minLongitude, self.maxLongitude]
        if all(field is not None for field in bbox_fields):
            for f in ["latitude", "longitude", "radius"]:
                setattr(self, f, None)
                raise Warning(f"Do not use {f} and bbox at same time. Ignoring {f} in favor of bbox.")

            # Check that minLatitude < maxLatitude and minLongitude < maxLongitude
            if not (self.minLatitude < self.maxLatitude):
                raise ValueError("minLatitude must be less than maxLatitude.")
            if not (self.minLongitude < self.maxLongitude):
                raise ValueError("minLongitude must be less than maxLongitude.")
            
        # If any of the bbox fields are provided, all must be provided
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
                    # Check if the date is in YYYY-MM-DD format
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
            return ",".join(str(x) for x in v)   # [1,2,3] -> "1,2,3"
        if isinstance(v, (int, str)):
            return str(v)                         # 1 -> "1", "4" -> "4"
        raise TypeError("Expected int, str, or list of int/str")

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
            "file_size": "int16",
            "latitude": "int16",
            "longitude": "int16",
            "altitude": "int16",
            "isAccessible": "bool",
            }
        self.data = {}


    @cached_property
    def api_schema(self):
        """
        Returns a yaml object containing api_schema information
        """
        response = requests.get(self.base_url + "/openapi.yml")
        response.raise_for_status()
        return yaml.safe_load(response.text)

    @cached_property
    def products(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all product types.
        Contains:
            - id
            - processing_type_name
            - description
            - display_order
            - public
            - show_on_graph_page
        """
        response = requests.get(self.base_url + r"/data/product_types")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    @cached_property
    def file_types(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all file types.
        Contains:
            - id
            - file_type_name
            - description
            - display_order
            - public
        """
        response = requests.get(self.base_url + r"/data/file_types")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    @cached_property
    def instrument_groups(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all instrument groups.
        Contains:
            - id
            - instrument_group_name
            - folder_name
            - description
            - display_order
            - current_pi(Principle Investigator)
            - doi
            - citation_url
        """
        response = requests.get(self.base_url + r"/instruments/groups")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    @cached_property
    def processing_types(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing all processing types.
        Contains:
            - id
            - processing_type_name
            - description
            - display_order
            - public
            - show_on_graph_page.
        """
        response = requests.get(self.base_url + r"/data/processing_types")
        response.raise_for_status()
        return (
            pd.DataFrame(response.json())
            .sort_values(by=["id"])
            .set_index("id", drop=True)
            )

    def _get_files_list(self, **params):
        query = TOLNET_DATA_QUERY(**params)
        i = 1
        url = urlunparse(
            components(
                scheme="https",
                netloc="tolnet.larc.nasa.gov",
                path=f"/api/data/{i}",
                params="",
                query=urlencode(query.model_dump( exclude_none=True)),
                fragment=""
            )
        )
        response = requests.get(url)
        data_frames = []
        while response.status_code == 200:
            data_frames.append(pd.DataFrame(response.json()))
            i += 1
            url = urlunparse(
            components(
                scheme="https",
                netloc="tolnet.larc.nasa.gov",
                path=f"/api/data/{i}",
                params="",
                query=urlencode(query.model_dump( exclude_none=True)),
                fragment=""
                )
            )
            response = requests.get(url)

        df = pd.concat(data_frames, ignore_index=True)
        df["start_data_date"] = pd.to_datetime(df["start_data_date"])
        df["end_data_date"] = pd.to_datetime(df["end_data_date"])
        df["upload_date"] = pd.to_datetime(df["upload_date"])
        
        return df.astype(self.file_list_dtypes)
