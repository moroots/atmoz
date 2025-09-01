# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:22:28 2025

@author: Maurice Roots

A Module for Downloading and Plotting TOLNet Data

"""
#%% Import Packages

import yaml
import pandas as pd
import requests
from pathlib import Path
from functools import cached_property



#%% Classes

class utilities:
    def __init__(self):
        return

class filter_files:
    def __init__(self):
        return

class TOLNet(utilities):
    def __init__(self):
        super().__init__()
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
        return

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

    def files_list(self, min_date, max_date) -> pd.DataFrame:
        """
        Parameters
        ----------
        min_date : STR
            The starting date for the query, in YYYY-MM-DD format.
        max_date : STR
            The ending date for the query, in YYYY-MM-DD format.

        Returns
        -------
        A DataFrame containing all files from the TOLNet API that fall between the two provided dates.
        The DataFrame contains each file name as well as various descriptors.
        """

        i = 1
        url = f"https://tolnet.larc.nasa.gov/api/data/1?min_date={min_date}&max_date={max_date}&order=data_date&order_direction=desc"
        response = requests.get(url)
        data_frames = []
        while response.status_code == 200:
            data_frames.append(pd.DataFrame(response.json()))
            i += 1
            url = f"https://tolnet.larc.nasa.gov/api/data/{i}?min_date={min_date}&max_date={max_date}&order=data_date&order_direction=desc"
            response = requests.get(url)

        df = pd.concat(data_frames, ignore_index=True)
        df["start_data_date"] = pd.to_datetime(df["start_data_date"])
        df["end_data_date"] = pd.to_datetime(df["end_data_date"])
        df["upload_date"] = pd.to_datetime(df["upload_date"])
        return df.astype(self.file_list_dtypes)


if __name__ == "__main__":
    tolnet = TOLNet()

    tolnet.products



#%%



