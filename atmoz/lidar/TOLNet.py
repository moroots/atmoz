# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:22:28 2025

@author: Maurice Roots

A Module for Downloading and Plotting TOLNet Data

"""

# Math & Data
import pandas as pd
import numpy as np

# Housekeeping
import yaml
import string
import datetime
import requests
from dateutil import tz
from pathlib import Path
from typing import Union
import importlib.resources as resources
from functools import cached_property, cache

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits

# Multi-Threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal
from atmoz.resources import debug, plot_utilities, useful_functions, colorbars, debug, default_plot_params
from atmoz import models

# --------------------------------------------------------------------------------------------------------------------------------- #
# Filtering Request Return
# --------------------------------------------------------------------------------------------------------------------------------- #
class filter_files:
    def __init__(self, df, ptypes):
        self.df = df
        self.processing_types = ptypes
        return

    def daterange(self, min_date: str = None, max_date: str = None, **kwargs):
        try:
            self.df = self.df[
                (self.df["start_data_date"] >= min_date) & (self.df["start_data_date"] <= max_date)
            ]
        except Exception:
            pass
        return self

    def instrument_group(self, instrument_group: list = None, **kwargs):
        try:
            self.df = self.df[self.df["instrument_group_id"].isin(instrument_group)]
        except Exception:
            pass
        return self

    def product_type(self, product_type: list = None, **kwargs):
        try:
            self.df = self.df[self.df["product_type_id"].isin(product_type)]
        except Exception:
            pass
        return self

    def file_type(self, file_type: list = None, **kwargs):
        try:
            self.df = self.df[self.df["file_type_id"].isin(file_type)]
        except Exception:
            pass
        return self

    def processing_type(self, processing_type: list = None, **kwargs):
        """
        id 1 = centrally proccessed
        id 2 = in-house
        id 3 = unprocessed
        """
        try:
            processing_type_names = []
            types = self.processing_types
            for process in processing_type:
                processing_type_names.append(
                    list(types["processing_type_name"][types["id"] == process])[0]
                )

            self.df = self.df[self.df["processing_type_name"].isin(processing_type_names)]
        except Exception:
            pass
        return self

class TOLNet(debug.utilities):
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
        
        self.plot_theme = default_plot_params.curtain_plot_theme
        self.plot_params = default_plot_params.tolnet_plot_params
        self.data = {}
        self.troubleshoot["TOLNet"] = []
        self.watermark =  useful_functions.get_asset("Watermark_TOLNet.png")
        self.nrt = True
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

    def get_files_list(self, min_date, max_date) -> pd.DataFrame:
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

    def _add_timezone(self, time):
        return [utc.replace(tzinfo=tz.gettz("UTC")) for utc in time]

    def change_timezone(self, timezone: str):
        to_zone = tz.gettz(timezone)

        for key in self.data.keys():
            for filename in self.data[key].keys():
                time = self.data[key][filename].index.to_list()
                self.data[key][filename].index = [t.astimezone(to_zone) for t in time]

        return self

    def _json_to_dict(self, file_id: int):
        """
        Parameters
        ----------
        file_id : INT
            The ID of the file to retrieve from the API.

        Returns
        -------
        A dictionary containing the file's ozone values and metadata.
        """
        try:
            url = f"https://tolnet.larc.nasa.gov/api/data/json/{file_id}"
            response = requests.get(url).json()
        except Exception:
            self.troubleshoot["TOLNet"].append(f"Error with pulling {file_id}")
        return response

    def _unpack_data(self, meta_data):
        """
        Parameters
        ----------
        meta_data : A dictionary of a file's metadata

        Returns
        -------
        A DataFrame containing that same metadata

        """
        try:
            df = pd.DataFrame(
                meta_data["value"]["data"],
                columns=meta_data["altitude"]["data"],
                index=pd.to_datetime(meta_data["datetime"]["data"]),
                )
            df = df.apply(pd.to_numeric)
            df[df.isnull()] = np.nan
            df.sort_index(inplace=True)
        except Exception as e:
            self.troubleshoot["TOLNet"].append(f"_unpack_data: {e}")
            return

        return df

    def import_data(self, min_date, max_date, **kwargs):
        """
        Parameters
        ----------
        min_date : String
            The starting date to take from. Formatted as YYYY-MM-DD.
        max_date: String
            The ending date to take data from. Formatted as YYYY-MM-DD.

        """
        params = {"GEOS_CF": False}
        params.update(kwargs)

        def process_file(file_name, file_id):
            meta_data = self._json_to_dict(file_id)
            data = self._unpack_data(meta_data)
            data.index = self._add_timezone(data.index.to_list())
            return file_name, meta_data, data

        files = self.get_files_list(min_date, max_date)
        self.files = (
            filter_files(files, self.processing_types)
            .daterange(**kwargs)
            .instrument_group(**kwargs)
            .product_type(**kwargs)
            .file_type(**kwargs)
            .processing_type(**kwargs)
            .df
            )

        self.request_dates = (min_date, max_date)
        self.meta_data = {}

        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {
                executor.submit(process_file, file_name, file_id): file_name
                for file_name, file_id in zip(self.files["file_name"], self.files["id"])
                }

            for future in tqdm(as_completed(future_to_file),
                               total=len(future_to_file),
                               desc="Downloading TOLNet Data for",
                               ncols=100
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

                    if key not in self.data.keys():
                        self.data[key] = {}
                        self.meta_data[key] = {}

                    self.data[key][date] = data
                    self.meta_data[key][file_name] = meta_data

                except Exception as e:
                    self.troubleshoot["TOLNet"].append(f"Error processing file {file_name}: {e}")

        if params["GEOS_CF"]:
            keys = list(self.data.keys())
            for key in keys:
                lat_lon = key[2]
                models.geos_cf().get_geos_data_multithreaded(
                    lat_lon, 
                    self.request_dates[0], 
                    self.request_dates[1]
                    )

        return self

    def tolnet_curtain_plot(self, data: dict, **kwargs):
        params = useful_functions.merge_dicts(self.plot_params, kwargs)

        with plt.rc_context(self.plot_theme):
            fig, ax = plt.subplots()

            xlims = params.get("xlims", "auto")
            dates = sorted(list(data.keys()))

            if xlims == "auto": 
                pass   
            elif isinstance(xlims, list): 
                xlims = pd.to_datetime(xlims, utc=True)
                xlims = [xlims.min(), xlims.max()]
                dates = pd.to_datetime(dates, utc=True)
                dates = [ str(x.strftime("%Y-%m-%d")) for x in dates[(dates >= xlims[0]) & (dates <= xlims[1])] ]

            for date in dates:
                if xlims == "auto": 
                    df = data[date].copy()
                else:
                    df = data[date].copy()[xlims[0]:xlims[1]]

                if df.empty:
                    continue

                time_resolution = kwargs.get("time_resolution", "auto")

                if time_resolution == "auto": 
                    resolution = np.min([df.index[i] - df.index[i-1] for i in range(1, len(df))])
                    df = df.resample(f"{resolution.seconds}s").mean()
                else: 
                    df = df.resample(f"{time_resolution}").mean()

                X, Y, Z = ( df.index, df.columns, df.to_numpy().T )

                cmap, norm = colorbars.tolnet_ozone()

                if params.get("use_countourf", False):
                    levels = norm.boundaries
                    im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
                else:
                    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading="nearest", alpha=1)
            
            params["fig.colorbar"]["mappable"] = im

            plot_utilities.apply_plot_params(fig, ax, **params)

            plot_utilities.apply_datetime_axis(ax)

            if self.watermark: 
                plot_utilities.apply_watermark(fig, self.watermark)
            
            if self.nrt is True: 
                plot_utilities.apply_near_real_time(ax)
            
            plt.show()
        return 
    
if __name__ == "__main__":
    tolnet = TOLNet()

    tolnet.products
    tolnet.file_types
    tolnet.instrument_groups
    tolnet.processing_types

    date_start = "2024-08-08"
    date_end = "2024-08-09"
    product_IDs = [4]

    data = tolnet.import_data(
        min_date=date_start, 
        max_date=date_end, 
        product_type=product_IDs, 
        GEOS_CF=False
        )

    translator = str.maketrans({c: "_" for c in string.punctuation})

    tolnet.tolnet_curtain_plot(data.data[('NASA JPL SMOL-2', 'Centrally Processed (GLASS)', '40.89x-111.89')])





