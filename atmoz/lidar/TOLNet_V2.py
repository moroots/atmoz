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
from functools import cached_property, lru_cache, cache

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as image

from dateutil import tz

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

import string

import importlib.resources as resources
from pathlib import Path
from typing import Union

from atmoz.resources.utilities import merge_dicts


#%% Classes

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

class utilities:
    curtain_params = {
                "ylabel": "Altitude (km ASL)",
                "xlabel": "Datetime (UTC)",
                "fontsize_label": 18,
                "fontsize_ticks": 16,
                "fontsize_title": 20,
                "title": {
                    "label": r"$O_3$ Mixing Ratio Profile",
                    "fontsize": 16
                },
                "savefig": {
                    "fname": None,
                    "dpi": 300,
                    "transparent": True,
                    "format": "png",
                    "bbox_inches": "tight"
                },
                "ylims": [0, 12],
                "yticks": np.arange(0, 15.1, 1),
                "figsize": (30, 8),
                "layout": "tight",
                "cbar_label": "Ozone ($ppb_v$)",
                "fontsize_cbar": 16,
                "xlims": ["2025-06-15", "2025-08-15"],
                "grid": {
                    "visible": True,
                    "color": "gray",
                    "linestyle": "--",
                    "linewidth": 0.5
                }
            }

    def __init__(self):
        self.data = {}
        self.troubleshoot = {}
        return

    @cached_property
    def O3_curtain_colors(self):
        """
        Returns
        -------
        The color scheme used in the O3 curtain plots on the TOLNet website.

        """
        ncolors = [
            np.array([255, 140, 255]) / 255.0,
            np.array([221, 111, 242]) / 255.0,
            np.array([187, 82, 229]) / 255.0,
            np.array([153, 53, 216]) / 255.0,
            np.array([119, 24, 203]) / 255.0,
            np.array([0, 0, 187]) / 255.0,
            np.array([0, 44, 204]) / 255.0,
            np.array([0, 88, 221]) / 255.0,
            np.array([0, 132, 238]) / 255.0,
            np.array([0, 165, 255]) / 255.0,
            np.array([0, 235, 255]) / 255.0,
            np.array([39, 255, 215]) / 255.0,
            np.array([99, 255, 150]) / 255.0,
            np.array([163, 255, 91]) / 255.0,
            np.array([211, 255, 43]) / 255.0,
            np.array([255, 255, 0]) / 255.0,
            np.array([250, 200, 0]) / 255.0,
            np.array([255, 159, 0]) / 255.0,
            np.array([255, 111, 0]) / 255.0,
            np.array([255, 63, 0]) / 255.0,
            np.array([255, 0, 0]) / 255.0,
            np.array([216, 0, 15]) / 255.0,
            np.array([178, 0, 31]) / 255.0,
            np.array([140, 0, 47]) / 255.0,
            np.array([102, 0, 63]) / 255.0,
            np.array([200, 200, 200]) / 255.0,
            np.array([140, 140, 140]) / 255.0,
            np.array([80, 80, 80]) / 255.0,
            np.array([52, 52, 52]) / 255.0,
            np.array([0, 0, 0]),
        ]

        ncmap = mpl.colors.ListedColormap(ncolors)
        ncmap.set_under([1, 1, 1])
        ncmap.set_over([0, 0, 0])
        bounds = [0.001, *np.arange(5, 121, 5), 150, 200, 300, 600]
        nnorm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
        return ncmap, nnorm
    
    def _plot_settings(self, fig, ax, params, im):
        cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, *np.arange(10, 121, 10), 150, 200, 300, 600])
        cbar.set_label(label=params["cbar_label"], size=16)

        plt.setp(ax.get_xticklabels(), fontsize=params["fontsize_ticks"])
        plt.setp(ax.get_yticklabels(), fontsize=params["fontsize_ticks"])

        cbar.ax.tick_params(labelsize=params["fontsize_ticks"])
        plt.title(params["title"], fontsize=params["fontsize_title"])

        ax.set_ylabel(params["ylabel"], fontsize=params["fontsize_label"])
        ax.set_xlabel(params["xlabel"], fontsize=params["fontsize_label"])

        ax.set_xlim([np.datetime64(params["xlims"][0]), np.datetime64(params["xlims"][1])])

        ax.set_yticks(params["yticks"])

        ax.set_ylim(params["ylims"])

        converter = mdates.ConciseDateConverter()
        munits.registry[datetime.datetime] = converter
        ax.xaxis_date()

        if params["grid"]:
            ax.grid(
                color=params.get("grid_color", "gray"),
                linestyle=params.get("grid_linestyle", "--"),
                linewidth=params.get("grid_linewidth", 0.5),
            )
        return

    def curtain_plot(self, X, Y, Z, use_countourf=False, **kwargs):
        params = {
            "ylabel": "Altitude (km ASL)",
            "xlabel": "Datetime (UTC)",
            "fontsize_label": 18,
            "fontsize_ticks": 16,
            "fontsize_title": 20,
            "title": r"$O_3$ Mixing Ratio Profile",
            "savefig": False,
            "savename": None,
            "ylims": [0, 15],
            "xlims": [X[0], X[-1]],
            "yticks": np.arange(0, 15.1, 0.5),
            "figsize": (15, 8),
            "layout": "tight",
            "cbar_label": "Ozone ($ppb_v$)",
            "fontsize_cbar": 16,
            "grid": True,  # Add a parameter for grid
        }

        params.update(kwargs)
        ncmap, nnorm = self.O3_curtain_colors()

        fig, ax = kwargs.get("figure", (None, None))
        if not fig or ax:
            fig, ax = plt.subplots(1, 1, figsize=params["figsize"], layout=params["layout"])

        if use_countourf:
            levels = nnorm.boundaries
            im = ax.contourf(X, Y, Z, levels=levels, cmap=ncmap, norm=nnorm)

        else:
            im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest", alpha=1)

        if not fig or ax:
            self._plot_settings(fig, ax, params, im)

            if params["savename"]:
                plt.savefig(params["savename"], dpi=350)

            plt.show()

        return
    
    def _apply_time_axis(self, ax, major="Day", major_interval=7, minor="Day",
                     minor_interval=1, auto=True, date_format=None):
        """Apply time-axis formatting to an Axes."""

        locator_map = {
            "Year": mdates.YearLocator,
            "Month": mdates.MonthLocator,
            "Day": mdates.DayLocator,
            "Hour": mdates.HourLocator,
            "Minute": mdates.MinuteLocator,
            "Auto": mdates.AutoDateLocator,
        }

        if auto:
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateConverter(locator))
        else:
            ax.xaxis.set_major_locator(locator_map[major](
                interval=major_interval,
                nbins=max_major_))
            ax.xaxis.set_minor_locator(locator_map[minor](interval=minor_interval))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(date_format) if date_format
                else mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )

        ax.minorticks_on()
        ax.grid(which="both", linestyle="--", alpha=0.7)
        return ax

class GEOS_CF(utilities):
    # https://dphttpdev01.nccs.nasa.gov/data-services/cfapi/assim/chm/v72/O3/39x-77/20230808/20230811
    # https://dphttpdev01.nccs.nasa.gov/data-services/cfapi/assim/met/v72/MET/39x-77/20230808/20230811
    def __init__(self, internal=True):
        super().__init__()
        if internal is False:
            self.base_url = r"https://dphttpdev01.nccs.nasa.gov/data-services"
        else:
            self.base_url = r"https://fluid.nccs.nasa.gov/cfapi"

        self.data[("GEOS_CF", "Replay")] = {}
        self.troubleshoot["GEOS_CF"] = []
        return

    def _get_geos_data(self, lat_lon, date_start, date_end, collection="assim", molecule="O3"):
        ozone_query = (
            f"{self.base_url}/{collection}/chm/v72/{molecule}/{lat_lon}/{date_start}/{date_end}"
        )
        heights_query = (
            f"{self.base_url}/{collection}/met/v72/{molecule}/{lat_lon}/{date_start}/{date_end}"
        )

        ozone_response = requests.get(ozone_query).json()
        met_response = requests.get(heights_query).json()
        times = pd.to_datetime(ozone_response["time"], utc=True, format="%Y-%m-%dT%H:%M:%S")

        ozone = pd.DataFrame(ozone_response["values"]["O3"], index=times)
        ozone.columns = pd.to_numeric(ozone.columns)
        ozone.sort_index(axis=1, inplace=True)

        heights = pd.DataFrame(met_response["values"]["ZL"], index=times)
        heights.columns = pd.to_numeric(heights.columns)
        heights.sort_index(axis=1, inplace=True)

        times = np.tile(times, (72, 1))
        data = {"height": heights, "ozone": ozone, "time": times.T}

        return (ozone_response, met_response), (date_start, date_end), data

    def get_geos_data_multithreaded(self, lat_lon, start_date, end_date):
        def fetch_geos_data_for_date(self, lat_lon, date):
            date_str = date.strftime("%Y%m%d")
            return self._get_geos_data(lat_lon, date_str, date_str)

        date_range = pd.date_range(start=start_date, end=end_date)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(fetch_geos_data_for_date, self, lat_lon, date)
                for date in date_range
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Downloading GEOS_CF Data for {lat_lon}",
            ):
                try:
                    response, dates, data = future.result()

                    key = ("GEOS_CF", "Replay", f"{lat_lon}")
                    if key not in self.data.keys():
                        self.data[key] = {}

                    self.data[("GEOS_CF", "Replay", f"{lat_lon}")][dates[0]] = data

                except Exception as e:
                    self.troubleshoot["GEOS_CF"].append(f"{e}")

        return self
    
def get_asset(filename: str) -> Path:
    """
    Retrieve an asset file included in the same package as this module.

    Args:
        filename: Name of the file to retrieve.

    Returns:
        pathlib.Path pointing to the asset file.

    Usage:
        path = get_asset("config.yaml")
        with open(path, "r") as f:
            data = f.read()
    """
    # Automatically use the package this module is in
    package = __package__  # will resolve to current package
    if not package:
        raise RuntimeError("Cannot determine current package")

    try:
        return resources.files(package).joinpath(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Asset '{filename}' not found in package '{package}'")

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
        self.curtain_plot_theme = {
            # ime_Tick label font sizes
            "xtick.labelsize": 12,      # previously params["fontsize_ticks"]
            "ytick.labelsize": 12,

            # Axes labels
            "axes.labelsize": 14,       # previously params["fontsize_label"]

            # Title font
            "axes.titlesize": 16,       # previously params["title"]["fontsize"]
            "axes.titleweight": "bold",

            # Grid style
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.grid.which": "both",

            # Minor ticks
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,

            # Major ticks
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,

            # Figure size (optional)
            "figure.figsize": (20, 6),

            # Fonts
            "font.family": "Courier New",
            "font.size": 14
            }
        self.data = {}
        self.troubleshoot["TOLNet"] = []
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
                GEOS_CF().get_geos_data_multithreaded(
                    lat_lon, self.request_dates[0], self.request_dates[1]
                    )

        return self

    @cached_property
    def watermark(self) -> np.ndarray:
        """
        Returns the watermark image as a NumPy array, ready for plt.Figure.figimage().
        """
        from PIL import Image
        path = get_asset("assets/watermarks/TOLNet.png")
        img = Image.open(path).convert("RGBA")  # ensure alpha channel
        return np.array(img)

    def tolnet_curtain_plot(self, data: dict, **kwargs):
        with plt.rc_context(self.curtain_plot_theme):
            fig, ax = plt.subplots()
            self._apply_time_axis(ax, major="Hour", major_interval=2, minor="Minute", minor_interval=30)

            xlims = kwargs.get("xlims", "auto")
            dates = sorted(list(data.keys()))

            if xlims == "auto": 
                pass   
            elif isinstance(xlims, list): 
                xlims = pd.to_datetime(xlims, utc=True)
                xlims = [xlims.min(), xlims.max()]
                dates = pd.to_datetime(dates, utc=True)
                dates = [ str(x.strftime("%Y-%m-%d")) for x in dates[(dates >= xlims[0]) & (dates <= xlims[1])] ]

            for date in dates:
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

                ncmap, nnorm = self.O3_curtain_colors

                if kwargs.get("use_countourf", False):
                    levels = nnorm.boundaries
                    im = ax.contourf(X, Y, Z, levels=levels, cmap=ncmap, norm=nnorm)
                else:
                    im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest", alpha=1)

            cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, *np.arange(10, 121, 10), 150, 200, 300, 600])
            cbar.set_label(label=params["cbar_label"], size=16, weight="bold")

            cbar.ax.tick_params(labelsize=params["fontsize_ticks"])
            plt.title(**params["title"])

            ax.set_ylabel(params["ylabel"], fontsize=params["fontsize_label"])
            ax.set_xlabel(params["xlabel"], fontsize=params["fontsize_label"])

            xlims = params.get("xlims", None)
            if xlims:
                xlims = [np.datetime64(x) for x in xlims]
                ax.set_xlim(xlims)

            ax.set_yticks(params["yticks"])


            ax.set_ylim(params["ylims"])

            if params["savefig"]["fname"]:
                plt.savefig(**params["savefig"])

            
            with open(r"E:/Projects/atmoz/atmoz/assets/watermarks/TOLNet.png", "rb") as file:
                im = image.imread(file)

            ax_wm = fig.add_axes([0.12, 0.73, 0.3, 0.15], anchor='SW', zorder=10)
            ax_wm.imshow(im, alpha=0.7)
            ax_wm.axis('off')

            for i in np.arange(0.3, 1, 0.4):
                ax.text(i, 0.5, 'NRT DATA. NOT CITABLE.', transform=ax.transAxes,
                        fontsize=30, color='black', alpha=0.5,
                        ha='center', va='center', rotation=25
                        )

            plt.show()

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
        min_date=date_start, max_date=date_end, product_type=product_IDs, GEOS_CF=False
        )


#%%



translator = str.maketrans({c: "_" for c in string.punctuation})

#%%

# tolnet.tolnet_curtain_plot(data.data[('NASA JPL SMOL-2', 'Centrally Processed (GLASS)', '40.89x-111.89')], **params)


#%% 

import matplotlib.pyplot as plt


def __apply_water_mark(fig, filepath, **kwargs):
    params = {
        "add_axes": {
            "rect": [0.12, 0.73, 0.3, 0.15],
            "anchor": "SW",
            "zorder": 10
        }
    }
    params = merge_dicts(params, kwargs)
    with open(filepath, "rb") as file:
            im = image.imread(file)
    ax_wm = fig.add_axes(**params["add_axes"])
    ax_wm.imshow(im, alpha=0.7)
    ax_wm.axis('off')
    return fig

def __add_plot_params(ax, params: dict):
    for func_name, kwargs in params.items():
        target = getattr(ax, func_name, None) or getattr(plt, func_name, None)
        if callable(target):
            if isinstance(kwargs, dict):
                target(**kwargs)
            elif isinstance(kwargs, (list, tuple)):
                target(*kwargs)
            else:
                target(kwargs)  # single scalar
    return 

def __apply_near_real_time(ax, **kwargs):
    params = {
        "s": 'NRT DATA. NOT CITABLE.',
        "fontsize": 30,
        "color": "black",
        "ha": "center",
        "va": "center",
        "rotation": 25,
        "transform": ax.transAxes,
        "alpha": 0.5 
        }
    params = merge_dicts(params, kwargs)
    
    ax.text(0.3, 0.5, , transform=ax.transAxes,
            fontsize=30, color='black', alpha=0.5,
            ha='center', va='center', rotation=25
            )
    
    ax.text(0.7, 0.5, 'NRT DATA. NOT CITABLE.', transform=ax.transAxes,
        fontsize=30, color='black', alpha=0.5,
        ha='center', va='center', rotation=25
        )
    
    return 

def tolnet_curtain_plot(data: dict, **kwargs):
    default = {
        "set_ylabel": {
            "ylabel": "Altitude (km ASL)"
            },

        "set_xlabel": {
            "xlabel": "Time"
            },

        "set_title": {
            "label": r"Ozone Mixing Ratio Profile",
            "fontsize": 16
            },

        "grid": {
            "visible": True,
            "color": "gray",
            "linestyle": "--",
            "linewidth": 0.5
            },

        "cbar.ax.tick_params": {
            "labelsize": 16
            },

        "set_ylims": [0, 15],
        "savefig": {
            "fname": "test.png",
            "dpi": 300,
            "transparent": True,
            "format": "png",
            "bbox_inches": "tight"
            },

        "layout": "tight",
        "cbar.set_label": {
            "label": "Ozone ($ppb_v$)", 
            "size": 16, 
            "weight": "bold"
            }
        }

    params = merge_dicts(default, kwargs)
    with plt.rc_context(tolnet.curtain_plot_theme):
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

            ncmap, nnorm = utilities().O3_curtain_colors

            if params.get("use_countourf", False):
                levels = nnorm.boundaries
                im = ax.contourf(X, Y, Z, levels=levels, cmap=ncmap, norm=nnorm)
            else:
                im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest", alpha=1)

        cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, *np.arange(10, 121, 10), 150, 200, 300, 600])


        
        __apply_water_mark(fig, r"E:/Projects/atmoz/atmoz/assets/watermarks/Watermark_TOLNet.png")
        __add_plot_params(ax, params)
        

        plt.show()


params = {
    "set_ylabel": {
        "ylabel": "Altitude (km ASL)"
        },

    "set_yticks": {
        "ticks": np.arange(0, 16, 1)
        },

    "set_xlabel": {
        "xlabel": "Time"
        },

    "set_title": {
        "label": r"Ozone Mixing Ratio Profile",
        "fontsize": 16
        },

    "grid": {
        "visible": True,
        "color": "gray",
        "linestyle": "--",
        "linewidth": 0.5
        },

    "cbar.ax.tick_params": {
        "labelsize": 12
        },

    "set_ylims": [0, 15],

    "savefig": {
        "fname": "test.png",
        "dpi": 300,
        "transparent": True,
        "format": "png",
        "bbox_inches": "tight"
        },

    "layout": "tight",

    "cbar.set_label": {
        "label": "Ozone ($ppb_v$)", 
        "size": 16, 
        "weight": "bold"
        },

    "xaxis.set_major_locator": {
        "locator": mdates.AutoDateLocator()
        },

    "xaxis.set_major_formatter": {
        "formatter": mdates.ConciseDateFormatter(
            mdates.AutoDateLocator()
            )
        }
    }

tolnet_curtain_plot(data.data[('NASA JPL SMOL-2', 'Centrally Processed (GLASS)', '40.89x-111.89')], **params)
# %%
