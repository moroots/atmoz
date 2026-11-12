# -*- coding: utf-8 -*-
"""
Created on 2025-09-04

@author: Maurice Roots

Description
    - GEOS-CF data retrieval and processing class.   
"""

# necessary imports
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# internal imports
from atmoz.resources import debug

class GEOS_CF(debug.utilities):
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