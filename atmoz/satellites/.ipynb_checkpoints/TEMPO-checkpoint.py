# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:18:54 2024

@author: Maurice Roots

First Level implementation of TEMPO API call and Plotting Data
"""

# import earthaccess
import netCDF4 as nc
import os
import platform
from subprocess import Popen
import shutil
import numpy as np
# from shapely.geometry import Point, Polygon
from scipy.interpolate import griddata
import sys

import json

import pandas as pd
from pathlib import Path

from atmoz.data_access import NASA
#%%


class TEMPO(NASA.EarthData):

    def __init__(self):
        super().__init__()  # This initializes the _TEMPO_Download class
        return

    def products(self):
        self._get_short_names(keyword="TEMPO_*")
        print(self.descriptions[["ShortName", "EntryTitle"]].to_string())
        return

    def download_data(self, dir_path: str="./results", **kwargs):
        # Default arguments
        args = {"short_names": ["TEMPO_NO2_L2"],
                "count": 10,
                "temporal": ((pd.Timestamp("now") - pd.Timedelta(1, unit="d")).strftime("%Y%m%d"),
                             pd.Timestamp("now").strftime("%Y%m%d")
                             ),
                }

        args.update(kwargs)
        # Ensure short_names is a list
        short_names = args.pop("short_names") if isinstance(args["short_names"], list) else [args.pop("short_names")]

        # Perform search for each short_name
        for short_name in short_names:
            self._search_data({**args, "short_name": short_name})
            print(f"\n Found {len(self._results)} Files in {args['temporal']} for {short_name}")
            self._download_data(dir_path)

        return


# tempo = TEMPO()


# params = {"short_name": "TEMPO", "temporal": ("2024-08-01", "2024-08-02"), "count":100}

# tempo_search = NASA.EarthData()._search_data(params)

# params = {"short_names"; }
# tempo.download_data(short_names=["TEMPO_NO2_L3", "TE5MPO_HCHO_L3"], temporal=("2024-08-01", "2024-08-01"), count=100, dir_path="./TEMPO_Data_2024")


#%%












