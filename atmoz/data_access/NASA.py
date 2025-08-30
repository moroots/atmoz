# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:02:59 2025

@author: Maurice Roots

A module for working with NASA's EarthAcess

"""

# import earthaccess
import netCDF4 as nc
import os
import platform
from subprocess import Popen
import shutil
import numpy as np

from scipy.interpolate import griddata
import sys

import json

import pandas as pd
from pathlib import Path

from custom_utilities import messages
from custom_utilities.decorators import report, logs

import keyring

try:
    import earthaccess
except ModuleNotFoundError:
    print("Please install 'earthaccess' \n"
          "Use this link for instruction: "
          "https://earthaccess.readthedocs.io/en/stable/quick-start/"
          )

class EarthData:
    #This class is largely unfinished
    def __init__(self):
        earthaccess.login(persist=True)
        return

    def get_short_names(self, keyword: str = "TEMPO_*", count: int = -1):
        response = earthaccess.search_datasets(
            keyword=keyword, count=count
            )

        responses = [r["umm"] for r in response]
        short_names = [r["ShortName"] for r in responses]
        entry_titles = [r["EntryTitle"] for r in responses]
        abstracts = [r["Abstract"] for r in responses]

        results = pd.DataFrame(
            {
                "ShortName": short_names,
                "EntryTitle": entry_titles,
                "Abstract": abstracts,
                "Response": responses
            }
            )

        return results

    def search_data(self, **params):
        return earthaccess.search_data(**params)

    def download_data(self, search_results, dir_path: str ="./results"):
        #This function call is weird and slow. tqdm is semi broken.
        return earthaccess.download(search_results, dir_path)


#%%

if __name__ == "__main__":
    params = {"short_name": "TEMPO_NO2_L3", "temporal": ("2025-06-01", "2025-08-01"), "count":100}

    tempo_search = EarthData().get_short_names()
    tempo_data_search = EarthData().search_data(**params)
    tempo_data = EarthData().download_data(tempo_data_search, dir_path=r"E:\Projects\atmoz\results")
