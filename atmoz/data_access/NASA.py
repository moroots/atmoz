# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:02:59 2025

@author: Maurice Roots

A module for working with NASA's EarthAccess

"""
#%%

# import earthaccess
import os
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from subprocess import Popen
import pandas as pd
from pathlib import Path

try:
    import earthaccess
except ModuleNotFoundError:
    print("Please install 'earthaccess' \n"
          "Use this link for instruction: "
          "https://earthaccess.readthedocs.io/en/stable/quick-start/"
          )
    raise ModuleNotFoundError("Please install 'earthaccess' to use this module.")

class EarthDataLogin(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(description="EarthData username", default=None)
    password: SecretStr = Field(description="EarthData password", default=None)

class EarthData:
    #This class is largely unfinished
    def __init__(self, username: str = None, password: str = None):
        self.netrc_paths = [Path.home() / ".netrc", Path.home() / "_netrc"]
        self._login(
            EarthDataLogin(username=username, password=SecretStr(password))
            if username and password
            else None
            )
        return 

    def _set_credentials(self, earth_data_login: EarthDataLogin):
        os.environ["EARTHDATA_USERNAME"] = earth_data_login.username
        os.environ["EARTHDATA_PASSWORD"] = earth_data_login.password.get_secret_value()
    
    def _login(self, earth_data_login: EarthDataLogin = None):
        netrc_path = next((p for p in self.netrc_paths if p.exists()), None)
        if netrc_path and netrc_path.exists():
            temp = earthaccess.login(strategy="netrc", persist=True)
        else: 
            if earth_data_login:
                self._set_credentials(earth_data_login)
                temp = earthaccess.login(strategy="environment", persist=True)
            else: 
                raise FileNotFoundError("No .netrc file found and no credentials (username, password) provided. \n Please provide credentials (username, password) or create a .netrc file.")
            
        if not temp.authenticated:
            raise PermissionError("Authentication failed. Please check your credentials.")
        self.auth = temp

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
