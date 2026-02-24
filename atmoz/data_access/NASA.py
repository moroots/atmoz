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

from atmoz.resources.useful_functions import merge_dicts

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

    def _write_netrc(self, earth_data_login: EarthDataLogin):
        for path in self.netrc_paths: 
            with open(path, "w") as f:
                f.write(f"machine urs.earthdata.nasa.gov login {earth_data_login.username} password {earth_data_login.password.get_secret_value()}\n")
    
    def _login(self, earth_data_login: EarthDataLogin = None):
        netrc_path = next((p for p in self.netrc_paths if p.exists()), None)
        if netrc_path and netrc_path.exists():
            pass
        else: 
            if earth_data_login:
                self._write_netrc(earth_data_login)
            else: 
                raise FileNotFoundError(f"{self.__class__.__name__}: No .netrc file found and no credentials (username, password) provided. \n Please provide credentials (username, password) or create a .netrc file.")
            
        temp = earthaccess.login(strategy="netrc", persist=True)

        if not temp.authenticated:
            raise PermissionError(f"{self.__class__.__name__}: Authentication failed. Please check your credentials.")
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
        default = {"count": -1}
        params = merge_dicts(default, params)
        return earthaccess.search_data(**params)

    def download_data(self, **params):
        default = {
            "local_path": Path(__file__).parent / "data" / "nasa_earth_data",
            }
        params = merge_dicts(default, params)
        return earthaccess.download(**params)


#%%

if __name__ == "__main__":
    params = {"short_name": "TEMPO_NO2_L3", "temporal": ("2025-06-01", "2025-08-01"), "count":100}

    tempo_search = EarthData().get_short_names()
    tempo_data_search = EarthData().search_data(**params)
    tempo_data = EarthData().download_data(**params)
