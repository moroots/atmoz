# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:02:59 2025

@author: Maurice Roots

A module for working with NASA's EarthAccess

"""
#%%

import os
import keyring
import keyring.errors
import pandas as pd
from pathlib import Path

from atmoz.resources.useful_functions import merge_dicts

try:
    import earthaccess
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install 'earthaccess': "
        "https://earthaccess.readthedocs.io/en/stable/quick-start/"
    )


class EarthData:
    _SERVICE = "atmoz_earthdata"

    def __init__(self):
        self._login()

    @staticmethod
    def store_credentials(username: str, password: str) -> None:
        """Store (or update) EarthData credentials in the system keyring."""
        keyring.set_password(EarthData._SERVICE, "username", username)
        keyring.set_password(EarthData._SERVICE, username, password)

    @staticmethod
    def remove_credentials() -> None:
        """Remove stored EarthData credentials from the system keyring."""
        username = keyring.get_password(EarthData._SERVICE, "username")
        for key in filter(None, ["username", username]):
            try:
                keyring.delete_password(EarthData._SERVICE, key)
            except keyring.errors.PasswordDeleteError:
                pass

    def _login(self) -> None:
        username = keyring.get_password(self._SERVICE, "username")
        if not username:
            raise PermissionError(
                f"{self.__class__.__name__}: No credentials found. "
                "Call EarthData.store_credentials(username, password) first."
            )
        password = keyring.get_password(self._SERVICE, username)
        if not password:
            raise PermissionError(
                f"{self.__class__.__name__}: Incomplete credentials in keyring. "
                "Call EarthData.store_credentials(username, password) again."
            )

        _SENTINEL = object()
        saved = {k: os.environ.get(k, _SENTINEL) for k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")}
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        try:
            auth = earthaccess.login(strategy="environment")
        finally:
            for k, v in saved.items():
                if v is _SENTINEL:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        if not auth.authenticated:
            raise PermissionError(
                f"{self.__class__.__name__}: Authentication failed. "
                "Update credentials with EarthData.store_credentials(username, password)."
            )
        self.auth = auth

    def get_short_names(self, keyword: str = "TEMPO_*", count: int = -1) -> pd.DataFrame:
        response = earthaccess.search_datasets(keyword=keyword, count=count)
        umm = [r["umm"] for r in response]
        return pd.DataFrame({
            "ShortName": [r["ShortName"] for r in umm],
            "EntryTitle": [r["EntryTitle"] for r in umm],
            "Abstract": [r["Abstract"] for r in umm],
        })

    def search_data(self, **params) -> list:
        params = merge_dicts({"count": -1}, params)
        return earthaccess.search_data(**params)

    def download_data(self, local_path: Path = None, **params) -> list:
        if local_path is None:
            local_path = Path(__file__).parent / "data" / "nasa_earth_data"
        granules = self.search_data(**params)
        if not granules:
            return []
        return earthaccess.download(granules, local_path)


#%%

if __name__ == "__main__":
    # One-time setup (only needed once per machine):
    # EarthData.store_credentials("my_username", "my_password")

    params = {
        "short_name": "TEMPO_NO2_L3",
        "temporal": ("2025-06-01", "2025-08-01"),
        "count": 10,
    }

    ed = EarthData()
    short_names = ed.get_short_names()
    granules = ed.search_data(**params)
    files = ed.download_data(**params)
