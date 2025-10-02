# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:34:20 2025

@author: Magnolia
"""

#%% 
from atmoz.resources import useful_functions
from importlib import resources
from pathlib import Path
from typing import Optional

#%% 

def get_asset(filename: str, package: str = "atmoz") -> Optional[Path]:
    """
    Recursively search for a file within the given package and return its path.

    Args:
        filename: Name of the file to find.
        package: Root package to search (default: 'atmoz').

    Returns:
        pathlib.Path to the file if found, else None.
    """
    try:
        # Recursively walk through all files in the package
        for asset in resources.files(package).rglob("*"):
            if asset.name == filename and asset.is_file():
                try:
                    return asset.locate()  # Python 3.12+
                except AttributeError:
                    with resources.as_file(asset) as file_path:
                        return file_path
        return None
    except Exception as e:
        print(f"Error searching for asset: {e}")
        return None

#%% 
def merge_dicts(default, override):
    """Recursively merge two dictionaries. Override values replace defaults."""
    result = default.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            # merge nested dicts
            result[k] = useful_functions.merge_dicts(result.get(k, {}), v)
        else:
            # override or add new key
            result[k] = v
    return result