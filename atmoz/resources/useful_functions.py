# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:34:20 2025

@author: Magnolia
"""
import importlib.resources as resources
from pathlib import Path
from typing import Union

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
        return None
    except Exception as e: 
        raise e

def merge_dicts(default, override):
    """Recursively merge two dictionaries. Override values replace defaults."""
    result = default.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            # merge nested dicts
            result[k] = merge_dicts(result.get(k, {}), v)
        else:
            # override or add new key
            result[k] = v
    return result