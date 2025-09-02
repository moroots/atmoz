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
        raise FileNotFoundError(f"Asset '{filename}' not found in package '{package}'")