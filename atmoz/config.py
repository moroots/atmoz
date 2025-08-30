# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 00:10:13 2025

@author: Magnolia
"""

from pathlib import Path
from configparser import ConfigParser
import importlib

def get_package_path(**kwargs):
    dirpath = kwargs.get("package_path", None)
    if dirpath is None:
        spec = importlib.util.find_spec("atmoz")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot find package 'atmoz'")
        return Path(spec.origin).parent
    return dirpath

def get_config(**kwargs):
    config = ConfigParser()

    config_path = kwargs.get("config_path", None)

    if config_path is None:
        config_path = (get_package_path() / "config.ini")
        with importlib.resources.open_text("atmoz", "config.ini") as f:
            config.read_file(f)
            return config, config_path

    with open(config_path, "r") as f:
        config.read_file(f)
        return config, config_path

def write_config(section: str, variable: str, value, **kwargs):
    config, config_file = get_config(**kwargs)

    if not config.has_section(section):
        config.add_section(section)
    config.set(section, variable, str(value))
    with open(config_file, "w") as f:
        config.write(f)
    return

def read_config(section, variable, **kwargs):
    config, config_file = get_config(**kwargs)

    if config.has_section(section) and config.has_option(section, variable):
        return config.get(section, variable)
    else:
        print(f"[{section}] => {variable}: Not found in {config_file.name}")
        print(r"Please set this value using:"
              f"atmoz.config.write_config('{section}', '{variable}', 'your-value')"
              )
        return None
