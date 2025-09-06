# -*- coding: utf-8 -*-
"""
Created on 2025-09-05 22:28:42

@author: Maurice Roots

Description:
     - Some useful dataclasses
"""
#%% 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from atmoz.resources import useful_functions
from atmoz.resources.useful_functions import get_asset
from atmoz.resources import debug
from atmoz.resources import plot_utilities
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mimg
import copy
from types import SimpleNamespace
from collections import namedtuple

#%% 

@dataclass
class LidarProfiles:
    """
    Dataclass to store one or many lidar profiles with multiple variables.
    Each variable is dynamically exposed as an attribute with .data and .units.
    """
    time: Union[List[datetime], np.ndarray]
    altitude: Union[List[np.ndarray], np.ndarray]
    data: Dict[str, Dict[str, Any]]  # e.g., {"ozone": {"data": [...], "units": "ppbv"}, ...}
    latitude: Dict[str, Any]
    longitude: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        n_profiles = len(self.time)
        if isinstance(self.altitude, list):
            if len(self.altitude) != n_profiles:
                raise ValueError("altitude must have same length as time")
        for var, varinfo in self.data.items():
            if len(varinfo["data"]) != n_profiles:
                raise ValueError(f"data for variable '{var}' must have same length as time")
            # Dynamically set each variable as an attribute with .data and .units
            setattr(self, var, SimpleNamespace(data=varinfo["data"], units=varinfo.get("units", "")))
        self.variables = list(self.data.keys())
    
    # @cached_property
    # def resolution(self) -> List[float]:
    #     """
    #     Calculate the smallest vertical (m) and temporal (s) resolution for each profile.
    #     """
    #     r = namedtuple("vertical", "temporal")
    #     resolutions = r(np.nan, np.nan)
        
        
        
    #     for if len(alt) > 2:
    #         res = np.nanmin(np.diff(time))
    #         resolutions._replace(temporal=res)
    #     return resolutions

    def copy(self):
        return copy.deepcopy(self)

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Return a dictionary of DataFrames, one for each data variable.
        Each DataFrame has a MultiIndex (time, latitude, longitude, instrument, site),
        columns are altitude, and values are the data for that variable.
        """
        dfs = {}
        n_profiles = len(self.time)
        for var in self.variables:
            records = []
            index = []
            for i in range(n_profiles):
                # Each profile: 1D array of altitudes and data
                alt = self.altitude[i]
                data = getattr(self, var).data[i]
                # Build a row: index is (time, lat, lon, instrument, site), columns are altitude
                index.append((self.time[i], (self.longitude, self.latitude)))
                records.append(pd.Series(data, index=alt))
            # Build DataFrame for this variable
            df = pd.DataFrame(records, index=pd.MultiIndex.from_tuples(
                index, names=["time", "latitude", "longitude"]
            ))
            df.columns.name = "altitude"
            dfs[var] = df
        return dfs

    def __repr__(self):
        def get_shape(x):
            try:
                return np.shape(x)
            except Exception:
                return f"len={len(x)}"
        data_lines = "\n".join([f"    - {k}" for k in self.data.keys()])
        s = (
            f"atmoz.LidarProfiles\n"
            f"  time: {get_shape(self.time)}\n"
            f"  altitude: {get_shape(self.altitude)}\n"
            f"  latitude: {get_shape(self.latitude['data']) if isinstance(self.latitude, dict) else get_shape(self.latitude)}\n"
            f"  longitude: {get_shape(self.longitude['data']) if isinstance(self.longitude, dict) else get_shape(self.longitude)}\n"
            f"  data: \n{data_lines}\n"
        )
        return s
    
    def plot(self, plot_type: str, ax: Optional[plt.Axes] = None, **kwargs):
        pass

#%%

def hdf5_to_dict(h5obj):
    """
    Recursively extract all groups, datasets, and attributes from an h5py File or Group
    into a nested dictionary, preserving the structure.
    """
    out = {}
    # Add attributes
    if hasattr(h5obj, "attrs"):
        out["_attrs"] = {k: v for k, v in h5obj.attrs.items()}
    # Add datasets and groups
    for key, item in h5obj.items():
        if "fake" in key.lower():
            continue
        if isinstance(item, h5py.Group):
            out[key] = hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            out[key] = {
                "_data": item[()],
                "_attrs": {k: v for k, v in item.attrs.items()}
            }
    return out



#%% 
# Usage example:
import h5py

filepath = r"C:\Users\Magnolia\OneDrive - UMBC\Research\Analysis\May2021\data\TROPOZ\lidar\groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210519t000000z_20210520t000000z_001.h5"

with h5py.File(filepath, "r") as f:
    everything = hdf5_to_dict(f)

#%% 

def bytes_to_str(val):
    if isinstance(val, bytes):
        return val.decode()
    return str(val)

data_vars = {
    "ozone": {
        "data": everything["O3.MIXING.RATIO.VOLUME_DERIVED"]["_data"], 
        "units": bytes_to_str(everything["O3.MIXING.RATIO.VOLUME_DERIVED"]["_attrs"]['VAR_UNITS'])
    },
    "uncertainty": {
        "data": everything["O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.COMBINED.STANDARD"]["_data"],
        "units": bytes_to_str(everything["O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.COMBINED.STANDARD"]["_attrs"]['VAR_UNITS'])
        },
    "ozone_number_density": {
        "data": everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL"]["_data"],
        "units": bytes_to_str(everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL"]["_attrs"]['VAR_UNITS'])
    },
    "ozone_number_density_uncertainty": {
        "data": everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL_UNCERTAINTY.COMBINED.STANDARD"]["_data"],
        "units": bytes_to_str(everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL_UNCERTAINTY.COMBINED.STANDARD"]["_attrs"]['VAR_UNITS'])
    },
}

latitude = everything["LATITUDE.INSTRUMENT"]["_data"]
longitude = everything["LONGITUDE.INSTRUMENT"]["_data"]
times = everything["DATETIME"]["_data"]
altitudes = everything["ALTITUDE"]["_data"]

# Create LidarProfiles instance
lidar_profiles = LidarProfiles(
    time=times,
    altitude=altitudes,
    data=data_vars,  # <-- change here
    latitude=latitude,
    longitude=longitude,
)

#%% 

from pympler import asizeof
print(f"Total memory used by 'everything': {asizeof.asizeof(everything)/1024/1024:.2f} MB")

#%%

import pickle
import gzip

# Save with high compression
with gzip.open("everything.pkl.gz", "wb") as f:
    pickle.dump(everything, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load
with gzip.open("everything.pkl.gz", "rb") as f:
    everything_loaded = pickle.load(f)