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
from pint import UnitRegistry

ureg = UnitRegistry()
try:
    ureg.define('ppbv = 1e-9 = parts_per_billion_by_volume')
except Exception:
    pass
try:
    ureg.define('ppmv = 1e-6 = parts_per_million_by_volume')
except Exception:
    pass

@dataclass
class dataframe:
    data: np.ndarray  # e.g., pd.DataFrame or pd.Series
    units: str

    def __post_init__(self):
        # Convert masked arrays to regular arrays with NaN
        if isinstance(self.data, np.ma.MaskedArray):
            self.data = self.data.filled(np.nan)
        # Convert Pint Quantities to plain numpy arrays (drop units)
        if hasattr(self.data, "magnitude") and hasattr(self.data, "units"):
            self.data = np.asarray(self.data.magnitude)
        # If not already a DataFrame or Series, convert to DataFrame
        if not isinstance(self.data, (pd.DataFrame, pd.Series)):
            self.data = pd.DataFrame(self.data)
    
    def __repr__(self):
        return f"atmoz.dataframe(units='{self.units}', shape={self.data.shape})\n{self.data.__repr__()}"
    def __getattr__(self, name):
        # Only called if attribute not found on self; delegate to data
        if name == "units":
            # Prevent recursion for units
            return self.__dict__["units"]
        return getattr(self.data, name)
    
    def _get_compatible(self, other):
        """Return other's data converted to self's units if possible."""
        if not isinstance(other, dataframe):
            # Handle Pint Quantity or similar
            if hasattr(other, "magnitude") and hasattr(other, "units"):
                # Convert to self.units if possible
                try:
                    factor = (1 * ureg(str(other.units))).to(self.units).magnitude
                    return other.magnitude * factor
                except Exception:
                    # If not convertible, just use magnitude (will combine units in op)
                    return other.magnitude
            return other
        if self.units == other.units:
            return other.data
        # Try to convert other's units to self.units
        try:
            factor = (1 * ureg(other.units)).to(self.units).magnitude
            return other.data * factor
        except Exception as e:
            raise ValueError(f"Cannot convert units '{other.units}' to '{self.units}': {e}")

    # Math operations
    def __add__(self, other):
        if isinstance(other, dataframe) or (hasattr(other, "magnitude") and hasattr(other, "units")):
            data_other = self._get_compatible(other)
            return dataframe(self.data + data_other, units=self.units)
        else:
            return dataframe(self.data + other, units=self.units)

    def __sub__(self, other):
        if isinstance(other, dataframe) or (hasattr(other, "magnitude") and hasattr(other, "units")):
            data_other = self._get_compatible(other)
            return dataframe(self.data - data_other, units=self.units)
        else:
            return dataframe(self.data - other, units=self.units)

    def __mul__(self, other):
        if isinstance(other, dataframe):
            # Combine units for multiplication
            new_units = f"({self.units})*({other.units})"
            return dataframe(self.data * other.data, units=new_units)
        elif hasattr(other, "magnitude") and hasattr(other, "units"):
            new_units = f"({self.units})*({other.units})"
            return dataframe(self.data * other.magnitude, units=new_units)
        else:
            return dataframe(self.data * other, units=self.units)

    def __truediv__(self, other):
        if isinstance(other, dataframe):
            # Combine units for division
            new_units = f"({self.units})/({other.units})"
            return dataframe(self.data / other.data, units=new_units)
        elif hasattr(other, "magnitude") and hasattr(other, "units"):
            new_units = f"({self.units})/({other.units})"
            return dataframe(self.data / other.magnitude, units=new_units)
        else:
            return dataframe(self.data / other, units=self.units)
        

    def convert_units(self, target_unit: str, ureg_in=None) -> 'dataframe':
        """
        Convert the units of the dataframe using Pint.
        """
        _ureg = ureg_in or ureg
        if not self.units:
            raise ValueError("Current units are not set; cannot convert.")
        factor = (1 * _ureg(self.units)).to(target_unit).magnitude
        converted_data = self.data * factor
        return dataframe(converted_data, units=target_unit)
    

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
        for var, varinfo in self.data.items():
            # Convert masked arrays to regular arrays before DataFrame
            arr = varinfo["data"]
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(np.nan)
            # Convert Pint Quantities to plain numpy arrays
            if hasattr(arr, "magnitude") and hasattr(arr, "units"):
                arr = np.asarray(arr.magnitude)
            if len(arr) != len(self.time):
                raise ValueError(f"data for variable '{var}' must have same length as time")
            test = dataframe(
                data=pd.DataFrame(arr, index=self.time),
                units=varinfo.get("units", "")
            )
            setattr(self, var, test)
        self.variables = list(self.data.keys())
        if self.variables: 
            self.time = getattr(self, self.variables[0]).data.index

    def copy(self):
        return copy.deepcopy(self)

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

    def __getattr__(self, name):
        """
        Delegate attribute access to each variable's data (if possible).
        For example, lidar_profiles.head() will return {var: data.head()} for each var.
        """
        # Only called if attribute not found on self
        def apply_to_all(*args, **kwargs):
            results = {}
            for var in self.variables:
                data = self.data[var]["data"]
                print(data)
                # If data is a DataFrame or Series, apply the method
                if hasattr(data, name):
                    results[var] = getattr(data, name)(*args, **kwargs)
                else:
                    # For numpy arrays, support some common methods
                    if hasattr(np, name):
                        results[var] = getattr(np, name)(data, *args, **kwargs)
                    else:
                        raise AttributeError(f"'{type(data)}' object has no attribute '{name}'")
            return results
        # Only delegate if not a dataclass field or method
        if name in self.__dict__ or name in self.__class__.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return apply_to_all


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
filepath = r"c:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\data\TROPOZ\lidar\groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210519t000000z_20210520t000000z_001.h5"

with h5py.File(filepath, "r") as f:
    everything = hdf5_to_dict(f)

def bytes_to_str(val):
    if isinstance(val, bytes):
        return val.decode()
    return str(val)

data_vars = {
    "ozone": {
        "data": everything["O3.MIXING.RATIO.VOLUME_DERIVED"]["_data"].astype(np.float32), 
        "units": bytes_to_str(everything["O3.MIXING.RATIO.VOLUME_DERIVED"]["_attrs"]['VAR_UNITS'])
    },
    "uncertainty": {
        "data": everything["O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.COMBINED.STANDARD"]["_data"].astype(np.float32),
        "units": bytes_to_str(everything["O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.COMBINED.STANDARD"]["_attrs"]['VAR_UNITS'])
        },
    "ozone_number_density": {
        "data": everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL"]["_data"].astype(np.float32),
        "units": bytes_to_str(everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL"]["_attrs"]['VAR_UNITS'])
    },
    "ozone_number_density_uncertainty": {
        "data": everything["O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL_UNCERTAINTY.COMBINED.STANDARD"]["_data"].astype(np.float32),
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


#%% 


ureg = UnitRegistry()

# Define ppbv and ppmv if not already present
ureg.define('ppbv = 1e-9 = parts_per_billion_by_volume')
ureg.define('ppmv = 1e-6 = parts_per_million_by_volume')

# Define a value in ppbv
val = 1500 * ureg('ppbv')

# Convert to ppmv
val_in_ppmv = val.to('ppmv')
print(val_in_ppmv)  # 1.5 ppmv


#%%

