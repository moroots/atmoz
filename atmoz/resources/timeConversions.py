# -*- coding: utf-8 -*-
"""
Created on 2025-10-25 10:11:44

@author: Maurice Roots

Description:
     - Describing Time stuff
"""

import re
import pandas as pd
import numpy as np

epochs = {
    "MJD2K": ( "days", pd.Timestamp("2000-01-01T00:00:00Z") ),
    "MJD": ( "days", pd.Timestamp("1858-11-17T00:00:00Z") ) ,
    "UNIX": ( "seconds", pd.Timestamp("1970-01-01T00:00:00Z") ),
    "MJD2000": ( "days", pd.Timestamp("2000-01-01T00:00:00Z") ),
    "JD": ( "days", pd.Timestamp("4713-01-01T12:00:00Z") ),
}

def parse_time_units(units_str):
    """
    Parse a CF-style time units string like 'seconds since 2023-06-24T00:00:00Z'
    and return the unit and reference time.
    """
    match = re.match(r"(\w+)\s+since\s+([\d\-T:Z]+)", units_str)
    if not match:
        return None, None
    unit = match.group(1)
    ref_time = pd.to_datetime(match.group(2), utc=True)
    return unit, ref_time

def h5Dataset_timestamp(dataset_time, units_key=None):
    datetime = dataset_time[()]

    if units_key is not None:
        units_key = list(units_key)
    else: 
        units_key = ["units", "var_units", "unit", "var_unit"]

    units = None

    for key in dataset_time.attrs.keys():
        if key.lower() in units_key:
            units = dataset_time.attrs[key]
            break

    if units is None:
        raise ValueError("No 'units' attribute found in dataset_time. Assign argument 'units_key' in function call.")

    if isinstance(units, (bytes, np.bytes_)):
        units = units.decode()

    units = units.lower()
    
    ref_time = None
    for key in epochs.keys():
        if key.lower() == units:
            unit_str, ref_time = epochs[key]
            break
    
    if ref_time is None:
        unit_str, ref_time = parse_time_units(units)
        if ref_time is None:
            raise ValueError(f"Could not parse reference time from units: {units}. Sorry, you'll have to do this one manually.")

    # Convert numeric seconds to timedeltas and add to reference
    timestamps = ref_time + pd.to_timedelta(datetime, unit=unit_str)

    return timestamps