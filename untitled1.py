# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 19:47:30 2025

@author: Magnolia
"""
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from uuid import uuid4
import pandas as pd
import random

import time
from functools import wraps

import time
import psutil
import os
from functools import wraps

from custom_utilities.decorators import report, logs
from custom_utilities import messages

from pathlib import Path
import subprocess
import sys

#%%%

@logs
@report
def merge_catalog(**kwargs):
    catalog_file = kwargs.get("catalog_file", "None"); print(catalog_file)
    staging_dir = kwargs.get("staging_dir", "None"); print(staging_dir)

    staging_dir = Path(kwargs.get("staging_dir", "None")).expanduser()
    catalog_file = Path(kwargs.get("catalog_file", "None")).expanduser()

    if catalog_file.is_file():
        messages.info(f"Found {catalog_file}", **kwargs)
        master_df = pd.read_parquet(catalog_file)
        messages.trek(f"Imported {catalog_file} as {type(master_df)}", **kwargs)
    else:
        messages.info(f"Did Not Find {catalog_file}.", **kwargs)
        master_df = pd.DataFrame()
        messages.trek(f"Created catalog_file placeholder as {type(master_df)}", **kwargs)

    new_files = [x for x in staging_dir.glob("*.parquet")]

    if not new_files:
        messages.info(f"No New Files in {staging_dir}", **kwargs)
        return

    new_dfs = [pd.read_parquet(f) for f in new_files]
    messages.trek(
        (
            f"Read all parquet files in {staging_dir} "
            "as {type(new_dfs[0])} into {type(new_dfs)}"
        ),
        **kwargs
        )

    new_data = pd.concat(new_dfs, ignore_index=True)
    messages.trek(
        (
            f"Concatenated all {new_files[0].suffix} "
            "files in {staging_dir} into {type(new_data)}"
        ),
        **kwargs
        )

    updated_df = pd.concat([master_df, new_data], ignore_index=True)
    messages.trek(
        (
            f"Read all parquet files in {staging_dir} "
            "as {type(new_dfs[0])} into {type(new_dfs)}"
        ),
        **kwargs
        )

    updated_df.to_parquet(catalog_file, index=False)
    messages.trek(
        (
            f"Read all parquet files in {staging_dir} "
            "as {new_dfs[0]} into {type(new_dfs)}"
        ),
        **kwargs
        )

    for f in new_files:
        messages.trek(f"Removing {f}", **kwargs)
        f.unlink(missing_ok=True)

    messages.info("Staging files removed", **kwargs)

    messages.info(f"Merged {len(new_files)} files into '{catalog_file}'", **kwargs)

    return

@logs
@report
def _write(df, staging_dir, **kwargs):

    staging_dir = Path(staging_dir).expanduser()

    # Extract year, month, day from start_time for partitioning
    df = df.copy().explode("variables").reset_index(drop=True)

    catalog_file = Path(kwargs.get("catalog_file", "None"))

    if not catalog_file.is_file():
        print("No catalog_file yet")
        df.to_parquet(catalog_file)

    if not staging_dir.is_dir():
        result = input(f"{staging_dir} is not a directory! \n"
                       "Would you like to create it? [y, n] => ")

        if result.lower() == "y":
            staging_dir.mkdir(parents=True, exist_ok=True)
        else:
            messages.trek(f"User chose not to create {staging_dir}: Exiting...")
            raise NotADirectoryError("No valid staging directroy offered")

    else:
        df.to_parquet(staging_dir / f"{uuid4()}.parquet")

    if kwargs.get("merge_catalog", True):
        params = {"staging_dir": staging_dir}; params.update(kwargs)
        merge_catalog(**params)

    return

dates = pd.date_range(start="2000-01-01", end="2025-01-01", freq="1d")
for k in range(0, len(dates)):
    num_files = 5
    rows = 1000
    num_heights = 1000
    sensor_ids = [random.randint(1, 1000) for _ in range(num_files)]
    variables = ["temperature", "humidity", "wind", "ozone", "nitrogen-dioxide", "height", "backscatter"]

    # Files List Demo
    timestamps = pd.date_range(dates[k], periods=num_files+1, freq="1d")
    sensor_id = [str(random.sample(sensor_ids,k=1)[0]) for k in range(num_files)]
    start_time = [ timestamps[i] for i in range(0, len(timestamps)-1) ]
    end_time = [ timestamps[i] for i in range(1, len(timestamps)) ]
    filenames = [str(uuid4())]*num_files

    var = []
    for i in range(num_files):
        num_vars = random.randint(1, len(variables))
        var_list = random.sample(variables, k=num_vars)
        var.append(var_list)

    df = pd.DataFrame({
        "sensor_id": sensor_id,
        "start_time": start_time,
        "end_time": end_time,
        "variables": var,
        "filename": filenames
        })

    print(dates[k])

    params = {
        "df": df,
        "staging_dir": r"E:\Projects\atmoz\catalog_staging",
        "report": True,
        "report_output": False,
        "report_input": False,
        "catalog_file": r"E:\Projects\atmoz\catalog_test.parquet",
        "note": True,
        "trek": True,
        "info": True
        }

    if k == 0 or k == len(dates):
        params["merge_catalog"] = False

    _write(**params)



#%%

@logs()
@report
def catalog_info(column, **kwargs):
    parquet_file = pq.ParquetFile("catalog_test.parquet")

    seen = set()

    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i, columns=[column])
        seen.update(table.column(column).to_pylist())  # Add values to set

    return list(seen)

cols_list = ["sensor_id", "start_time", "end_time", "variables"]
info = { col: catalog_info(col) for col in cols_list }


#%%

table = pq.read_table(r"E:\Projects\atmoz\catalog_test.parquet", columns=[])
num_rows = table.num_rows
print(f"Number of rows: {num_rows}")