# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:30:44 2025

@author: Maurice Roots

Last Updated: 2025-08-09
Purpose: Data Management & Data Catalog

"""

import pandas as pd

import numpy as np
import pyarrow as pa

from pathlib import Path
from datetime import datetime

from uuid import uuid4
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from atmoz import config

from custom_utilities.decorators import report, logs
from custom_utilities import messages

import subprocess
import importlib.resources as resources

class DataManagement:
    def __initi__(self):
        return

    def _load_schema(self, schema_name: str, **kwargs):
        schema_path = kwargs.get(
            "schema_dir",
            Path(config.get_package_path()) / "resources" / "schema"
            )

        path = str( schema_path / f"{schema_name}.metadata")
        print(path)
        return pq.read_schema(path)

    def _to_database(self, df, schema_name: str, schema_location = None):
        schema = self._load_schema(schema_name)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        file_uuid = uuid4()
        pq.write_table(table, f"{file_uuid}.parquet", compression="zstd")
        return

class DataCatalog(DataManagement):
    def __init__(self, **kwargs):
        catalog_path = config.read_config("database", "dirpath", **kwargs)
        self.catalog =  Path(catalog_path) / "catalog.parquet"
        self.schema = self._load_schema("catalog", **kwargs)
        return

    @logs
    @report
    def merge_catalog(staging_dir, catalog_file, **kwargs):
        staging_dir = Path(staging_dir).expanduser()
        catalog_file = Path(catalog_file).expanduser()

        if not staging_dir.is_dir():
            raise NotADirectoryError(
                f"Expected a directory at '{staging_dir}',"
                "but it is not a directory."
                )

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
        messages.trek(f"Read all parquet files in {staging_dir} as {new_dfs[0]} into {type(new_dfs)}", **kwargs)

        new_data = pd.concat(new_dfs, ignore_index=True)
        messages.trek(f"Concatenated all {new_files[0].stem} files in {staging_dir} into {type(new_data)}", **kwargs)

        updated_df = pd.concat([master_df, new_data], ignore_index=True)
        messages.trek(f"Read all parquet files in {staging_dir} as {new_dfs[0]} into {type(new_dfs)}", **kwargs)

        updated_df.to_parquet(catalog_file, index=False)
        messages.trek(f"Read all parquet files in {staging_dir} as {new_dfs[0]} into {type(new_dfs)}", **kwargs)

        for f in new_files:
            messages.trek("Removing {f}", **kwargs)
            f.unlink(missing_ok=True)

        messages.info("Staging files removed", **kwargs)

        messages.info(f"Merged {len(new_files)} files into '{catalog_file}'", **kwargs)

        return

    @logs
    @report
    def _write(self, df: pd.DataFrame, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be of type 'pd.DataFrame'")

        df = df.copy().explode("variables").reset_index(drop=True)

        messages.trek("DataFrame preped with '.explode('variables')' ", **kwargs)

        filename = self.staging / f"{uuid4()}.parquet"

        df.to_parquet(filename)
        messages.trek(f"DataFrame saved as {filename}", **kwargs)

        if kwargs.get("merge_catalog", True):
            self._merge_catalog(**kwargs)

        return

    @logs
    @report
    def catalog_info(self, column, **kwargs):
        parquet_file = pq.ParquetFile(self.catalog)

        seen = set()

        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i, columns=[column])
            seen.update(table.column(column).to_pylist())  # Add values to set

        return list(seen)

    @property
    def columns(self):
        return self.schema.names

    @logs
    @report
    def query(self, **filters):
        """
        Query catalog with optional filters:
        - variables (string match, partial ok)
        - station_id, source, filenames (exact match)
        - timestamp range: min_timestamp, max_timestamp
        """
        dataset = ds.dataset(self.catalog, format="parquet", partitioning="hive")

        expressions = []
        for col, val in filters.items():
            if col == "variables":
                # Variables stored as list<string> → check if any element matches substring
                expressions.append(ds.field("variables").isin([val]))
            elif col == "timestamp_range":
                start, end = val
                expressions.append((ds.field("min_timestamp") <= end) & (ds.field("max_timestamp") >= start))
            else:
                expressions.append(ds.field(col) == val)

        if expressions:
            combined = expressions[0]
            for expr in expressions[1:]:
                combined = combined & expr
            return dataset.to_table(filter=combined)
        else:
            return dataset.to_table()


params = {
    "schema_dir": r"E:\Projects\atmoz\atmoz\resources\schema\catalog.arrow",
    "config_path": r"E:/Projects/atmoz/atmoz/config.ini"
    }

catalog = DataCatalog()

#%%
num_files = 1000
rows = 1000
num_heights = 1000
sensor_ids = [random.randint(1, 1000) for _ in range(num_files)]
variables = ["temperature", "humidity", "wind", "ozone", "nitrogen-dioxide", "height", "backscatter"]

# Files List Demo
timestamps = pd.date_range("2021-01-01", periods=num_files+1, freq="1d")
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


# Query by station and variable
result = catalog.query(station_id="ST123", variables="temperature")
print(result)




#%%

# Define your time range and variable
start_time = pd.Timestamp("2025-01-01 03:00")
end_time = pd.Timestamp("2025-01-01 06:00")
target_variable = "temperature"

# Open the dataset
dataset = ds.dataset("profiles_nested.parquet", format="parquet")

# Build filter expression: timestamp BETWEEN start_time AND end_time AND variable == target_variable
filter_expr = (
    (ds.field("timestamp") >= start_time) &
    (ds.field("timestamp") <= end_time) &
    (ds.field("variable") == target_variable)
)

# Scan with filter
table_filtered = dataset.to_table(filter=filter_expr)

# Convert to pandas
df_filtered = table_filtered.to_pandas()

print(df_filtered)



#%%



if __name__ == "__main__":
    periods = 10000
    profile_length = 1000  # number of heights per profile

    timestamps = pd.date_range("2025-01-01 00:00", periods=periods, freq="10min")
    station_ids = ["ST001"] * periods
    sensor_types = ["LIDAR"] * periods
    variables = ["temperature"] * periods

    # Heights fixed for all profiles
    heights = np.arange(profile_length) * 10  # shape (1000,)

    # Prepare columns for DataFrame: lists of heights and values per timestamp
    heights_list = [heights.tolist()] * periods  # same for every row, repeated

    # Generate random profile values (periods x profile_length)
    values_array = np.random.rand(periods, profile_length).astype(np.float32)

    # Prepare other metadata
    qc_flags = ["good"] * periods
    units = ["C"] * periods
    sources = ["simulator"] * periods

    # Create pandas DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_ids,
        "sensor_type": sensor_types,
        "variable": variables,
        "heights": heights_list,
        "values": list(values_array),  # list of np arrays auto-converted to lists
        "qc_flag": qc_flags,
        "units": units,
        "source": sources
    })

    # Convert categorical columns to save space
    for col in ["station_id", "sensor_type", "variable", "qc_flag", "units", "source"]:
        df[col] = df[col].astype("category")


    cols_list = ["sensor_id", "start_time", "end_time", "variables"]
    info = { col: catalog_info(col, report=True) for col in cols_list }