# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 00:19:10 2025

@author: Maurice Roots

"""

import ast
import argparse
import pandas as pd
from pathlib import Path
from custom_utilities.decorators import report, logs
from custom_utilities import messages

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parquet files from staging into catalog.")

    parser.add_argument(
        "--kwargs",
        type=str,
        help="Extra keyword arguments as a dictionary string"
        )

    args = parser.parse_args()
    kwargs_dict = ast.literal_eval(args.kwargs) if args.kwargs else {}

    merge_catalog(**kwargs_dict)



