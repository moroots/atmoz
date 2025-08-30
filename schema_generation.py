# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 01:04:57 2025

@author: Magnolia
"""

import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq

schemas = {
    "lidar": pa.schema([
        ("timestamp", pa.timestamp("ns")),
        ("station_id", pa.dictionary(pa.int32(), pa.string())),
        ("sensor_type", pa.dictionary(pa.int32(), pa.string())),
        ("variable", pa.dictionary(pa.int32(), pa.string())),
        ("heights", pa.list_(pa.float64())),
        ("values", pa.list_(pa.float32())),
        ("qc_flag", pa.dictionary(pa.int32(), pa.string())),
        ("units", pa.dictionary(pa.int32(), pa.string())),
        ("source", pa.dictionary(pa.int32(), pa.string())),
    ]),

    "catalog": pa.schema([
        ("id", pa.string()),  # UUID
        ("file_path", pa.string()),
        ("min_timestamp", pa.timestamp("ns")),
        ("max_timestamp", pa.timestamp("ns")),
        ("variables", pa.list_(pa.string())),
        ("sensor_type", pa.string()),
        ("station_id", pa.string()),
        ("source", pa.string()),
        ("filenames", pa.list_(pa.string())),
        ("created_at", pa.timestamp("ns"))
    ])

    }

schema_dir = Path(r"E:\Projects\atmoz\atmoz\resources\schema")

for key in schemas.keys():
    with pa.OSFile(str(schema_dir / f"{key}.metadata"),  "wb") as f:
        pq.write_metadata(schemas[key], f)

