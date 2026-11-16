"""A small ICARTT (FFI 1001) reader: NASA airborne and campaign files such as the STAQS sondes.

>>> header, df = read_icartt("staqs-..._ozonesonde.ict")
>>> header["start_date"], header["variables"]["Ozone_ppmv"]       # ("2023-07-18", {"units": "ppmv", ...})

``df`` holds the data columns with every missing / limit-of-detection flag turned into NaN. The time columns are left as they are
(``Seconds_UTC`` is seconds since *start_date* 00:00 UTC), so the caller decides how to build a time index.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _split(line: str) -> list:
    return [x.strip() for x in line.rstrip("\r\n").split(",")]


def read_icartt(path: PathLike) -> Tuple[dict, pd.DataFrame]:
    """Read one ICARTT file (format index 1001) into ``(header, DataFrame)``.

    ``header`` has ``pi``, ``organization``, ``dataset``, ``mission``, ``start_date`` (ISO), ``variables``
    (``{name: {"units": ..., "description": ...}}``), ``flags`` (every missing/ULOD/LLOD flag value), ``comments``
    (``{KEY: text}`` from the ``KEY: value`` lines) and ``lines`` (the raw header lines).
    """
    path = Path(path)
    with open(path, "r", encoding="latin-1") as fh:
        first = _split(fh.readline())
        try:
            n_head, ffi = int(first[0]), int(first[1])
        except (ValueError, IndexError):
            raise ValueError(f"{path.name}: not an ICARTT file (first line should be 'NLHEAD, FFI').") from None
        if ffi != 1001:
            raise ValueError(f"{path.name}: ICARTT format {ffi} is not supported (only 1001).")
        lines = [",".join(first) + "\n"] + [fh.readline() for _ in range(n_head - 1)]

    y, m, d = (int(x) for x in _split(lines[6])[:3])
    n_vars = int(_split(lines[9])[0])
    flags = {float(x) for x in _split(lines[11]) if re.fullmatch(r"-?\d+(\.\d+)?", x)}
    variables: Dict[str, dict] = {}
    for line in lines[12:12 + n_vars]:
        parts = _split(line)
        variables[parts[0]] = {"units": parts[1] if len(parts) > 1 else None,
                               "description": ", ".join(parts[2:]) if len(parts) > 2 else None}
    comments = {}
    for line in lines[12 + n_vars:-1]:
        key, sep, value = line.partition(":")
        if sep and re.fullmatch(r"[A-Za-z0-9_]+", key.strip()):
            comments[key.strip()] = value.strip()
    for key in ("ULOD_FLAG", "LLOD_FLAG"):
        if re.fullmatch(r"-?\d+(\.\d+)?", comments.get(key, "")):
            flags.add(float(comments[key]))

    columns = _split(lines[-1])
    df = pd.read_csv(path, skiprows=n_head, names=columns, skipinitialspace=True, encoding="latin-1", low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    if flags:
        df = df.mask(df.isin(list(flags)))
    # variable declarations of the independent variable (line 8) are part of the column list too
    independent = _split(lines[8])
    variables.setdefault(independent[0], {"units": independent[1] if len(independent) > 1 else None,
                                          "description": ", ".join(independent[2:]) or None})
    header = {"pi": lines[1].strip(), "organization": lines[2].strip(), "dataset": lines[3].strip(),
              "mission": lines[4].strip(), "start_date": f"{y:04d}-{m:02d}-{d:02d}", "variables": variables,
              "flags": sorted(flags), "comments": comments, "lines": [ln.rstrip("\r\n") for ln in lines]}
    return header, df
