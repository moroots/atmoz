"""Balloon sondes (ozonesondes and wind/met sondes) from ICARTT files, e.g. NASA LaRC STAQS.

>>> ds = Sonde.read("data/Sondes")                       # a file, a directory of .ict files or a list
>>> flight = ds["staqs-..._ozonesonde.ict"]              # one AtmozFrame per flight
>>> flight.df[["alt", "value", "lat", "lon"]]            # value = ozone in ppb

One :class:`~atmoz.AtmozFrame` per file (a flight). The index is the UTC time of every reading (a sonde is a trajectory: each row
has its own time), ``alt`` is the altitude in km above sea level (``alt_reference="asl"``; the launch site elevation is in
``metadata["site_elevation_km"]`` when the file gives it), ``lat``/``lon`` are per row (the balloon drifts; NaN where the file has no GPS) and the frame's
``location`` is the median position (None without GPS). Ozonesonde files get ``value`` = ozone in ppb; wind/met sondes have no ``value`` (pick a
column, e.g. ``atmoz.collocate(..., cols={"sonde": "WindSpeed_mps"})``). Every other variable keeps its file name and unit.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozDataset import AtmozDataset
from atmoz.dataclasses.AtmozFrame import AtmozFrame, normalize_unit
from atmoz.resources.icartt import read_icartt

PathLike = Union[str, Path]

_ALT = {"Altitude_km": 1.0, "Altitude_m_MSL": 0.001}          # column -> factor to km (above sea level)
_ALT_AGL = {"Altitude_m_AGL": 0.001}
_LAT, _LON = ("Latitude_deg", "Latitude"), ("Longitude_deg", "Longitude")
_OZONE = ("Ozone_ppmv", "Ozone_ppbv")                          # ozone mixing-ratio columns (the declared unit sets the factor)
_TIME = ("Seconds_UTC", "UTC", "Time_UTC")


def _first(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    return next((n for n in names if n in df.columns), None)


def _paths(source: Union[PathLike, Iterable[PathLike]]) -> List[Path]:
    items = [source] if isinstance(source, (str, Path)) else list(source)
    out: List[Path] = []
    for item in items:
        p = Path(item)
        out.extend(sorted(p.glob("*.ict")) if p.is_dir() else [p])
    if not out:
        raise FileNotFoundError(f"No .ict files found in {source!r}.")
    return out


def _unwrap_seconds(sec: pd.Series) -> pd.Series:
    """Seconds since the start date, continuing past midnight when the file's clock wraps back to 0."""
    s = sec.astype(float)
    filled = s.ffill().bfill()
    wraps = (filled.diff() < -43200).cumsum()
    return s + 86400.0 * wraps


def _ozone_ppb(raw: pd.DataFrame, column: str, name: str, declared: Optional[str]) -> pd.Series:
    """Ozone in ppb. The unit the file declares is checked against its own ozone partial pressure and pressure, when it has them.

    ppb = partial pressure (mPa) / pressure (hPa) * 1e4. Some STAQS files label a ppbv column ``Ozone_ppmv``; trusting the label
    would give values 1000 times too large, so the unit implied by the physics wins, with a warning.
    """
    factor = 1000.0 if str(declared or column).lower().endswith(("ppmv", "ppm")) else 1.0
    if "Ozone_mPa" in raw.columns and "Pressure_hPa" in raw.columns:
        derived = raw["Ozone_mPa"] / raw["Pressure_hPa"] * 1e4          # ppb implied by the file's own partial pressure
        ratio = (raw[column] / derived).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[ratio > 0]
        if len(ratio) >= 10:
            implied = float(np.median(ratio))
            if abs(np.log10(implied)) < 0.3:
                actual = 1.0                                              # the column is in ppb
            elif abs(np.log10(implied * 1000.0)) < 0.3:
                actual = 1000.0                                           # the column is in ppm
            else:
                actual = factor                                           # inconclusive: trust the declared unit
            if actual != factor:
                warnings.warn(f"{name}: {column} is declared in {'ppmv' if factor == 1000.0 else 'ppbv'} but matches "
                              f"{'ppmv' if actual == 1000.0 else 'ppbv'} (checked against Ozone_mPa and Pressure_hPa); using that.",
                              UserWarning, stacklevel=3)
                factor = actual
    return raw[column] * factor


class Sonde:
    """Read balloon-sonde ICARTT files into an :class:`AtmozDataset`."""

    @staticmethod
    def _frame(path: Path) -> AtmozFrame:
        header, raw = read_icartt(path)
        tcol = _first(raw, _TIME)
        if tcol is None:
            raise ValueError(f"{path.name}: no UTC time column (looked for {list(_TIME)}).")
        acol = _first(raw, _ALT)
        if acol is None:
            raise ValueError(f"{path.name}: no altitude column (looked for {list(_ALT)}).")
        latcol, loncol = _first(raw, _LAT), _first(raw, _LON)
        t0 = pd.Timestamp(header["start_date"], tz="UTC")
        time = t0 + pd.to_timedelta(_unwrap_seconds(raw[tcol]), unit="s")
        data = {"alt": raw[acol] * _ALT[acol]}
        units = {"alt": "km", "lat": "deg", "lon": "deg"}
        ocol = _first(raw, _OZONE)
        if ocol is not None:
            data["value"] = _ozone_ppb(raw, ocol, path.name, (header["variables"].get(ocol) or {}).get("units"))
            units["value"] = "ppb"
        data["lat"] = raw[latcol] if latcol else np.nan      # a file without GPS keeps lat/lon as NaN columns
        data["lon"] = raw[loncol] if loncol else np.nan
        agl = _first(raw, _ALT_AGL)
        skip = {tcol, acol, latcol, loncol, agl, ocol, "Index_number"}
        if agl is not None:
            data["alt_agl"] = raw[agl] * _ALT_AGL[agl]
            units["alt_agl"] = "km"
        for name in raw.columns:
            if name in skip:
                continue
            data[name] = raw[name]
            unit = (header["variables"].get(name) or {}).get("units")
            if unit:
                units[name] = normalize_unit(unit)
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex(time, name="time")
        df = df[df["alt"].notna()].sort_index()

        text = " ".join(header["comments"].get("LOCATION", "").split())
        found = re.search(r"([-\d.]+)\s*m\s*(?:altitude\s*)?(?:AMSL|MSL|ASL)", text, flags=re.IGNORECASE)
        meta = {"file": path.name, "pi": header["pi"], "platform": header["comments"].get("PLATFORM"),
                "dataset": header["dataset"], "mission": header["mission"], "location": header["comments"].get("LOCATION"),
                "revision": header["comments"].get("REVISION"), "start_date": header["start_date"],
                "alt_reference": "asl", "site_elevation_km": float(found.group(1)) / 1000.0 if found else None,
                "variables": header["variables"], "comments": header["comments"]}
        if len(df):
            meta["launch"] = df.index[0]
        return AtmozFrame(df, units={k: v for k, v in units.items() if k in df.columns}, lat="lat", lon="lon", alt="alt",
                          metadata=meta)

    @classmethod
    def read(cls, source: Union[PathLike, Iterable[PathLike]]) -> AtmozDataset:
        """Read one or more ICARTT sonde files (paths, or a directory of ``.ict`` files) into an AtmozDataset.

        One frame per file, keyed by the file name. Raises ``ValueError`` for a file that is not ICARTT 1001 or lacks
        a time, altitude or position column.
        """
        files = _paths(source)
        frames: Dict[str, AtmozFrame] = {p.name: cls._frame(p) for p in files}
        return AtmozDataset(name="Sonde", source="ICARTT sonde", frames=frames,
                            metadata={"endpoint": "sonde", "files": [p.name for p in files],
                                      "rows": {k: len(f) for k, f in frames.items()}})
