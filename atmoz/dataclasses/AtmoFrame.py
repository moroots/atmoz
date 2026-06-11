# -*- coding: utf-8 -*-
"""
Created on 2026-06-09 20:47:14

@author: Maurice Roots

Description:
     - Hoping this makes sense to me in the future (i.e., pd.DataFrame -> atmoz.AtmoFrame = {df, x, y, c, lat, lon})
"""
# %%


from __future__ import annotations
from typing import Optional, Dict
import warnings
import pandas as pd


# ------------------------------------------------------------------ #
# Valid SI unit strings                                               #
# ------------------------------------------------------------------ #
_SI_UNITS: frozenset[str] = frozenset({
    # dimensionless / time
    "1", "%", "s", "min", "h", "d",
    # length / altitude
    "m", "km", "hm",
    # pressure
    "Pa", "hPa", "kPa", "bar", "mbar",
    # temperature
    "K", "°C", "degC",
    # mass
    "kg", "g", "mg", "µg",
    # concentration / mixing ratio
    "kg/kg", "g/kg", "mg/kg", "µg/kg",
    "kg/m3", "g/m3", "mg/m3", "µg/m3",
    "mol/mol", "ppm", "ppb", "ppt",
    # wind / velocity
    "m/s", "km/h", "kn",
    # angle / coordinates
    "deg", "rad",
    # humidity
    "kg/kg", "g/kg", "%",
    # radiation
    "W/m2", "W/m²", "J/m2", "J/m²",
})

# Expected dtypes for each axis role
_AXIS_EXPECTED_DTYPES: dict[str, tuple[str, ...]] = {
    "x":   ("datetime64", "float", "int", "timedelta64"),
    "y":   ("float", "int"),
    "z":   ("float", "int"),
    "lat": ("float", "int"),
    "lon": ("float", "int"),
}


def _check_column_exists(col: str, role: str, columns: pd.Index) -> None:
    """Raise KeyError if col is not in columns."""
    if col not in columns:
        raise KeyError(
            f"AtmoFrame: column '{col}' assigned to '{role}' not found in DataFrame. "
            )


def _check_axis_dtype(series: pd.Series, role: str) -> None:
    """Warn if the axis column dtype looks wrong for its role."""
    expected = _AXIS_EXPECTED_DTYPES.get(role, ())
    dtype_str = str(series.dtype)
    if not any(e in dtype_str for e in expected):
        warnings.warn(
            f"AtmoFrame: '{role}' axis column '{series.name}' has dtype '{series.dtype}',",
            UserWarning,
            stacklevel=3,
        )


def _check_unit(col: str, unit: str) -> None:
    """Warn if unit string is not a recognised SI unit."""
    if unit not in _SI_UNITS:
        warnings.warn(
            f"AtmoFrame: unit '{unit}' for column '{col}' is not a recognised SI unit. ",
            UserWarning,
            stacklevel=3,
        )


class AtmoFrame:
    """
    A pandas DataFrame wrapper for atmospheric observation data.

    Construction:  AtmoFrame(df=df, x="time", y="altitude", z="pressure")
    af._x  → "time"        (column name string)
    af.x   → df["time"]    (pd.Series)

    Validation:
    - Missing columns for any axis/coord → raises KeyError
    - Wrong dtype for an axis            → warns
    - Unrecognised SI unit               → warns
    """

    __slots__ = ("df", "units", "_x", "_y", "_z", "_lat", "_lon")

    def __init__(self,
                 df: pd.DataFrame,
                 *,
                 units: dict[str, str] | None = None,
                 x:   Optional[str] = None,
                 y:   Optional[str] = None,
                 z:   Optional[str] = None,
                 lat: Optional[str] = None,
                 lon: Optional[str] = None,
                 metadata: Dict = {},
                 ):
        
        units = units or {}

        # --- column-existence checks (raise) ---
        for col, role in [(x, "x"), (y, "y"), (z, "z"), (lat, "lat"), (lon, "lon")]:
            if col is not None:
                _check_column_exists(col, role, df.columns)
        for col in units:
            _check_column_exists(col, f"units['{col}']", df.columns)

        # --- dtype checks (warn) ---
        for col, role in [(x, "x"), (y, "y"), (z, "z"), (lat, "lat"), (lon, "lon")]:
            if col is not None:
                _check_axis_dtype(df[col], role)

        # --- unit string checks (warn) ---
        for col, unit in units.items():
            _check_unit(col, unit)

        self.df    = df
        self.units = units
        self._x    = x
        self._y    = y
        self._z    = z
        self._lat  = lat
        self._lon  = lon

    # ------------------------------------------------------------------ #
    # Proxy                                                                #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        attr = getattr(self.df, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    return self._wrap(result)
                return result
            return wrapper
        if isinstance(attr, pd.DataFrame):
            return self._wrap(attr)
        return attr

    def __getitem__(self, key):
        result = self.df[key]
        if isinstance(result, pd.DataFrame):
            return self._wrap(result)
        return result

    def __setitem__(self, key, value):
        self.df[key] = value

    def __len__(self):
        return len(self.df)

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        axes = ", ".join(
            f"{k}={v}" + (f" [{self.units[v]}]" if v and v in self.units else "")
            for k, v in [("x", self._x), ("y", self._y), ("z", self._z),
                         ("lat", self._lat), ("lon", self._lon)]
            if v
        )
        return f"AtmoFrame [{axes or 'no axes set'}]\n{self.df.__repr__()}"

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    # ------------------------------------------------------------------ #
    # Axis properties  (string → Series)                                  #
    # ------------------------------------------------------------------ #

    @property
    def x(self) -> Optional[pd.Series]:
        return self.df[self._x] if self._x else None

    @property
    def y(self) -> Optional[pd.Series]:
        return self.df[self._y] if self._y else None

    @property
    def z(self) -> Optional[pd.Series]:
        return self.df[self._z] if self._z else None

    @property
    def lat(self) -> Optional[pd.Series]:
        return self.df[self._lat] if self._lat else None

    @property
    def lon(self) -> Optional[pd.Series]:
        return self.df[self._lon] if self._lon else None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _wrap(self, df: pd.DataFrame) -> AtmoFrame:
        """Re-wrap without re-validating (internal use only)."""
        cols = set(df.columns)
        af = object.__new__(AtmoFrame)
        object.__setattr__(af, "df",    df)
        object.__setattr__(af, "units", {k: v for k, v in self.units.items() if k in cols})
        object.__setattr__(af, "_x",    self._x   if self._x   in cols else None)
        object.__setattr__(af, "_y",    self._y   if self._y   in cols else None)
        object.__setattr__(af, "_z",    self._z   if self._z   in cols else None)
        object.__setattr__(af, "_lat",  self._lat if self._lat in cols else None)
        object.__setattr__(af, "_lon",  self._lon if self._lon in cols else None)
        return af

    # ------------------------------------------------------------------ #
    # Mutation helpers                                                     #
    # ------------------------------------------------------------------ #

    def with_units(self, **col_units: str) -> AtmoFrame:
        """Usage: af.with_units(temp="K", pressure="hPa")"""
        return AtmoFrame(
            df=self.df, units={**self.units, **col_units},
            x=self._x, y=self._y, z=self._z, lat=self._lat, lon=self._lon,
        )

    def with_axes(
        self,
        x:   Optional[str] = None,
        y:   Optional[str] = None,
        z:   Optional[str] = None,
        lat: Optional[str] = None,
        lon: Optional[str] = None,
    ) -> AtmoFrame:
        return AtmoFrame(
            df=self.df, units=self.units,
            x=x     if x   is not None else self._x,
            y=y     if y   is not None else self._y,
            z=z     if z   is not None else self._z,
            lat=lat if lat is not None else self._lat,
            lon=lon if lon is not None else self._lon,
        )

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def unit(self, col: str) -> Optional[str]:
        return self.units.get(col)

    def summary(self) -> dict:
        return {
            "rows":    self.df.shape[0],
            "cols":    self.df.shape[1],
            "nulls":   int(self.df.isnull().sum().sum()),
            "columns": list(self.df.columns),
            "axes":    {"x": self._x, "y": self._y, "z": self._z},
            "coords":  {"lat": self._lat, "lon": self._lon},
            "units":   self.units,
        }

AtmoFrame.__module__ = "atmoz.dataclasses.AtmoFrame"