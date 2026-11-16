"""AtmozFrame: one table of atmospheric data (a DataFrame plus axes, units and metadata).

Conventions (see ``docs/design-api.md``):

* **time is the index.** ``t`` is the frame's tz-aware UTC ``DatetimeIndex``. It is required. A long-format profile repeats
  a timestamp once per altitude, so the index may contain duplicates.
* **other axes are columns**: ``lat``, ``lon`` and ``alt`` (altitude, km) are names of columns, or ``None``.
* **units live in one place**: the ``units`` mapping ``{column: unit}``. ``ppbv`` is normalised to ``ppb``.
* **metadata** is a free dict (provenance, site info) that travels with the frame.

>>> af = AtmozFrame(df, units={"value": "ppb"}, lat="lat", lon="lon", metadata={"site": "840240053001"})
>>> af.t            # the DatetimeIndex
>>> af.alt          # df["alt"] or None
>>> af.resample("1h").mean()    # pandas methods keep the axes, units and metadata
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import pandas as pd

AXES = ("lat", "lon", "alt")

# Spellings normalised on construction: volume mixing ratios are stored as ppb / ppm / ppt.
UNIT_ALIASES: Dict[str, str] = {"ppbv": "ppb", "ppmv": "ppm", "pptv": "ppt", "degC": "°C"}

_KNOWN_UNITS: frozenset = frozenset({
    "1", "%", "s", "min", "h", "d",
    "m", "km", "hm",
    "Pa", "hPa", "kPa", "bar", "mbar",
    "K", "°C",
    "kg", "g", "mg", "µg",
    "kg/kg", "g/kg", "mg/kg", "µg/kg",
    "kg/m3", "g/m3", "mg/m3", "µg/m3", "molecules/cm3",
    "mol/mol", "ppm", "ppb", "ppt",
    "m/s", "km/h", "kn",
    "km^-1.sr^-1", "1/(km sr)",
    "deg", "rad",
    "W/m2", "W/m²", "J/m2", "J/m²",
})


def normalize_unit(unit: str) -> str:
    """The canonical spelling of *unit* (``ppbv`` -> ``ppb``); other strings are returned unchanged."""
    return UNIT_ALIASES.get(unit, unit)


def _as_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"AtmozFrame needs a DatetimeIndex (time is the index), got {type(idx).__name__}. "
                        "Use df.set_index('<time column>') first.")
    if idx.tz is None:
        raise ValueError("AtmozFrame needs a timezone-aware index (UTC). Localise it first, e.g. "
                         "df.index = df.index.tz_localize('UTC').")
    if str(idx.tz) != "UTC":
        df = df.copy()
        df.index = idx.tz_convert("UTC")
    return df


class AtmozFrame:
    """One table: a DataFrame indexed by UTC time, with named axis columns, units and metadata."""

    __slots__ = ("df", "units", "metadata", "_lat", "_lon", "_alt")

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        units: Optional[Dict[str, str]] = None,
        lat: Optional[str] = None,
        lon: Optional[str] = None,
        alt: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"AtmozFrame wraps a pandas DataFrame, got {type(df).__name__}.")
        df = _as_utc_index(df)
        for col, role in ((lat, "lat"), (lon, "lon"), (alt, "alt")):
            if col is not None and col not in df.columns:
                raise KeyError(f"AtmozFrame: column {col!r} given as '{role}' is not in the DataFrame "
                               f"(columns: {list(df.columns)}).")
        units = {col: normalize_unit(u) for col, u in (units or {}).items()}
        for col, unit in units.items():
            if col not in df.columns:
                raise KeyError(f"AtmozFrame: units given for column {col!r}, which is not in the DataFrame.")
            if unit not in _KNOWN_UNITS:
                warnings.warn(f"AtmozFrame: unit {unit!r} for column {col!r} is not a recognised unit.", UserWarning, stacklevel=2)
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_lat", lat)
        object.__setattr__(self, "_lon", lon)
        object.__setattr__(self, "_alt", alt)

    # -- axes ---------------------------------------------------------------------------------------------------------

    @property
    def t(self) -> pd.DatetimeIndex:
        """The time axis: the tz-aware UTC index."""
        return self.df.index

    @property
    def lat(self) -> Optional[pd.Series]:
        return self.df[self._lat] if self._lat else None

    @property
    def lon(self) -> Optional[pd.Series]:
        return self.df[self._lon] if self._lon else None

    @property
    def alt(self) -> Optional[pd.Series]:
        return self.df[self._alt] if self._alt else None

    @property
    def location(self) -> Optional[tuple]:
        """``(lat, lon)`` of the frame (the median of the lat/lon columns), or None when it has no horizontal position."""
        if not (self._lat and self._lon):
            return None
        lat, lon = self.df[self._lat].median(), self.df[self._lon].median()
        return None if pd.isna(lat) or pd.isna(lon) else (float(lat), float(lon))

    @property
    def alt_reference(self) -> Optional[str]:
        """What ``alt`` is measured from: ``"asl"`` (above sea level), ``"agl"`` (above ground) or None (not stated).

        Stored in ``metadata["alt_reference"]``; ``metadata["site_elevation_km"]`` is the ground elevation of the site
        (above sea level), which converts between the two.
        """
        return self.metadata.get("alt_reference")

    @property
    def axes(self) -> Dict[str, Optional[str]]:
        """Column names of the axes (``t`` is always the index)."""
        return {"t": "index", "lat": self._lat, "lon": self._lon, "alt": self._alt}

    # -- pandas proxy -------------------------------------------------------------------------------------------------

    def _wrap(self, df: pd.DataFrame) -> "AtmozFrame":
        """Re-wrap a derived DataFrame, keeping the axes, units and metadata that still apply (no re-validation)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df  # the result no longer has time as its index (e.g. reset_index): hand back plain pandas
        cols = set(df.columns)
        out = object.__new__(AtmozFrame)
        object.__setattr__(out, "df", df)
        object.__setattr__(out, "units", {k: v for k, v in self.units.items() if k in cols})
        object.__setattr__(out, "metadata", dict(self.metadata))
        for slot in ("_lat", "_lon", "_alt"):
            name = getattr(self, slot)
            object.__setattr__(out, slot, name if name in cols else None)
        return out

    def _rewrap(self, result):
        if isinstance(result, pd.DataFrame):
            return self._wrap(result)
        if not isinstance(result, (pd.Series, pd.Index)) and hasattr(result, "__dict__") and not callable(result):
            return _Chain(self, result)   # a Resampler / GroupBy: wrap what its methods return
        return result

    def __getattr__(self, name: str):
        if name.startswith("__") or name in AtmozFrame.__slots__:
            raise AttributeError(name)
        attr = getattr(self.df, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                return self._rewrap(attr(*args, **kwargs))
            return wrapper
        return self._wrap(attr) if isinstance(attr, pd.DataFrame) else attr

    def __getitem__(self, key):
        result = self.df[key]
        return self._wrap(result) if isinstance(result, pd.DataFrame) else result

    def __setitem__(self, key, value):
        self.df[key] = value

    def __len__(self):
        return len(self.df)

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        axes = ", ".join(f"{k}={v}" + (f" [{self.units[v]}]" if v in self.units else "")
                         for k, v in (("lat", self._lat), ("lon", self._lon), ("alt", self._alt)) if v)
        units = ", ".join(f"{c} [{u}]" for c, u in self.units.items() if c not in (self._lat, self._lon, self._alt))
        head = f"AtmozFrame t=index" + (f", {axes}" if axes else "") + (f"; {units}" if units else "")
        return f"{head}\n{self.df!r}"

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    # -- helpers ------------------------------------------------------------------------------------------------------

    def with_units(self, **col_units: str) -> "AtmozFrame":
        """A new frame with more column units: ``af.with_units(value="ppb")``."""
        return AtmozFrame(self.df, units={**self.units, **col_units}, lat=self._lat, lon=self._lon, alt=self._alt,
                          metadata=self.metadata)

    def with_axes(self, *, lat: Optional[str] = None, lon: Optional[str] = None, alt: Optional[str] = None) -> "AtmozFrame":
        """A new frame with some axis columns changed."""
        return AtmozFrame(self.df, units=self.units, metadata=self.metadata,
                          lat=lat if lat is not None else self._lat,
                          lon=lon if lon is not None else self._lon,
                          alt=alt if alt is not None else self._alt)

    def with_metadata(self, **updates) -> "AtmozFrame":
        """A new frame with *updates* merged into the metadata."""
        return AtmozFrame(self.df, units=self.units, lat=self._lat, lon=self._lon, alt=self._alt,
                          metadata={**self.metadata, **updates})

    def with_alt_reference(self, target: str) -> "AtmozFrame":
        """The frame with ``alt`` re-expressed above sea level (``"asl"``) or above ground (``"agl"``).

        Needs ``metadata["alt_reference"]`` and, to change reference, ``metadata["site_elevation_km"]`` (``alt`` in km).
        """
        if target not in ("asl", "agl"):
            raise ValueError("target must be 'asl' or 'agl'.")
        if self._alt is None:
            raise ValueError("This frame has no altitude axis.")
        current = self.alt_reference
        if current not in ("asl", "agl"):
            raise ValueError("The altitude reference of this frame is not stated: set metadata['alt_reference'] "
                             "to 'asl' or 'agl' (frame.with_metadata(alt_reference=...)).")
        if current == target:
            return self
        elevation = self.metadata.get("site_elevation_km")
        if elevation is None or pd.isna(elevation):
            raise ValueError(f"Converting {current} -> {target} needs metadata['site_elevation_km'] (the site's ground "
                             "elevation above sea level, km).")
        if self.units.get("alt", "km") != "km":
            raise ValueError("Altitude reference conversion needs alt in km.")
        df = self.df.copy()
        df[self._alt] = df[self._alt] - float(elevation) if target == "agl" else df[self._alt] + float(elevation)
        return AtmozFrame(df, units=self.units, lat=self._lat, lon=self._lon, alt=self._alt,
                          metadata={**self.metadata, "alt_reference": target})

    def unit(self, col: str) -> Optional[str]:
        return self.units.get(col)

    def convert_units(self, col: str, target: str) -> "AtmozFrame":
        """Convert one column to another unit with pint (mixing-ratio aliases such as ppbv are understood)."""
        from atmoz.resources.atmoz_dataclasses import _get_ureg
        current = self.units.get(col)
        if current is None:
            raise ValueError(f"Column {col!r} has no declared unit; set it with with_units() first.")
        target = normalize_unit(target)
        reg = _get_ureg()
        factor = (1 * reg(UNIT_TO_PINT.get(current, current))).to(UNIT_TO_PINT.get(target, target)).magnitude
        df = self.df.copy()
        df[col] = df[col] * factor
        return AtmozFrame(df, units={**self.units, col: target}, lat=self._lat, lon=self._lon, alt=self._alt,
                          metadata=self.metadata)

    def summary(self) -> dict:
        return {
            "rows": len(self.df),
            "columns": list(self.df.columns),
            "t": [self.df.index.min(), self.df.index.max()] if len(self.df) else [None, None],
            "axes": self.axes,
            "units": dict(self.units),
            "nulls": int(self.df.isnull().sum().sum()),
        }


class _Chain:
    """Wraps an intermediate pandas object (e.g. a Resampler) so ``af.resample('1h').mean()`` returns an AtmozFrame."""

    def __init__(self, frame: AtmozFrame, obj):
        self._frame = frame
        self._obj = obj

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr
        return lambda *a, **k: self._frame._rewrap(attr(*a, **k))


# pint spellings for units that the registry does not know by that name
UNIT_TO_PINT: Dict[str, str] = {"ppb": "ppbv", "ppm": "ppmv", "ppt": "pptv"}
