# -*- coding: utf-8 -*-
"""
Created on 2026-06-09 23:13:37

@author: Maurice Roots

Description:
     - Trying to wrap Pandas Functionality over a dataframe so that it loops. 
"""
# %%


from typing import Optional, Iterator
from datetime import datetime
import warnings
import pandas as pd

from .AtmoFrame import AtmoFrame

# ------------------------------------------------------------------ #
# Schema helpers                                                      #
# ------------------------------------------------------------------ #

def _frame_schema(af: AtmoFrame) -> dict:
    """Extract the axis/units schema from a frame for comparison."""
    return {
        "_x":   af._x,
        "_y":   af._y,
        "_z":   af._z,
        "_lat": af._lat,
        "_lon": af._lon,
        "units": af.units,
    }


def _assert_schema_match(reference: dict, key: str, candidate: dict) -> None:
    mismatches = []
    for field in ("_x", "_y", "_z", "_lat", "_lon"):
        r, c = reference[field], candidate[field]
        if r != c:
            mismatches.append(f"  {field}: reference={r!r}, frames[{key!r}]={c!r}")
    if reference["units"] != candidate["units"]:
        mismatches.append(
            f"  units: reference={reference['units']!r}, frames[{key!r}]={candidate['units']!r}"
        )
    if mismatches:
        raise ValueError(
            f"AtmoDataset: frame '{key}' schema does not match the reference frame:\n"
            + "\n".join(mismatches)
        )

# ------------------------------------------------------------------ #
# AtmoDataset                                                         #
# ------------------------------------------------------------------ #

class AtmoDataset:
    """
    A named, metadata-bearing collection of AtmoFrames keyed by site/variable name.

    - All frames must share the same axis (x, y, z, lat, lon) and units schema.
    - time_start / time_end are inferred from each frame's x axis (if datetime),
      but can be overridden at construction.
    - Pandas ops broadcast across all frames and return a new AtmoDataset.
    - Chained ops (e.g. .resample().mean()) are supported via a lightweight proxy.

    Usage:
        ds = AtmoDataset(
            name="EPA Surface Ozone",
            source="EPA AQS",
            frames={"site_A": af1, "site_B": af2},
        )
        ds.dropna()
        ds.resample(rule="8h", on="Datetime").mean()
        ds["site_A"]
    """

    __slots__ = ("name", "source", "time_start", "time_end", "_frames", "_schema")

    def __init__(
        self,
        name:       str = None,
        source:     str = None,
        frames:     dict[str, AtmoFrame] = {AtmoFrame},
        time_start: Optional[datetime] = None,
        time_end:   Optional[datetime] = None,
    ):
        if not frames:
            raise ValueError("AtmoDataset: frames dict must not be empty.")

        # Type-check all frames
        for key, val in frames.items():
            if not isinstance(val, AtmoFrame):
                raise TypeError(
                    f"AtmoDataset: frames['{key}'] must be an AtmoFrame, "
                    f"got {type(val).__name__}."
                )

        # Schema consistency — first frame is the reference
        frame_list  = list(frames.items())
        ref_key, ref_af = frame_list[0]
        ref_schema  = _frame_schema(ref_af)
        for key, af in frame_list[1:]:
            _assert_schema_match(ref_schema, key, _frame_schema(af))

        object.__setattr__(self, "name",    name)
        object.__setattr__(self, "source",  source)
        object.__setattr__(self, "_frames", dict(frames))
        object.__setattr__(self, "_schema", ref_schema)

        # Infer time range from x axes (datetime only), then apply overrides
        inferred_start, inferred_end = self._infer_time_range(frames)
        object.__setattr__(self, "time_start", time_start if time_start is not None else inferred_start)
        object.__setattr__(self, "time_end",   time_end   if time_end   is not None else inferred_end)

    # ------------------------------------------------------------------ #
    # Time range inference                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_time_range(
        frames: dict[str, AtmoFrame],
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        mins, maxs = [], []
        for af in frames.values():
            if af._x is None:
                continue
            series = af.df[af._x]
            if not pd.api.types.is_datetime64_any_dtype(series):
                continue
            mins.append(series.min())
            maxs.append(series.max())
        if not mins:
            return None, None
        return pd.Timestamp(min(mins)).to_pydatetime(), pd.Timestamp(max(maxs)).to_pydatetime()

    # ------------------------------------------------------------------ #
    # Dict-like interface                                                  #
    # ------------------------------------------------------------------ #

    @property
    def frames(self) -> dict[str, AtmoFrame]:
        return self._frames

    def __getitem__(self, key: str) -> AtmoFrame:
        return self._frames[key]

    def __setitem__(self, key: str, value: AtmoFrame) -> None:
        if not isinstance(value, AtmoFrame):
            raise TypeError(
                f"AtmoDataset only holds AtmoFrames, got {type(value).__name__}."
            )
        _assert_schema_match(self._schema, key, _frame_schema(value))
        self._frames[key] = value
        # Re-infer time range
        start, end = self._infer_time_range(self._frames)
        object.__setattr__(self, "time_start", start)
        object.__setattr__(self, "time_end",   end)

    def __delitem__(self, key: str) -> None:
        del self._frames[key]

    def __contains__(self, key: str) -> bool:
        return key in self._frames

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[str]:
        return iter(self._frames)

    def items(self):
        return self._frames.items()

    def keys(self):
        return self._frames.keys()

    def values(self):
        return self._frames.values()

    def __repr__(self) -> str:
        time_range = (
            f"{self.time_start} → {self.time_end}"
            if self.time_start or self.time_end
            else "no time range"
        )
        return (
            f"AtmoDataset(name={self.name!r}, source={self.source!r}, "
            f"time={time_range}, n_frames={len(self._frames)}, "
            f"frames={list(self._frames.keys())})"
        )

    # ------------------------------------------------------------------ #
    # Pandas broadcast proxy                                               #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        frames = object.__getattribute__(self, "_frames")

        # Check the method exists on a pandas DataFrame
        if not hasattr(pd.DataFrame, name):
            raise AttributeError(f"AtmoDataset has no attribute '{name}'")

        def broadcast(*args, **kwargs):
            results:       dict[str, AtmoFrame | object] = {}
            intermediates: dict[str, object]             = {}

            for key, af in frames.items():
                try:
                    result = getattr(af.df, name)(*args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"AtmoDataset: pandas op '{name}' failed on frame '{key}': {e}"
                    ) from e

                if isinstance(result, pd.DataFrame):
                    results[key] = af._wrap(result)
                else:
                    # Could be a Resampler, GroupBy, Series, etc.
                    intermediates[key] = result
                    results[key]       = result

            # All DataFrames → new AtmoDataset
            if all(isinstance(v, AtmoFrame) for v in results.values()):
                return AtmoDataset(
                    name=self.name,
                    source=self.source,
                    frames={k: v for k, v in results.items()},
                )

            # Mixed or non-DataFrame → return proxy for chaining if chainable,
            # otherwise return the raw dict
            sample = next(iter(intermediates.values()), None)
            if sample is not None and not isinstance(sample, pd.Series):
                return _BroadcastProxy(self, intermediates)

            return results

        return broadcast

    # ------------------------------------------------------------------ #
    # Pickle support (__slots__)                                           #
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        return {
            "name":       self.name,
            "source":     self.source,
            "time_start": self.time_start,
            "time_end":   self.time_end,
            "n_frames":   len(self._frames),
            "schema":     self._schema,
            "frames":     {k: v.summary() for k, v in self._frames.items()},
        }



# ------------------------------------------------------------------ #
# Broadcast proxy for chained ops (e.g. .resample().mean())          #
# ------------------------------------------------------------------ #

class _BroadcastProxy:
    """
    Wraps an intermediate pandas object (e.g. Resampler) for each frame.
    Calling a method on this proxy calls it on every frame's intermediate
    and, if all results are DataFrames, returns a new AtmoDataset.
    """

    def __init__(self, dataset: AtmoDataset, intermediates: dict[str, object]):
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_intermediates", intermediates)

    def __getattr__(self, name: str):
        dataset      = object.__getattribute__(self, "_dataset")
        intermediates = object.__getattribute__(self, "_intermediates")

        def broadcast(*args, **kwargs):
            results: dict[str, AtmoFrame | object] = {}
            for key, obj in intermediates.items():
                try:
                    result = getattr(obj, name)(*args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"AtmoDataset: pandas op '{name}' failed on frame '{key}': {e}"
                    ) from e
                if isinstance(result, pd.DataFrame):
                    af = dataset[key]
                    # If x-axis was promoted to the index, bring it back as a column
                    if af._x is not None and af._x not in result.columns and result.index.name == af._x:
                        result = result.reset_index()
                    results[key] = af._wrap(result)
                else:
                    results[key] = result

            if all(isinstance(v, AtmoFrame) for v in results.values()):
                return AtmoDataset(
                    name=dataset.name,
                    source=dataset.source,
                    frames=results,
                )
            return results

        return broadcast
    