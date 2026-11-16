"""AtmozDataset: a named collection of AtmozFrames that share one axis schema.

What every ``download()`` returns. Keys are site/parameter (surface) or file/chunk names (profiles).

* every frame has the same axes (``lat``, ``lon``, ``alt`` column names). Units are per frame (``frame.units``), because one
  dataset can hold several pollutants (ppb for ozone, ug/m3 for PM2.5);
* ``time_start`` / ``time_end`` come from the frames' ``t`` index;
* ``metadata`` holds provenance (endpoint, files, rows, query);
* pandas methods broadcast to every frame: ``ds.resample("1h").mean()`` returns a new AtmozDataset.

>>> ds = AirNow.download(...)
>>> ds["840240053001/OZONE"]          # one AtmozFrame
>>> ds.select("*/OZONE")              # the frames whose key matches
>>> ds.frame                          # the only frame, when there is exactly one
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Dict, Iterator, Optional

import pandas as pd

from .AtmozFrame import AtmozFrame


def _schema(af: AtmozFrame) -> dict:
    return {"lat": af._lat, "lon": af._lon, "alt": af._alt}


def _check_schema(reference: dict, key: str, candidate: dict) -> None:
    problems = [f"  {k}: reference={reference[k]!r}, {key!r}={candidate[k]!r}"
                for k in ("lat", "lon", "alt") if reference[k] != candidate[k]]
    if problems:
        raise ValueError(f"AtmozDataset: frame {key!r} does not share the dataset schema:\n" + "\n".join(problems))


class AtmozDataset:
    """A named collection of :class:`AtmozFrame` with a shared schema."""

    __slots__ = ("name", "source", "metadata", "time_start", "time_end", "_frames", "_schema")

    def __init__(
        self,
        name: Optional[str] = None,
        source: Optional[str] = None,
        frames: Optional[Dict[str, AtmozFrame]] = None,
        metadata: Optional[Dict] = None,
    ):
        frames = dict(frames or {})
        for key, val in frames.items():
            if not isinstance(val, AtmozFrame):
                raise TypeError(f"AtmozDataset: frames[{key!r}] must be an AtmozFrame, got {type(val).__name__}.")
        schema = None
        for key, af in frames.items():
            if schema is None:
                schema = _schema(af)
            else:
                _check_schema(schema, key, _schema(af))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "time_start", None)
        object.__setattr__(self, "time_end", None)
        self._refresh_range()

    def _refresh_range(self) -> None:
        spans = [(af.df.index.min(), af.df.index.max()) for af in self._frames.values() if len(af.df)]
        object.__setattr__(self, "time_start", min(s for s, _ in spans) if spans else None)
        object.__setattr__(self, "time_end", max(e for _, e in spans) if spans else None)

    # -- dict-like ----------------------------------------------------------------------------------------------------

    @property
    def frames(self) -> Dict[str, AtmozFrame]:
        return self._frames

    @property
    def schema(self) -> Optional[dict]:
        """The shared schema: ``{"lat", "lon", "alt"}`` (None while empty)."""
        return None if self._schema is None else dict(self._schema)

    @property
    def frame(self) -> AtmozFrame:
        """The only frame (a shortcut for single-frame results)."""
        if len(self._frames) != 1:
            raise ValueError(f"`.frame` needs exactly one frame, found {len(self._frames)}: use ds[key] (keys: {self.keys_preview()}).")
        return next(iter(self._frames.values()))

    def keys_preview(self, n: int = 5) -> str:
        keys = list(self._frames)
        return ", ".join(map(repr, keys[:n])) + (f", ... ({len(keys)} in all)" if len(keys) > n else "")

    def __getitem__(self, key: str) -> AtmozFrame:
        return self._frames[key]

    def __setitem__(self, key: str, value: AtmozFrame) -> None:
        if not isinstance(value, AtmozFrame):
            raise TypeError(f"AtmozDataset only holds AtmozFrames, got {type(value).__name__}.")
        if self._schema is None:
            object.__setattr__(self, "_schema", _schema(value))
        else:
            _check_schema(self._schema, key, _schema(value))
        self._frames[key] = value
        self._refresh_range()

    def __delitem__(self, key: str) -> None:
        del self._frames[key]
        if not self._frames:
            object.__setattr__(self, "_schema", None)
        self._refresh_range()

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

    def select(self, pattern: str) -> "AtmozDataset":
        """The frames whose key matches the glob *pattern* (``"*/OZONE"``), as a new dataset."""
        chosen = {k: v for k, v in self._frames.items() if fnmatch(k, pattern)}
        return AtmozDataset(name=self.name, source=self.source, frames=chosen, metadata=self.metadata)

    def __repr__(self) -> str:
        rng = f"{self.time_start} -> {self.time_end}" if self.time_start is not None else "empty"
        return (f"AtmozDataset(name={self.name!r}, source={self.source!r}, time={rng}, "
                f"n_frames={len(self._frames)}, frames=[{self.keys_preview()}])")

    # -- pandas broadcast ---------------------------------------------------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith("__") or name in AtmozDataset.__slots__:
            raise AttributeError(name)
        if not hasattr(pd.DataFrame, name):
            raise AttributeError(f"AtmozDataset has no attribute {name!r}")
        frames = self._frames

        def broadcast(*args, **kwargs):
            return _collect(self, {k: getattr(af.df, name)(*args, **kwargs) for k, af in frames.items()}, frames, name)
        return broadcast

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def summary(self) -> dict:
        return {"name": self.name, "source": self.source, "time_start": self.time_start, "time_end": self.time_end,
                "n_frames": len(self._frames), "schema": self.schema, "metadata": dict(self.metadata),
                "frames": {k: v.summary() for k, v in self._frames.items()}}


def _collect(dataset: AtmozDataset, results: dict, frames: dict, name: str):
    """DataFrames -> a new AtmozDataset; resamplers/groupbys -> a proxy for chaining; anything else -> a dict."""
    if results and all(isinstance(r, pd.DataFrame) for r in results.values()):
        wrapped = {k: frames[k]._wrap(r) for k, r in results.items()}
        if all(isinstance(w, AtmozFrame) for w in wrapped.values()):
            return AtmozDataset(name=dataset.name, source=dataset.source, frames=wrapped, metadata=dataset.metadata)
        return wrapped
    sample = next(iter(results.values()), None)
    if sample is not None and not isinstance(sample, (pd.Series, pd.DataFrame)) and hasattr(sample, "__dict__"):
        return _Chain(dataset, results)
    return results


class _Chain:
    """Holds one intermediate pandas object per frame (a Resampler, say) so ``ds.resample("1h").mean()`` works."""

    def __init__(self, dataset: AtmozDataset, parts: dict):
        self._dataset = dataset
        self._parts = parts

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        frames = self._dataset.frames

        def broadcast(*args, **kwargs):
            return _collect(self._dataset, {k: getattr(p, name)(*args, **kwargs) for k, p in self._parts.items()}, frames, name)
        return broadcast
