"""Vertical summaries of profile data: layer means and mean profiles.

>>> atmoz.vertical.layer_mean(tolnet_frame, 0, 2)              # 0-2 km mean ozone, one value per profile time
>>> atmoz.vertical.mean_profile(tolnet_frame, bins=0.5)        # mean/std/count per altitude bin
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozDataset import AtmozDataset
from atmoz.dataclasses.AtmozFrame import AtmozFrame

Data = Union[AtmozFrame, AtmozDataset]


def _in_reference(af: AtmozFrame, reference: Optional[str]) -> AtmozFrame:
    if af.axes["alt"] is None:
        raise ValueError("This frame has no altitude axis.")
    return af if reference is None else af.with_alt_reference(reference)


def _layer(af: AtmozFrame, bottom: float, top: float, col: str, reference: Optional[str], min_points: int) -> AtmozFrame:
    if not bottom < top:
        raise ValueError("bottom must be below top.")
    if col not in af.df.columns:
        raise KeyError(f"Column {col!r} is not in the frame.")
    f = _in_reference(af, reference)
    alt = f.df[f.axes["alt"]]
    inside = f.df[(alt >= bottom) & (alt <= top)]
    grouped = inside[col].groupby(level=0)
    mean = grouped.mean()
    mean[grouped.count() < min_points] = np.nan
    mean = mean.reindex(f.df.index.unique())
    out = pd.DataFrame({"value": mean})
    for key in ("lat", "lon"):
        name = f.axes[key]
        if name is not None:
            out[name] = f.df[name].groupby(level=0).median().reindex(out.index)
    units = {"value": f.units[col]} if col in f.units else {}
    return AtmozFrame(out, units={**units, **{k: v for k, v in f.units.items() if k in out.columns}},
                      lat=f.axes["lat"], lon=f.axes["lon"],
                      metadata={**f.metadata, "layer_km": (bottom, top), "layer_reference": reference or f.alt_reference,
                                "layer_column": col, "min_points": min_points})


def layer_mean(data: Data, bottom: float, top: float, col: str = "value", *, reference: Optional[str] = None,
               min_points: int = 3) -> Data:
    """Mean of *col* between *bottom* and *top* km, one value per profile time.

    Parameters
    ----------
    bottom, top:
        Layer limits in km.
    reference:
        ``"agl"`` or ``"asl"``: the altitude reference the limits are in. Frames are converted with
        :meth:`AtmozFrame.with_alt_reference` (needs the site elevation). Default: each frame's own reference.
    min_points:
        Profiles with fewer valid points inside the layer give NaN.

    Returns
    -------
    AtmozFrame or AtmozDataset
        A time series (column ``value``, the unit of *col*; ``lat``/``lon`` kept) per frame; same container as the input.
    """
    if isinstance(data, AtmozFrame):
        return _layer(data, bottom, top, col, reference, min_points)
    if isinstance(data, AtmozDataset):
        frames = {k: _layer(f, bottom, top, col, reference, min_points) for k, f in data.items()}
        return AtmozDataset(name=data.name, source=data.source, frames=frames, metadata=data.metadata)
    raise TypeError(f"expected an AtmozFrame or AtmozDataset, got {type(data).__name__}.")


def mean_profile(data: Data, col: str = "value", *, bins: Union[float, Sequence[float]] = 0.5,
                 reference: Optional[str] = None) -> pd.DataFrame:
    """Mean, standard deviation and count of *col* per altitude bin, over all times (and frames).

    Parameters
    ----------
    bins:
        Bin width in km (default 0.5) or the bin edges.
    reference:
        ``"agl"`` / ``"asl"`` to express altitudes in one reference before binning (each frame is converted).

    Returns
    -------
    DataFrame
        Indexed by the bin centre (km, column ``alt``) with ``mean``, ``std``, ``count``.
    """
    frames = [data] if isinstance(data, AtmozFrame) else list(data.values()) if isinstance(data, AtmozDataset) else None
    if frames is None:
        raise TypeError(f"expected an AtmozFrame or AtmozDataset, got {type(data).__name__}.")
    parts = []
    for f in frames:
        f = _in_reference(f, reference)
        if col not in f.df.columns:
            raise KeyError(f"Column {col!r} is not in the frame.")
        parts.append(pd.DataFrame({"alt": f.df[f.axes["alt"]].to_numpy(), "v": f.df[col].to_numpy()}))
    both = pd.concat(parts, ignore_index=True).dropna()
    if np.isscalar(bins):
        width = float(bins)
        edges = np.arange(np.floor(both["alt"].min() / width) * width, both["alt"].max() + width, width) if len(both) else np.array([0.0, width])
    else:
        edges = np.asarray(bins, dtype=float)
    cats = pd.cut(both["alt"], edges, right=False)
    res = both.groupby(cats, observed=True)["v"].agg(["mean", "std", "count"])
    res.index = pd.Index([iv.mid for iv in res.index], name="alt")
    res.columns.name = None
    return res
