"""Collocate several sources onto one: same times, same altitudes, same units.

>>> out = atmoz.collocate([lidar_frame, model_frame, sonde_frame])          # the first source is the anchor
>>> out = atmoz.collocate(tolnet_ds, geos_cf_ds, aqs_ds)                      # datasets pair by nearest site
>>> out.df[["alt", "obs", "model", "src3"]]

The **anchor** (the first source) defines the grid. Every other source is brought onto the anchor's rows:

* **time**: a source made of profiles at discrete times (a model, a lidar) is interpolated linearly between its two nearest
  times, and gives NaN when the anchor time is farther than *max_gap* from either. A source that is one continuous
  trajectory (a sonde: every row has its own time) is averaged over the rows within *window* of the anchor time.
* **altitude** (profiles): each other profile is brought onto the anchor's levels with *method*: ``"interp"`` (linear in
  altitude), ``"mean"`` (average of the points falling in each anchor level's bin) or ``"auto"`` (default: ``mean`` when the
  other profile is finer than the anchor, ``interp`` when it is coarser). NaN outside the other profile's altitude range.
* **altitude reference**: profile altitudes must say what they are measured from (``frame.alt_reference``: ``"asl"`` or
  ``"agl"``). When they differ the asl side is converted to above ground with its ``site_elevation_km``; missing
  information raises an error instead of guessing.
* **units**: every column is converted to the anchor's unit when both are declared and differ.
* **surface and profile together**: if any source is a surface series (no ``alt``), the comparison is made at the surface:
  profiles are first reduced to their lowest valid level at each time, or to *alt* km.
* **datasets**: frames are paired by location, the nearest frame of each other dataset within *max_distance_km*. The result is an
  :class:`AtmozDataset` keyed by the anchor's keys; ``metadata["pairs"]`` records the pairing and ``metadata["unpaired"]``
  the anchor frames that found no partner.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozDataset import AtmozDataset
from atmoz.dataclasses.AtmozFrame import AtmozFrame

Source = Union[AtmozFrame, AtmozDataset]


# ---- profiles and trajectories ------------------------------------------------------------------------------------------


def _is_profile_set(af: AtmozFrame) -> bool:
    """True when the frame holds profiles at discrete times (a timestamp repeats once per level), false for a trajectory."""
    if af.axes["alt"] is None or len(af.df) == 0:
        return False
    return float(af.df.groupby(level=0).size().median()) > 1


def _bins(levels: np.ndarray) -> np.ndarray:
    """Bin edges around sorted *levels* (midpoints, extended half a step at each end)."""
    first = levels[0] - (levels[1] - levels[0]) / 2
    last = levels[-1] + (levels[-1] - levels[-2]) / 2
    return np.concatenate([[first], (levels[:-1] + levels[1:]) / 2, [last]])


def _to_levels(src_alt: np.ndarray, src_val: np.ndarray, at: np.ndarray, method: str) -> np.ndarray:
    """One source profile (alt, value) -> values at the levels *at*; NaN where it has nothing."""
    ok = np.isfinite(src_alt) & np.isfinite(src_val)
    out = np.full(len(at), np.nan)
    if not ok.any() or len(at) == 0:
        return out
    a, v = src_alt[ok], src_val[ok]
    if method == "auto":
        at_step = np.median(np.diff(np.sort(at))) if len(at) > 1 else np.inf
        src_step = np.median(np.diff(np.sort(a))) if len(a) > 1 else np.inf
        method = "mean" if len(at) > 1 and src_step < 0.75 * at_step else "interp"
    if method == "mean" and len(at) > 1:
        order = np.argsort(at)
        edges = _bins(at[order])
        idx = np.digitize(a, edges) - 1
        inside = (idx >= 0) & (idx < len(at))
        sums = np.bincount(idx[inside], weights=v[inside], minlength=len(at))
        counts = np.bincount(idx[inside], minlength=len(at))
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(counts > 0, sums / counts, np.nan)
        out[order] = res
        return out
    if len(a) < 2:
        return out
    order = np.argsort(a)
    return np.interp(at, a[order], v[order], left=np.nan, right=np.nan)


def _surface_of(af: AtmozFrame, col: str, alt: Optional[float]) -> AtmozFrame:
    """A profile frame reduced to a surface series: the lowest valid level (or the value at *alt* km) at each time.

    A trajectory (a sonde) is one profile: it reduces to a single row at the time of that level.
    """
    altname = af.axes["alt"]
    df = af.df
    rows = []
    if _is_profile_set(af):
        groups = list(df.groupby(level=0, sort=True))
    else:
        groups = [(None, df)]
    for t, g in groups:
        a, v = g[altname].to_numpy(dtype=float), g[col].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(v)
        if alt is None:
            if not ok.any():
                rows.append((t if t is not None else g.index[0], np.nan, g))
                continue
            i = np.flatnonzero(ok)[np.argmin(a[ok])]
            rows.append((t if t is not None else g.index[i], v[i], g))
        else:
            val = _to_levels(a, v, np.array([alt]), "interp")[0]
            when = t if t is not None else g.index[np.argmin(np.abs(np.where(ok, a, np.inf) - alt))]
            rows.append((when, val, g))
    out = pd.DataFrame({col: [r[1] for r in rows]}, index=pd.DatetimeIndex([r[0] for r in rows], name=df.index.name))
    for key in ("lat", "lon"):
        name = af.axes[key]
        if name is not None:
            out[name] = [float(r[2][name].median()) for r in rows]
    units = {k: v for k, v in af.units.items() if k in out.columns}
    return AtmozFrame(out, units=units, lat=af.axes["lat"], lon=af.axes["lon"],
                      metadata={**af.metadata, "reduced_to_surface": "lowest level" if alt is None else f"{alt} km"})


def _common_reference(frames: List[AtmozFrame], names: List[str]) -> List[AtmozFrame]:
    """Express every profile's altitude in the same reference (above ground when they differ), or explain what is missing."""
    refs = [f.alt_reference for f in frames]
    if any(r is None for r in refs):
        missing = ", ".join(n for n, r in zip(names, refs) if r is None)
        raise ValueError(f"The altitude reference of {missing} is not stated, so the profiles cannot be aligned safely. "
                         "Set it with frame.with_metadata(alt_reference='asl' or 'agl', site_elevation_km=...), or pass "
                         "assume_same_reference=True if you know they are measured from the same level.")
    if len(set(refs)) == 1:
        return frames
    return [f.with_alt_reference("agl") for f in frames]


# ---- one source onto the anchor's rows ----------------------------------------------------------------------------------


def _series_onto(times: pd.DatetimeIndex, other: pd.Series, gap: pd.Timedelta) -> np.ndarray:
    """Linear time interpolation of a surface series onto *times*; NaN beyond *gap* of either neighbour."""
    s = other.groupby(level=0).mean().sort_index()
    mt = s.index.as_unit("ns").asi8
    mv = s.to_numpy(dtype=float)
    at = times.as_unit("ns").asi8
    out = np.full(len(at), np.nan)
    if len(mt) == 0:
        return out
    k = np.searchsorted(mt, at)
    exact = (k < len(mt)) & (mt[np.minimum(k, len(mt) - 1)] == at)
    out[exact] = mv[k[exact]]
    inner = ~exact & (k > 0) & (k < len(mt))
    lo, hi = k[inner] - 1, k[inner]
    t_lo, t_hi, t = mt[lo], mt[hi], at[inner]
    near = (t - t_lo <= gap.value) & (t_hi - t <= gap.value)
    w = (t - t_lo) / (t_hi - t_lo)
    vals = (1 - w) * mv[lo] + w * mv[hi]
    vals[~near] = np.nan
    out[inner] = vals
    return out


def _profiles_onto(anchor: AtmozFrame, other: AtmozFrame, col: str, gap: pd.Timedelta, window: pd.Timedelta,
                   method: str) -> np.ndarray:
    adf, odf = anchor.df, other.df
    aalt = adf[anchor.axes["alt"]].to_numpy(dtype=float)
    oalt = odf[other.axes["alt"]].to_numpy(dtype=float)
    oval = odf[col].to_numpy(dtype=float)
    result = np.full(len(adf), np.nan)
    positions = pd.Series(np.arange(len(adf)), index=adf.index)
    discrete = _is_profile_set(other)
    if discrete:
        o_times = pd.DatetimeIndex(odf.index.unique()).sort_values()
        mt = o_times.as_unit("ns").asi8
        rows_at = {t: np.flatnonzero(odf.index == t) for t in o_times}
    else:
        o_ns = odf.index.as_unit("ns").asi8
        order = np.argsort(o_ns, kind="stable")
        o_ns, oalt, oval = o_ns[order], oalt[order], oval[order]

    def profile(i: int, levels: np.ndarray) -> np.ndarray:
        r = rows_at[o_times[i]]
        return _to_levels(oalt[r], oval[r], levels, method)

    for t, pos in positions.groupby(level=0):
        rows = pos.to_numpy()
        levels = aalt[rows]
        tv = t.as_unit("ns").value
        if not discrete:
            lo = np.searchsorted(o_ns, tv - window.value, side="left")
            hi = np.searchsorted(o_ns, tv + window.value, side="right")
            if hi > lo:
                result[rows] = _to_levels(oalt[lo:hi], oval[lo:hi], levels, "mean" if method == "auto" else method)
            continue
        k = np.searchsorted(mt, tv)
        if k < len(mt) and mt[k] == tv:
            result[rows] = profile(k, levels)
        elif 0 < k < len(mt) and tv - mt[k - 1] <= gap.value and mt[k] - tv <= gap.value:
            w = (tv - mt[k - 1]) / (mt[k] - mt[k - 1])
            result[rows] = (1 - w) * profile(k - 1, levels) + w * profile(k, levels)
    return result


# ---- frames ------------------------------------------------------------------------------------------------------------


def _names_for(sources: Sequence[Source], names: Optional[Sequence[str]]) -> List[str]:
    n = len(sources)
    if names is not None:
        if len(names) != n or len(set(names)) != n:
            raise ValueError(f"names must be {n} distinct strings, got {list(names)!r}.")
        return list(names)
    return [("obs", "model")[i] if i < 2 else f"src{i + 1}" for i in range(n)]


def _cols_for(cols: Union[str, Sequence[str], Dict[str, str]], names: List[str]) -> List[str]:
    if isinstance(cols, str):
        return [cols] * len(names)
    if isinstance(cols, dict):
        return [cols.get(n, "value") for n in names]
    if len(cols) != len(names):
        raise ValueError("cols must be one name or one per source.")
    return list(cols)


def _collocate_frames(frames: List[AtmozFrame], names: List[str], cols: List[str], *, max_gap, window, method, alt,
                      assume_same_reference) -> AtmozFrame:
    for n, af, c in zip(names, frames, cols):
        if not isinstance(af, AtmozFrame):
            raise TypeError(f"{n} must be an AtmozFrame or AtmozDataset, got {type(af).__name__}.")
        if c not in af.df.columns:
            raise KeyError(f"Column {c!r} of {n} is not in the frame (columns: {list(af.df.columns)}).")
    gap, win = pd.Timedelta(max_gap), pd.Timedelta(window)
    has_alt = [f.axes["alt"] is not None for f in frames]
    if any(has_alt) and not all(has_alt):
        frames = [_surface_of(f, c, alt) if h else f for f, c, h in zip(frames, cols, has_alt)]
    elif all(has_alt) and not assume_same_reference:
        frames = _common_reference(frames, names)
    profiles = frames[0].axes["alt"] is not None
    anchor = frames[0]
    base_unit = anchor.units.get(cols[0])
    out: Dict[str, np.ndarray] = {}
    units: Dict[str, str] = {}
    out[names[0]] = anchor.df[cols[0]].to_numpy(dtype=float)
    if base_unit:
        units[names[0]] = base_unit
    for n, af, c in zip(names[1:], frames[1:], cols[1:]):
        other = af
        unit = af.units.get(c)
        if base_unit and unit and unit != base_unit:
            other = af.convert_units(c, base_unit)
            unit = base_unit
        if profiles:
            out[n] = _profiles_onto(anchor, other, c, gap, win, method)
        else:
            out[n] = _series_onto(anchor.df.index, other.df[c], gap)
        if unit:
            units[n] = unit
    data = {}
    for key in ("alt", "lat", "lon"):
        src = anchor.axes[key]
        if src is not None:
            data[key] = anchor.df[src].to_numpy()
            if src in anchor.units:
                units[key] = anchor.units[src]
    data.update(out)
    ref = anchor.alt_reference if profiles else None
    meta = {**({"alt_reference": ref} if ref else {}),
            **({"site_elevation_km": anchor.metadata["site_elevation_km"]}
               if ref and anchor.metadata.get("site_elevation_km") is not None else {}),
            "sources": {n: dict(f.metadata) for n, f in zip(names, frames)},
            "columns": dict(zip(names, cols)), "max_gap": str(gap), "window": str(win), "method": method}
    return AtmozFrame(pd.DataFrame(data, index=anchor.df.index), units=units, metadata=meta,
                      **{k: k for k in ("lat", "lon", "alt") if k in data})


# ---- datasets ----------------------------------------------------------------------------------------------------------


def _distance_km(a: tuple, b: tuple) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(h)))


def _overlap(a: AtmozFrame, b: AtmozFrame) -> float:
    if len(a.df) == 0 or len(b.df) == 0:
        return 0.0
    lo, hi = max(a.df.index.min(), b.df.index.min()), min(a.df.index.max(), b.df.index.max())
    return max((hi - lo).total_seconds(), 0.0)


def _nearest(anchor: AtmozFrame, ds: AtmozDataset, max_km: float, tie_km: float = 1.0) -> Optional[str]:
    """The key of the frame of *ds* nearest to *anchor*; among near-ties (same site) the one overlapping it most in time."""
    here = anchor.location
    if here is None:
        raise ValueError("Pairing datasets needs the anchor's lat/lon (set lat= and lon= axes on its frames).")
    cands = []
    for key, f in ds.items():
        loc = f.location
        if loc is not None:
            d = _distance_km(here, loc)
            if d <= max_km:
                cands.append((d, key, f))
    if not cands:
        return None
    nearest = min(c[0] for c in cands)
    close = [c for c in cands if c[0] <= nearest + tie_km]
    return max(close, key=lambda c: (_overlap(anchor, c[2]), -c[0]))[1]


def collocate(
    sources: Union[Source, Sequence[Source]],
    *more: Source,
    names: Optional[Sequence[str]] = None,
    cols: Union[str, Sequence[str], Dict[str, str]] = "value",
    max_gap: Union[str, pd.Timedelta] = "1h",
    window: Union[str, pd.Timedelta] = "1h",
    method: str = "auto",
    alt: Optional[float] = None,
    max_distance_km: float = 25.0,
    assume_same_reference: bool = False,
) -> Union[AtmozFrame, AtmozDataset]:
    """Bring several sources onto the first one (the anchor): its times, altitudes and unit.

    Parameters
    ----------
    sources, *more:
        A list of :class:`AtmozFrame` / :class:`AtmozDataset`, or the same passed as separate arguments. The first is the
        anchor. At least two.
    names:
        Output column names, one per source (default ``obs``, ``model``, ``src3``, ...).
    cols:
        The column compared in every source (default ``"value"``): one name for all, a list, or ``{name: column}``.
    max_gap:
        Largest distance to the nearest time of a discrete-time source (a model, a lidar) that still gives a value.
    window:
        Half-width of the time window averaged for a trajectory source (a sonde).
    method:
        Vertical method for profiles: ``"auto"`` (default), ``"interp"`` or ``"mean"``.
    alt:
        When surface and profile sources are mixed, the altitude (km, in the profile's own reference) at which profiles are
        evaluated. Default: the lowest valid level at each time.
    max_distance_km:
        Datasets only: the farthest a frame may be from the anchor frame and still be paired with it.
    assume_same_reference:
        Skip the altitude-reference check and treat all altitudes as measured from the same level.

    Returns
    -------
    AtmozFrame or AtmozDataset
        On the anchor's rows: its ``alt``/``lat``/``lon`` axes and one column per source (anchor first), UTC ``time`` index,
        ``units`` and ``metadata`` (the sources' metadata, the settings, ``alt_reference``). An :class:`AtmozDataset`, keyed by
        the anchor's keys, when the anchor is a dataset.
    """
    srcs: List[Source] = list(sources) if isinstance(sources, (list, tuple)) else [sources]
    srcs += list(more)
    if len(srcs) < 2:
        raise ValueError("collocate needs at least two sources.")
    if method not in ("auto", "interp", "mean"):
        raise ValueError("method must be 'auto', 'interp' or 'mean'.")
    nm = _names_for(srcs, names)
    cl = _cols_for(cols, nm)
    opts = dict(max_gap=max_gap, window=window, method=method, alt=alt, assume_same_reference=assume_same_reference)
    for s, n in zip(srcs, nm):
        if not isinstance(s, (AtmozFrame, AtmozDataset)):
            raise TypeError(f"{n} must be an AtmozFrame or AtmozDataset, got {type(s).__name__}.")
    if not any(isinstance(s, AtmozDataset) for s in srcs):
        return _collocate_frames(srcs, nm, cl, **opts)

    anchor = srcs[0]
    anchor_items = list(anchor.items()) if isinstance(anchor, AtmozDataset) else [(None, anchor)]
    frames: Dict[str, AtmozFrame] = {}
    pairs: Dict[str, Dict[str, str]] = {}
    unpaired: Dict[str, List[str]] = {}
    for akey, aframe in anchor_items:
        chosen, picked, missing = [aframe], {}, []
        for n, s in zip(nm[1:], srcs[1:]):
            if isinstance(s, AtmozDataset):
                key = _nearest(aframe, s, max_distance_km)
                if key is None:
                    missing.append(n)
                    continue
                chosen.append(s[key])
                picked[n] = key
            else:
                chosen.append(s)
        if missing:
            unpaired[str(akey)] = missing
            continue
        used = [nm[0]] + [n for n in nm[1:] if n not in missing]
        frames[str(akey)] = _collocate_frames(chosen, used, [cl[nm.index(n)] for n in used], **opts)
        pairs[str(akey)] = picked
    if not isinstance(anchor, AtmozDataset):
        if not frames:
            raise ValueError(f"No frame within {max_distance_km} km of the anchor for: {unpaired.get('None')}.")
        return frames["None"]
    meta = {"names": nm, "pairs": pairs, "unpaired": unpaired, "max_distance_km": max_distance_km,
            "sources": [getattr(s, "source", None) for s in srcs]}
    return AtmozDataset(name="collocated", source="atmoz.collocate", frames=frames, metadata=meta)
