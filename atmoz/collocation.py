"""Collocate model output onto observations: same times, same altitudes, same units.

>>> out = atmoz.collocate(tolnet_frame, geos_cf_frame)       # AtmozFrame with columns obs, model
>>> out.df[["alt", "obs", "model"]]

* **time**: for every observation time the model is interpolated linearly between its two nearest times; observations farther
  than *max_gap* from either neighbour get NaN.
* **altitude** (profiles): each model profile is interpolated onto the observation altitudes using the model's own ``alt``
  column (which may differ per time, as GEOS-CF layer heights do). Outside the model's altitude range the result is NaN.
* **units**: the model column is converted to the observation unit when both are declared and differ.
* Both frames must be profiles (have ``alt``) or both surface series (no ``alt``); to compare a surface site with a model
  profile, select the model's lowest level first.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozFrame import AtmozFrame


def _profile_at(alts: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Interpolate one profile (alts, values) onto *at*; NaN outside the profile's range and where values are NaN."""
    ok = np.isfinite(alts) & np.isfinite(values)
    if ok.sum() < 2:
        return np.full(len(at), np.nan)
    order = np.argsort(alts[ok])
    return np.interp(at, alts[ok][order], values[ok][order], left=np.nan, right=np.nan)


def collocate(
    obs: AtmozFrame,
    model: AtmozFrame,
    *,
    obs_col: str = "value",
    model_col: str = "value",
    max_gap: Union[str, pd.Timedelta] = "1h",
) -> AtmozFrame:
    """Interpolate *model* onto the times (and altitudes) of *obs*.

    Parameters
    ----------
    obs, model:
        :class:`AtmozFrame` objects, both profiles (an ``alt`` axis) or both surface series.
    obs_col, model_col:
        The columns to compare.
    max_gap:
        Largest distance to the nearest model time (on either side) that still gives a value.

    Returns
    -------
    AtmozFrame
        On the observation rows: ``obs`` and ``model`` columns (plus ``alt``, ``lat``, ``lon`` from the observations),
        UTC ``time`` index, ``units`` for ``obs`` and ``model`` (the observation unit), and ``metadata`` with the inputs' metadata.
    """
    for name, af in (("obs", obs), ("model", model)):
        if not isinstance(af, AtmozFrame):
            raise TypeError(f"{name} must be an AtmozFrame, got {type(af).__name__}.")
    if (obs.axes["alt"] is None) != (model.axes["alt"] is None):
        raise ValueError("Both frames must have an altitude axis (profiles) or neither (surface series). "
                         "To compare a surface site with a model profile, select the model's lowest level first.")
    for af, col in ((obs, obs_col), (model, model_col)):
        if col not in af.df.columns:
            raise KeyError(f"Column {col!r} is not in the frame (columns: {list(af.df.columns)}).")
    gap = pd.Timedelta(max_gap)
    profiles = obs.axes["alt"] is not None

    unit_obs, unit_model = obs.units.get(obs_col), model.units.get(model_col)
    mdf = model.df
    if unit_obs and unit_model and unit_obs != unit_model:
        mdf = model.convert_units(model_col, unit_obs).df

    obs_df = obs.df
    result = np.full(len(obs_df), np.nan)
    model_times = pd.DatetimeIndex(mdf.index.unique()).sort_values()
    mt = model_times.as_unit("ns").asi8
    groups = {t: mdf.loc[t] for t in model_times}      # one slice per model time

    def column_at(t_idx: int, alts: Optional[np.ndarray], n: int) -> np.ndarray:
        g = groups[model_times[t_idx]]
        if not profiles:
            vals = g[model_col].to_numpy(dtype=float) if isinstance(g, pd.DataFrame) else np.array([g[model_col]], dtype=float)
            return np.full(n, np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
        return _profile_at(g[model.axes["alt"]].to_numpy(dtype=float), g[model_col].to_numpy(dtype=float), alts)

    obs_times = obs_df.index
    positions = pd.Series(np.arange(len(obs_df)), index=obs_times)
    for t, pos in positions.groupby(level=0):
        rows = pos.to_numpy()
        alts = obs_df[obs.axes["alt"]].to_numpy(dtype=float)[rows] if profiles else None
        k = np.searchsorted(mt, t.value)
        if k < len(mt) and mt[k] == t.value:
            result[rows] = column_at(k, alts, len(rows))
            continue
        if k == 0 or k == len(mt):
            continue
        before, after = model_times[k - 1], model_times[k]
        if t - before > gap or after - t > gap:
            continue
        w = (t - before) / (after - before)
        result[rows] = (1 - w) * column_at(k - 1, alts, len(rows)) + w * column_at(k, alts, len(rows))

    cols = {}
    for key, name in (("alt", obs.axes["alt"]), ("lat", obs.axes["lat"]), ("lon", obs.axes["lon"])):
        if name is not None:
            cols[key] = obs_df[name].to_numpy()
    cols["obs"] = obs_df[obs_col].to_numpy(dtype=float)
    cols["model"] = result
    out = pd.DataFrame(cols, index=obs_df.index)
    units = {"obs": unit_obs, "model": unit_obs or unit_model}
    for key, src in (("alt", obs.axes["alt"]), ("lat", obs.axes["lat"]), ("lon", obs.axes["lon"])):
        if src is not None and src in obs.units:
            units[key] = obs.units[src]
    return AtmozFrame(out, units={k: v for k, v in units.items() if v}, metadata={
        "obs": dict(obs.metadata), "model": dict(model.metadata), "max_gap": str(gap),
        "obs_col": obs_col, "model_col": model_col},
        **{k: k for k in ("lat", "lon", "alt") if k in cols})
