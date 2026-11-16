"""Plots for comparing sources and summarising exceedances."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from atmoz.dataclasses.AtmozDataset import AtmozDataset
from atmoz.dataclasses.AtmozFrame import AtmozFrame
from atmoz.plotting.core import styled
from atmoz.plotting.profile import profile


def _frames(data) -> list:
    if isinstance(data, AtmozFrame):
        return [data]
    if isinstance(data, AtmozDataset):
        return list(data.values())
    raise TypeError(f"expected an AtmozFrame or AtmozDataset, got {type(data).__name__}.")


@styled("profile")
def comparison_profile(
    data: Union[AtmozFrame, AtmozDataset],
    *,
    columns: Optional[Sequence[str]] = None,
    bins: Union[float, Sequence[float]] = 1.0,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
    ax=None,
    style=None,
    figsize=None,
):
    """Mean vertical profiles of every source in a collocated frame, with the spread (1 standard deviation) shaded.

    Parameters
    ----------
    data:
        The result of :func:`atmoz.collocate` for profiles (an AtmozFrame, or a dataset of them, pooled).
    columns:
        Which columns to draw. Default: the sources recorded in ``metadata["columns"]``, else ``obs`` and ``model``.
    bins:
        Altitude bin width in km (default 1) or the bin edges.
    xlabel, title, ylim, ax, style, figsize:
        As for :func:`atmoz.plot.profile`.

    Returns
    -------
    (fig, ax)
    """
    frames = _frames(data)
    if not frames or frames[0].axes["alt"] is None:
        raise ValueError("comparison_profile needs a collocated profile frame (with an alt axis).")
    first = frames[0]
    names = list(columns) if columns is not None else (list(first.metadata.get("columns", {})) or ["obs", "model"])
    parts = []
    for f in frames:
        missing = [c for c in names if c not in f.df.columns]
        if missing:
            raise KeyError(f"Columns {missing} are not in the frame (columns: {list(f.df.columns)}).")
        parts.append(f.df[[f.axes["alt"], *names]].rename(columns={f.axes["alt"]: "alt"}))
    both = pd.concat(parts, ignore_index=True)
    if ylim is not None:
        both = both[(both["alt"] >= ylim[0]) & (both["alt"] <= ylim[1])]      # keep the x scale to what is shown
    if np.isscalar(bins):
        width = float(bins)
        top = float(both["alt"].max()) if both["alt"].notna().any() else width
        edges = np.arange(0.0, top + width, width)
    else:
        edges = np.asarray(bins, dtype=float)
    cats = pd.cut(both["alt"], edges, right=False)
    profiles = {}
    for name in names:
        g = both.groupby(cats, observed=True)[name].agg(["mean", "std"]).dropna(subset=["mean"])
        profiles[name] = {"X": g["mean"].to_numpy(), "Y": np.array([iv.mid for iv in g.index]),
                          "uncertainty": g["std"].fillna(0.0).to_numpy()}
    unit = first.units.get("obs") or first.units.get(names[0]) or first.units.get("value")
    ref = first.alt_reference
    fig, ax = profile(profiles, xlabel=xlabel or (f"Mixing ratio ({unit})" if unit else "Value"), ylim=ylim, title=title,
                      ax=ax, style=style, figsize=figsize)
    ax.set_ylabel(f"Altitude (km {ref.upper()})" if ref else "Altitude (km)")
    ax.legend(frameon=False)
    return fig, ax


@styled("exceedances")
def exceedances(
    table: pd.DataFrame,
    *,
    level_ppb: Optional[float] = 70.0,
    top: Optional[int] = 25,
    title: Optional[str] = None,
    ax=None,
    style=None,
    figsize=None,
):
    """Days above the ozone standard per site, as bars (largest first).

    Parameters
    ----------
    table:
        The table returned by :func:`atmoz.stats.exceedances` (an ``exceedance_days`` column and site columns or ``key``).
    level_ppb:
        Level of the standard, used for the default title only (``None`` to omit).
    top:
        Show only the *top* sites with the most exceedance days (``None`` for all).

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if "exceedance_days" not in table.columns:
        raise KeyError("table needs an 'exceedance_days' column: pass the result of atmoz.stats.exceedances.")
    result_cols = {"valid_days", "days_total", "exceedance_days", "max_mda8", "max_date", "fourth_highest", "first", "last"}
    site_cols = [c for c in table.columns if c not in result_cols]
    labels = (table[site_cols].astype(str).agg("-".join, axis=1) if site_cols else pd.Series(["all"] * len(table), index=table.index))
    df = pd.DataFrame({"label": labels, "days": table["exceedance_days"]}).sort_values("days", ascending=False)
    if top is not None:
        df = df.head(top)
    fig, ax = plt.subplots()
    ax.bar(range(len(df)), df["days"].to_numpy())
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=60, ha="right")
    ax.set_ylabel("Days above the standard")
    ax.set_title(title if title is not None else (f"MDA8 ozone above {level_ppb:g} ppb" if level_ppb is not None else ""))
    return fig, ax
