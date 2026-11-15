"""Vertical profile plot."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from atmoz.plotting.core import styled
from atmoz.resources import plot_utilities


@styled("profile")
def profile(
    profiles: Mapping[str, Mapping],
    *,
    xlabel: str = "Ozone mixing ratio (ppbv)",
    ylim: Optional[tuple] = (0, 5),
    title: Optional[str] = None,
    ax=None,
    style=None,
    figsize=None,
):
    """Vertical profiles, quantity on x and altitude on y.

    Parameters
    ----------
    profiles : dict
        ``{label: {"X": values, "Y": altitude_km, "uncertainty": optional, "params": optional line kwargs}}``.
    xlabel : str
    ylim : (low, high), optional
        Altitude limits in km. ``None`` leaves matplotlib's automatic limits.
    title : str, optional
    ax : matplotlib Axes, optional
    style, figsize
        See :mod:`atmoz.plotting`.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be a dict: {label: {'X': ..., 'Y': ...}}")

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    for label, prof in profiles.items():
        if "X" not in prof or "Y" not in prof:
            raise KeyError(f"profile {label!r} needs 'X' (values) and 'Y' (altitude) keys")
        x = np.asarray(prof["X"], dtype=float)
        y = np.asarray(prof["Y"], dtype=float)
        (line,) = ax.plot(x, y, label=str(label), **dict(prof.get("params") or {}))
        unc = prof.get("uncertainty")
        if unc is not None:
            unc = np.asarray(unc, dtype=float)
            ax.fill_betweenx(y, x - unc, x + unc, color=line.get_color(), alpha=0.18, linewidth=0)

    plot_utilities.apply_plot_params(fig, ax, **{k: v for k, v in style.params.items() if k.startswith("ax.")})
    ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    if len(profiles) > 1 or any("params" in p and "label" in (p["params"] or {}) for p in profiles.values()):
        ax.legend(loc="upper left")
    return fig, ax
