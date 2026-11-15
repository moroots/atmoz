"""Wind rose (uses the ``windrose`` package)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from atmoz.plotting.core import styled


@styled("windrose")
def windrose(
    direction,
    speed,
    *,
    bins: Optional[Sequence[float]] = None,
    units: str = "m/s",
    title: Optional[str] = "Wind rose",
    cmap: str = "viridis",
    ax=None,
    style=None,
    figsize=None,
):
    """Wind rose of speed by direction.

    Parameters
    ----------
    direction : array-like
        Direction the wind blows *from*, degrees clockwise from north.
    speed : array-like
        Wind speed, in ``units``.
    bins : sequence, optional
        Speed bin edges. Defaults to five evenly spaced bins up to the maximum speed.
    units : str
        Used in the legend title.
    title : str, optional
    cmap : str
        Matplotlib colormap for the speed bins.
    ax : WindroseAxes, optional
        Draw into an existing ``windrose.WindroseAxes``.
    style, figsize
        See :mod:`atmoz.plotting`.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import windrose  # noqa: F401  (importing it registers the "windrose" projection)

    direction = np.asarray(direction, dtype=float)
    speed = np.asarray(speed, dtype=float)
    keep = ~(np.isnan(direction) | np.isnan(speed))
    direction, speed = direction[keep], speed[keep]
    if speed.size == 0:
        raise ValueError("no valid (direction, speed) pairs to plot")

    if bins is None:
        top = max(float(np.nanmax(speed)), 1.0)
        bins = np.linspace(0, top, 6)

    if ax is None:
        # WindroseAxes does not take part in constrained layout, so the figure uses fixed margins that leave
        # room for the title above and the legend to the right.
        fig = plt.figure(figsize=figsize or style.figsize, layout="none")
        ax = fig.add_axes([0.07, 0.06, 0.62, 0.78], projection="windrose")
    else:
        fig = ax.figure

    ax.bar(direction, speed, bins=bins, normed=True, opening=0.9, edgecolor="white", linewidth=0.5,
           cmap=plt.get_cmap(cmap))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_legend(title=f"Speed ({units})", loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)
    if title:
        ax.set_title(title, pad=18)
    return fig, ax
