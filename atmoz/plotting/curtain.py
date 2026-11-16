"""Time-altitude curtain plot."""

from __future__ import annotations

from typing import Mapping, Optional, Union

import pandas as pd

from atmoz.dataclasses.AtmozFrame import AtmozFrame
from atmoz.plotting.core import styled
from atmoz.resources import colorbars, plot_utilities

# variable -> label, colour scale (cmap, norm) and a short title. Add an entry to teach `curtain` a new variable.
_VARIABLES = {
    "ozone": {
        "label": "Ozone mixing ratio (ppb)",
        "scale": colorbars.tolnet_ozone,
        "title": "Ozone mixing ratio profile",
    },
}


def _wide(frame: AtmozFrame) -> pd.DataFrame:
    """A long profile AtmozFrame (``alt`` and ``value`` columns, time index) -> time index x altitude columns."""
    if frame.axes["alt"] is None or "value" not in frame.df.columns:
        raise ValueError("curtain needs a profile AtmozFrame with an alt axis and a 'value' column.")
    tmp = pd.DataFrame({"t": frame.df.index, "alt": frame.df[frame.axes["alt"]].to_numpy(),
                        "value": frame.df["value"].to_numpy()})
    wide = tmp.pivot_table(index="t", columns="alt", values="value", aggfunc="mean")
    wide.index.name = None
    wide.columns.name = None
    return wide


def _frames(data) -> list[pd.DataFrame]:
    """One profile (AtmozFrame, or a wide DataFrame: time index, altitude columns) or a dict of them."""
    if isinstance(data, AtmozFrame):
        return [_wide(data)]
    if isinstance(data, pd.DataFrame):
        return [data]
    if isinstance(data, Mapping):
        frames = [_wide(v) if isinstance(v, AtmozFrame) else v for _, v in sorted(data.items(), key=lambda kv: str(kv[0]))]
        if all(isinstance(f, pd.DataFrame) for f in frames):
            return frames
    raise TypeError("curtain needs an AtmozFrame, a DataFrame (time index, altitude columns) or a dict of them")


@styled("curtain")
def curtain(
    data: Union[pd.DataFrame, Mapping],
    *,
    variable: Optional[str] = "ozone",
    altitude_units: str = "km",
    title: Optional[str] = None,
    tz: str = "UTC",
    sonde: Optional[Mapping] = None,
    ax=None,
    nrt: bool = False,
    watermark: Optional[str] = None,
    style=None,
    figsize=None,
):
    """Time-altitude curtain of a profiled quantity.

    Parameters
    ----------
    data : DataFrame or dict of DataFrames
        Index = time, columns = altitude, values = the quantity. A long profile :class:`~atmoz.AtmozFrame` is pivoted for you. A dict plots each
        frame side by side, in key order.
    variable : {"ozone", None}
        Picks colour scale and labels. ``None`` uses viridis with automatic limits.
    altitude_units : {"km", "m"}
        Units of the column labels. The axis is always km.
    title : str, optional
    tz : str
        Time zone of the time axis labels.
    sonde : dict, optional
        ``{"X": times, "Y": altitude km, "C": values}`` to overlay as coloured dots.
    ax : matplotlib Axes, optional
        Draw into this axes instead of making a figure. atmoz's rcParams style then does not apply to it.
    nrt : bool
        Stamp the plot "NRT DATA. NOT CITABLE." (near-real-time data).
    watermark : str, optional
        Path to a logo image added to the figure.
    style, figsize
        See :mod:`atmoz.plotting`.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    spec = _VARIABLES.get(variable) if variable else None
    if variable and spec is None:
        raise ValueError(f"Unknown variable {variable!r}. Available: {', '.join(_VARIABLES)} (or None).")
    cmap, norm = spec["scale"]() if spec else (plt.get_cmap("viridis"), None)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    scale = 1000.0 if altitude_units == "m" else 1.0
    im = None
    for frame in _frames(data):
        if frame.empty:
            continue
        altitude = frame.columns.astype(float) / scale
        im = ax.pcolormesh(frame.index, altitude, frame.to_numpy(dtype=float).T,
                           cmap=cmap, norm=norm, shading="nearest")

    if sonde:
        ax.scatter(sonde["X"], sonde["Y"], c=sonde["C"], s=60, cmap=cmap, norm=norm,
                   edgecolor="black", linewidth=0.6, zorder=3)

    if im is not None:
        opts = dict(style.params.get("colorbar", {}))
        ticks = opts.get("ticks") if norm is not None else None
        cbar = fig.colorbar(im, ax=ax, pad=opts.get("pad", 0.012), ticks=ticks)
        cbar.set_label(spec["label"] if spec else "", **opts.get("label", {}))
        cbar.ax.tick_params(labelsize=opts.get("labelsize", 9))

    plot_utilities.apply_datetime_axis(ax, tz=tz)
    plot_utilities.apply_plot_params(fig, ax, **{k: v for k, v in style.params.items() if k.startswith("ax.")})
    ax.set_xlabel("Time (" + str(tz) + ")")
    ax.set_title(title or (spec["title"] if spec else ""))
    ax.margins(x=0)

    if watermark:
        plot_utilities.apply_watermark(fig, watermark)
    if nrt:
        plot_utilities.apply_near_real_time(ax)
    return fig, ax
