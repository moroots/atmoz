"""Interactive site map (plotly)."""

from __future__ import annotations

from itertools import cycle
from typing import Mapping, Optional, Sequence

import numpy as np

from atmoz.plotting.core import styled
from atmoz.plotting.styles import PALETTE


@styled("sitemap", backend="plotly")
def site_map(
    sites: Mapping,
    *,
    title: str = "Site map",
    hover_cols: Optional[Sequence[str]] = None,
    zoom: Optional[float] = None,
    colors: Sequence[str] = PALETTE,
    style=None,
    figsize=None,
):
    """Interactive map of monitoring sites.

    Parameters
    ----------
    sites : dict
        ``{name: GeoDataFrame}`` with point geometry (one trace per name).
    title : str
    hover_cols : list of str, optional
        Columns shown in the hover tooltip.
    zoom : float, optional
        Map zoom. By default the view is fitted to the points.
    colors : sequence of str
        Colour cycle for the traces.
    style
        See :mod:`atmoz.plotting`. ``figsize`` is not used by plotly plots; use ``style.plotly['height']``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    if not isinstance(sites, Mapping):
        raise TypeError("sites must be a dict of {name: GeoDataFrame}")
    opts = dict(style.plotly)
    palette = cycle(colors)

    fig = go.Figure()
    lats, lons = [], []
    for name, gdf in sites.items():
        if not hasattr(gdf, "geometry"):
            raise TypeError(f"sites[{name!r}] must be a GeoDataFrame with point geometry")
        lat, lon = gdf.geometry.y.to_numpy(), gdf.geometry.x.to_numpy()
        lats.append(lat)
        lons.append(lon)
        extra = {}
        if hover_cols:
            extra = dict(
                customdata=gdf[list(hover_cols)].to_numpy(),
                hovertemplate="<br>".join(f"<b>{c}</b>: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols))
                + "<extra>" + str(name) + "</extra>",
            )
        fig.add_trace(go.Scattermap(
            lat=lat, lon=lon, name=str(name), mode="markers",
            marker=dict(size=opts.get("marker_size", 11), color=next(palette)), **extra,
        ))

    all_lat, all_lon = np.concatenate(lats), np.concatenate(lons)
    if zoom is None:
        span = max(np.ptp(all_lat), np.ptp(all_lon) * 0.8, 0.05)
        zoom = float(np.clip(np.log2(360.0 / span) - 1.2, 1, 12))

    layout = dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=opts.get("title_font_size", 20))),
        map=dict(style=opts.get("map_style", "carto-positron"), zoom=zoom,
                 center=dict(lat=float(all_lat.mean()), lon=float(all_lon.mean()))),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=opts.get("legend", {}),
        font=opts.get("font", {}),
        showlegend=True,
    )
    if opts.get("height"):
        layout["height"] = opts["height"]
    if opts.get("width"):
        layout["width"] = opts["width"]
    fig.update_layout(**layout)
    return fig
