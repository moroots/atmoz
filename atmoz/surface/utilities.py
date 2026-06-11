# -*- coding: utf-8 -*-
"""
Surface data utility functions for EPA AQS and AirNow workflows.
"""

from itertools import cycle
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from atmoz.resources import useful_functions
from atmoz.resources.timeConversions import format_resolution


_DEFAULT_COLORS = [
    "#CD6091",  # pink
    "#7195AB",  # steel blue
    "#6A4C93",  # purple
    "#4CC9F0",  # sky blue
    "#F4A261",  # orange
    "#2A9D8F",  # teal
    "#E76F51",  # burnt orange
    "#3A86FF",  # bright blue
    "#9B5DE5",  # violet
    "#8B3167",  # hot pink
]


def make_datetime(
    df: pd.DataFrame,
    time_cols: dict = {
        "date": "Date GMT",
        "time": "Time GMT",
        "format": {"date": "%Y-%m-%d ", "time": "%H:%M"},
    },
    **kwargs,
) -> pd.Series:
    """Combine separate date and time columns into a single datetime Series."""
    if not isinstance(time_cols, dict):
        raise TypeError("time_cols must be a dict")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return pd.to_datetime(
        df[time_cols["date"]] + " " + df[time_cols["time"]],
        format=time_cols["format"]["date"] + time_cols["format"]["time"],
    )


def get_resolution(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    col3: str,
    time_col: str = "Time GMT",
) -> pd.DataFrame:
    """Return a DataFrame showing the temporal resolution for each (col1, col2, col3) group."""
    group_cols = [col1, col2, col3]
    results = []
    for key, group in df.groupby(group_cols, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        if len(group) < 2:
            resolution = "unknown"
        else:
            delta = group[time_col].iloc[1] - group[time_col].iloc[0]
            resolution = format_resolution(delta)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["resolution"] = resolution
        results.append(row)
    return pd.DataFrame(results)


def get_stations(
    df: pd.DataFrame, lat_lon: List[str] = ["Latitude", "Longitude"]
) -> List[Tuple]:
    """Return a list of unique (latitude, longitude) tuples."""
    return df[lat_lon].drop_duplicates().apply(tuple, axis=1).tolist()


def extract_metadata(
    df: pd.DataFrame,
    data_columns: List[str] = [
        "Date Local",
        "Time Local",
        "Date GMT",
        "Time GMT",
        "Sample Measurement",
    ],
    ID_cols: List[str] = ["State Code", "County Code", "Site Num"],
    **kwargs,
) -> pd.DataFrame:
    """Drop measurement columns and return one row per unique monitoring station."""
    cols_to_drop = [c for c in data_columns if c in df.columns]
    return df.drop(columns=cols_to_drop).drop_duplicates(subset=ID_cols).reset_index(drop=True)


def to_gdf(
    df: pd.DataFrame,
    Latitude: str = "Latitude",
    Longitude: str = "Longitude",
    crs: str = "EPSG:4326",
    **kwargs,
) -> gpd.GeoDataFrame:
    """Convert a DataFrame with latitude/longitude columns to a GeoDataFrame."""
    geometry = gpd.points_from_xy(df[Longitude], df[Latitude])
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def count_duplicates(df: pd.DataFrame, cols: list) -> None:
    """Print any duplicate combinations found across all subsets of cols."""
    found = False
    for combo in [cols[i:j] for i in range(len(cols)) for j in range(i + 1, len(cols) + 1)]:
        n = df.duplicated(subset=combo).sum()
        if n > 0:
            print(f"{combo} -> {n} duplicates")
            found = True
    if not found:
        print("No Duplicates")


def site_map(
    instruments: Dict,
    title: str = "Site Map",
    colors: list = _DEFAULT_COLORS,
    **kwargs,
):
    """
    Render an interactive Plotly map of monitoring station locations.

    Parameters
    ----------
    instruments : dict[str, gpd.GeoDataFrame]
        Keys are trace names; values are GeoDataFrames with point geometry.
    title : str
        Map title.
    colors : list
        Cycle of hex colour strings for successive traces.
    **kwargs
        Override any entry in the default params dict.  Useful keys:
          hover_cols : list[str]  — columns to show in the hover tooltip.
          bbox       : dict       — {"lon": [west, east], "lat": [south, north]}
          map        : dict       — Plotly map sub-options (zoom, pitch, …)
          layout     : dict       — Plotly layout sub-options (height, width, …)
          Scattermap : dict       — marker / mode overrides.
    """
    if not isinstance(instruments, dict):
        raise TypeError("instruments must be a dict")

    color_cycle = cycle(colors)

    params = {
        "bbox": {"lon": [-125.0, -66.9], "lat": [24.4, 49.4]},
        "title": {
            "text": title,
            "x": 0.5, "y": 0.98,
            "xanchor": "center",
            "font": dict(size=24, color="black", family="Arial Black"),
        },
        "map": {"zoom": 3, "bearing": 0, "pitch": 20, "domain": dict(x=[0, 1], y=[0, 1])},
        "layout": {"height": 800, "width": 1600},
        "legend": dict(
            font=dict(size=18),
            itemsizing="constant",
            itemwidth=40,
            bgcolor="rgba(255,255,255,0.7)",
            x=0.01, y=0.99,
        ),
        "Scattermap": {"mode": "markers", "marker": dict(size=15)},
    }

    params = useful_functions.merge_dicts(params, kwargs)

    fig = go.Figure()

    for name, gdf in instruments.items():
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(f"instruments['{name}'] must be a GeoDataFrame")

        trace_kwargs = {}
        hover_cols = kwargs.get("hover_cols", None)
        if hover_cols is not None and isinstance(hover_cols, list):
            extra = "<extra></extra>" if not kwargs.get("show_trace_name", False) else ""
            trace_kwargs = dict(
                customdata=gdf[hover_cols].values,
                hovertemplate="<br>".join(
                    f"<b>{col}</b>: %{{customdata[{i}]}}"
                    for i, col in enumerate(hover_cols)
                ) + extra,
            )

        marker = dict(
            params["Scattermap"]["marker"],
            color=params["Scattermap"]["marker"].get("color", next(color_cycle)),
        )

        fig.add_trace(
            go.Scattermap(
                lat=gdf.geometry.y,
                lon=gdf.geometry.x,
                name=name,
                mode=params["Scattermap"]["mode"],
                marker=marker,
                **trace_kwargs,
            )
        )

    fig.update_layout(
        title=params["title"],
        map=dict(
            style="carto-positron",
            center=dict(
                lat=np.mean(params["bbox"]["lat"]),
                lon=np.mean(params["bbox"]["lon"]),
            ),
            **params["map"],
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=params["legend"],
        showlegend=True,
        paper_bgcolor="rgba(255,255,255,0.7)",
        **params["layout"],
    )

    fig.show()
    return


def slice_bbox(gdf: gpd.GeoDataFrame, bbox: dict) -> gpd.GeoDataFrame:
    """Return points within a lon/lat bounding box.  bbox = {"lon": [w, e], "lat": [s, n]}."""
    return (
        gdf.cx[bbox["lon"][0]:bbox["lon"][1], bbox["lat"][0]:bbox["lat"][1]]
        .reset_index(drop=True)
    )


def df_geo_slice(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    lat: str = "Latitude",
    lon: str = "Longitude",
) -> pd.DataFrame:
    """Filter a DataFrame to only rows whose (lat, lon) appear in gdf."""
    coords = set(zip(gdf[lat], gdf[lon]))
    mask = pd.Series(list(zip(df[lat], df[lon])), index=df.index).isin(coords)
    return df[mask]


def split_by_parameter(
    df: pd.DataFrame, parameter_col: str = "Parameter Name"
) -> Dict[str, pd.DataFrame]:
    """Return a dict keyed by parameter name."""
    return {param: df[df[parameter_col] == param] for param in df[parameter_col].unique()}


def split_by_station(
    df: pd.DataFrame,
    ID_cols: List[str] = ["State Code", "County Code", "Site Num"],
) -> Dict[tuple, pd.DataFrame]:
    """Return a dict keyed by (state_code, county_code, site_num) tuples."""
    return {coords: group for coords, group in df.groupby(ID_cols)}
