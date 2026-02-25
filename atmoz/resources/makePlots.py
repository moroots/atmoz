# -*- coding: utf-8 -*-
"""
Created on 2025-10-25 17:09:24

@author: Maurice Roots

Description:
     - Modules for storing plotting functions for specific types of plots
"""

import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import iplot

import matplotlib.pyplot as plt

from atmoz.resources import plot_utilities, useful_functions, colorbars, default_plot_params


vertical_profile_params = {
    "ax.set_ylim": [0, 5],

    "ax.legend": {
        "loc": "upper left",
        "fontsize": 12
        },

    "ax.set_ylabel": {
        "ylabel": "Altitude (km ASL)"
        },

    "ax.set_yticks": {
        "ticks": np.arange(0, 5.1, 1)
        },

    "ax.set_xlabel": {
        "xlabel": "Ozone Mixing Ratio (ppbv)"
        },

    "ax.set_title": {
        "label": "Vertical Profile (distance <10km)",
        "fontsize": 12
        },

    "ax.grid": {
        "visible": True,
        "color": "gray",
        "linestyle": "--",
        "linewidth": 0.5
        },
    "fig.layout": "tight",
    }

vertical_profile_theme = useful_functions.merge_dicts(default_plot_params.curtain_plot_theme, {
    "figure.figsize": (4, 6)
    })

def _align_heights(left_df, 
                   right_df, 
                   left_df_column="Altitude_km", 
                   right_df_column="Altitude_km", 
                   align="left"):
    
    left_df = left_df.copy()
    right_df = right_df.copy()

    match align:
        case "left":
            labels = left_df[left_df_column].values
            bins = np.append(labels, labels[-1] + (labels[-1] - labels[-2]))

            right_df[right_df_column] = pd.cut(right_df[right_df_column], 
                                               bins=bins, 
                                               labels=labels
                                               )
            
            return right_df.groupby(right_df_column, observed=False).mean()
        
        case "right":
            labels = right_df[right_df_column].values
            bins = np.append(labels, labels[-1] + (labels[-1] - labels[-2]))

            left_df[left_df_column] = pd.cut(left_df[left_df_column], 
                                             bins=bins, 
                                             labels=labels
                                             )
            
            return left_df.groupby(left_df_column, observed=False).mean()
    return None

def find_matching_key(key, keys, pattern):
    """
    Find a key in `keys` whose filename matches `pattern` and whose second element
    matches key[1], but do not return the key itself. Returns the first match found,
    or None if no match.
    """
    for x in keys:
        if x == key:
            continue  # skip the same key
        if re.match(pattern, x[0]) and x[1] == key[1]:
            return x
    return None

def _match_time(left_df: pd.DataFrame , right_df: pd.DataFrame, tolerance: str = "10 min", rounding: str = "10 min", align: str = "left"):
    if not (isinstance(left_df, pd.DataFrame) and isinstance(right_df, pd.DataFrame)):
        raise TypeError("Both left_df and right_df must be pandas DataFrames.")
    
    if not (isinstance(left_df.index, pd.DatetimeIndex) and isinstance(right_df.index, pd.DatetimeIndex)):
        raise TypeError("Both left_df and right_df must have a DatetimeIndex.")
    
    if not isinstance(rounding, str) and not isinstance(tolerance, str): 
        raise TypeError("Both 'rounding' and 'tolerance' must be strings.")
    
    if not isinstance(align, str): 
        raise TypeError(" 'align' must be a string as 'left' or 'right' ")
    
    tolerance = tolerance.split(" "); tolerance[0] = int(tolerance[0])
    if len(tolerance) != 2: 
        raise ValueError("'tolerance' must be 'int unit' ")
    
    match align:
        case "left":
            start = (left_df.index
                     .min()
                     .round(rounding) 
                     - pd.Timedelta(tolerance[0], unit=tolerance[1])
                    )
            
            end = (left_df.index 
                   .max()
                   .round(rounding) 
                   + pd.Timedelta(tolerance[0], unit=tolerance[1]) 
                   )

        case "right":
            start = (right_df.index
                     .min()
                     .round(rounding) 
                     - pd.Timedelta(tolerance[0], unit=tolerance[1])
                    )
            
            end = (right_df.index
                   .max()
                   .round(rounding) 
                   + pd.Timedelta(tolerance[0], unit=tolerance[1]) 
                   )
    return {
        "start": start,
        "end": end,
        "left_df": left_df[start:end],
        "right_df": right_df[start:end]
        }

def lidar_XYC(df, key): 
    match key:
        case "curtain": 
            return {
                "X": df.index,
                "Y": df.columns.astype(float) / 1000,
                "C": df.values.T * 1000
                }
        
        case "profile":
            return {
                "Y": df.index / 1000,
                "X": df.values
                }
    return None

def sonde_XYC(df, key): 
    match key:
        case "curtain": 
            return {
                "X": df.index.get_level_values('timestamp'),
                "Y": df["Altitude_km"],
                "C": df["Ozone_ppbv"]
                }
        
        case "profile":
            return {
                "Y": df["Altitude_km"],
                "X": df["Ozone_ppbv"]
                }
    return None

def vertical_profile(profiles: Dict,  
                     plot_params: Dict = {}, 
                     theme: Dict = {},
                     show=True
                     ):

    theme = useful_functions.merge_dicts(vertical_profile_theme, theme)
    plot_params = useful_functions.merge_dicts(vertical_profile_params, plot_params)

    if not isinstance( profiles, Dict ): 
        raise TypeError("'profiles' must be 'dict' ")

    with plt.rc_context(theme):
        fig, ax = plt.subplots()

        for key, profile in profiles.items():
            x = profile.get("X", None)
            y = profile.get("Y", None)

            if x is None or y is None:
                raise KeyError("Parameters of 'profile' in 'profiles' must be 'X', 'Y', and 'error' (optional)")
            ax.plot(x, y, **profile.get("params", None))
            
            uncertainty = profile.get("uncertainty", None)

            if uncertainty:
                ax.fill_betweenx(y, (x + uncertainty), (x - uncertainty), color='grey', alpha=0.3, label='Filled Area')

        plot_utilities.apply_plot_params(fig, ax, **plot_params)

        if show == True: 
            plt.show()
    return 

def plot_curtain(lidar, sonde=None, **kwargs):
    tz = kwargs.pop("tz", "UTC")
    params = useful_functions.merge_dicts(default_plot_params.tolnet_plot_params, kwargs)
    
    cmap, norm = colorbars.tolnet_ozone()

    if not isinstance(lidar["X"], list):
        lidar = {key: [lidar[key]] for key in lidar.keys()}
    
    with plt.rc_context(default_plot_params.curtain_plot_theme):
        fig, ax = plt.subplots()

        for X, Y, C in zip(lidar["X"], lidar["Y"], lidar["C"]):
            im = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm, shading="nearest", alpha=1)
        
        if sonde: 
            ax.scatter(
                x = sonde["X"],
                y = sonde["Y"],
                c = sonde["C"],
                s = 100,
                cmap = cmap, 
                norm = norm,
                **sonde.get("scatter_params", {})
                )

            vline_params = sonde.get("vline_params", {})
            skip = vline_params.pop("skip", 10) if vline_params and "skip" in vline_params else 10

            ax.vlines(sonde["X"][::skip]-pd.Timedelta(10, unit='min'), ymin=sonde["Y"][::skip] - 0.05, ymax=sonde["Y"][::skip] + 0.05,
                    **vline_params)

            ax.vlines(sonde["X"][::skip]+pd.Timedelta(10, unit='min'), ymin=sonde["Y"][::skip] - 0.05, ymax=sonde["Y"][::skip] + 0.05,
                    **vline_params)

        params["fig.colorbar"]["mappable"] = im

        plot_utilities.apply_datetime_axis(ax, tz=tz)
        
        plot_utilities.apply_plot_params(fig, ax, **params)

        plt.close()
    return 

from windrose import WindroseAxes
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_wind_rose(wind_direction, wind_speed, bar_params={}, plot_params={}):

    bar_params = useful_functions.merge_dicts({
        "bins": np.arange(0, 30, 5),
        "normed": True,
        "edgecolor": "black",
        "cmap": plt.cm.viridis,
        "alpha": 0.8
        }, bar_params)
    
    plot_params = useful_functions.merge_dicts({
        "ax.set_title": {
            "label": "Wind Rose",
            "fontsize": 16,
            "pad": 20
            },
        "ax.set_yticks": {
            "ticks": np.arange(0, 51, 5)
            },
        }, plot_params)
    
    fig = plt.figure(figsize=(7, 7))
    ax = WindroseAxes.from_ax(fig=fig)

    ax.bar(wind_direction, wind_speed, **bar_params)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plot_utilities.apply_plot_params(fig, ax, **plot_params)

    return
    
def site_map(instruments: Dict, 
             title: str = "Site Map", 
             **kwargs
             ): 
    
    if not isinstance( instruments, Dict ): 
        raise TypeError("'profiles' must be 'dict' ")
    
    params = {
        "bbox": {
            "lon": [-74.3, -71.8],
            "lat": [40.4, 41.3]
            },
        }
     
    fig = go.Figure()

    for name in instruments.keys():
        fig.add_trace(
            go.Scattermap( 
                lat=instruments[name].geometry.y,
                lon=instruments[name].geometry.x,
                mode='markers',
                marker=dict(size=10),
                name=name
                )
            )
        
    fig.update_layout(
        title=title,
        map=dict(
            style="carto-positron",
            center=dict(lat=np.mean(params["bbox"]["lat"]), lon=np.mean(params["bbox"]["lon"])),
            zoom=7,
            bearing=0,
            pitch=20,
            domain=dict(x=[0, 1], y=[0, 1]),
            ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            font=dict(size=18),      # text size
            itemsizing='constant',   # keeps marker size constant in legend
            itemwidth=40,            # optional, adds spacing between items
            bgcolor='rgba(255,255,255,0.7)',  # optional: semi-transparent background
            x=0.01, y=0.99           # position (bottom-left to top-right)
            )
        )
    
    fig.show()
    return 


