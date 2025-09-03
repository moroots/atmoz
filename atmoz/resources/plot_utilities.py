# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:34:20 2025

@author: Maurice Roots
"""


import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional
import matplotlib.image as mimg
from atmoz.resources import useful_functions
import matplotlib.dates as mdates

def apply_watermark(fig, filepath, **kwargs):
    params = {
        "add_axes": {
            "rect": [0.12, 0.73, 0.3, 0.15],
            "anchor": "SW",
            "zorder": 10
        }
    }

    params = useful_functions.merge_dicts(params, kwargs)
    with open(filepath, "rb") as file:
            im = mimg.imread(file)
    ax_wm = fig.add_axes(**params["add_axes"])
    ax_wm.imshow(im, alpha=0.7)
    ax_wm.axis('off')
    return ax_wm

def apply_plot_params(fig, ax, **params):
    """
    Apply a set of plotting parameters to a Matplotlib Axes or pyplot function.

    This function takes a dictionary of function names and their arguments,
    looks up the corresponding callable either on the given Axes instance (`ax`)
    or in the global pyplot (`plt`) namespace, and calls it with the provided
    arguments.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object associated with the plot. Returned unchanged.
    ax : matplotlib.axes.Axes
        The target axes object to which most plotting parameters will be applied.
    **params : dict
        A mapping of function names to argument specifications. Each key is the
        name of a method or function (string), and each value is one of:
        
        - dict : passed as keyword arguments to the target function
        - list or tuple : passed as positional arguments
        - any other value : passed as a single positional argument

        Example
        -------
        >>> params = {
        ...     "set_title": {"label": "Example Plot", "fontsize": 14},
        ...     "set_xlabel": "Time (s)",
        ...     "grid": {"visible": True, "linestyle": "--"}
        ... }
        >>> fig, ax = __apply_plot_params(fig, ax, **params)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same figure object, unchanged.
    ax : matplotlib.axes.Axes
        The same axes object, after modifications.

    Notes
    -----
    - The function first checks for the given function name on the Axes object.
      If not found, it falls back to checking `matplotlib.pyplot`.
    - Only callables are executed; missing or non-callable attributes are skipped.
    """

    def __call_attr(target, kwargs): 
        if callable(target):
            if isinstance(kwargs, dict):
                result = target(**kwargs)
            elif isinstance(kwargs, (list, tuple)):
                result = target(*kwargs)
            else:
                result = target(kwargs) 
        else: 
            return None
        
        return result

    for func_name, kwargs in params.items():
        obj, attr = func_name.split(".")
        if obj == "ax":
            target = getattr(ax, attr, None)
        elif obj == "fig":
            target = getattr(fig, attr, None)

        if target is not None: 
            if isinstance(kwargs, dict) and "sub_functions" in kwargs.keys(): 
                sub_functions = kwargs.pop("sub_functions", {})
                result = __call_attr(target, kwargs)
                for sub_func_name, sub_func_kwargs in sub_functions.items():
                    sub_target = getattr(result, sub_func_name, None)
                    __call_attr(sub_target, sub_func_kwargs)
            else:
                result = __call_attr(target, kwargs)

    return fig, ax



def apply_near_real_time(ax, **kwargs):
    params = {
        "s": 'NRT DATA. NOT CITABLE.',
        "fontsize": 30,
        "color": "black",
        "ha": "center",
        "va": "center",
        "rotation": 25,
        "transform": ax.transAxes,
        "alpha": 0.5 
        }
    
    params = useful_functions.merge_dicts(params, kwargs)

    ax.text(0.3, 0.5, **params)
    ax.text(0.7, 0.5, **params)
    
    return ax

def apply_datetime_axis(ax):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = [
        '%y',  
        '%b',       
        '%d',       
        '%H:%M',    
        '%H:%M',    
        '%S.%f'
        ]

    formatter.zero_formats = [''] + formatter.formats[:-1]

    formatter.zero_formats[3] = '%d-%b'

    formatter.offset_formats = [
        '',
        '%Y',
        '%b-%Y',
        '%d-%b-%Y',
        '%d-%b-%Y',
        '%d-%b-%Y %H:%M'
        ]
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    return ax
