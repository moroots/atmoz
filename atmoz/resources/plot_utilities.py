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

def __apply_watermark(fig, filepath, **kwargs):
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

def apply_plot_params(fig, ax, params: Dict[str, Any], obj: Optional[Any] = None) -> Dict[str, Any]:
    """
    Recursively apply plotting parameters from a nested dictionary
    to Matplotlib objects (Axes, Figure, Colorbar, etc.).

    This function supports dotted names (e.g., ``"fig.colorbar"``,
    ``"cbar.ax.tick_params"``) to traverse attributes and call
    methods with specified keyword arguments.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object.
    fig : matplotlib.figure.Figure
        Target figure object.
    params : dict
        Nested dictionary specifying method names and their kwargs.
        Each entry should be of the form::

            {
                "fig.colorbar": {
                    "pad": 0.01,
                    "ticks": [...],
                    "sub_functions": {
                        "set_label": {"label": "Ozone", "size": 16},
                        "ax.tick_params": {"labelsize": 16}
                    }
                }
            }

    obj : object, optional
        Current target object (used during recursion).
        Defaults to resolving from ``ax`` or ``fig``.

    Returns
    -------
    results : dict
        Dictionary mapping function names to their return values.

    Notes
    -----
    - Does not modify the input ``params`` dict.
    - If a function cannot be resolved or called, it is skipped.
    - ``sub_functions`` allows recursive calls on the returned object.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> params = {
    ...     "fig.colorbar": {
    ...         "pad": 0.01,
    ...         "ticks": [0, 1, 2],
    ...         "sub_functions": {
    ...             "set_label": {"label": "Example", "size": 12}
    ...         }
    ...     }
    ... }
    >>> apply_plot_params(ax, fig, params)
    """
    results = {}

    for func_name, kwargs in params.items():
        # Split dotted path
        parts = func_name.split(".")
        target = obj

        # Resolve top-level object
        if target is None:
            if parts[0] == "ax":
                target, parts = ax, parts[1:]
            elif parts[0] == "fig":
                target, parts = fig, parts[1:]
            else:
                raise ValueError(f"Cannot resolve target for {func_name}")

        # Traverse attributes
        for attr in parts:
            target = getattr(target, attr, None)
            if attr == "colorbar":
                print(target, attr)
            if target is None:
                break

        if target is None or not callable(target):
            continue

        # Extract sub-functions
        call_kwargs = copy.deepcopy(kwargs) if isinstance(kwargs, dict) else kwargs
        sub_funcs = None
        if isinstance(call_kwargs, dict):
            sub_funcs = call_kwargs.pop("sub_functions", None)

        # Call the target function
        if isinstance(call_kwargs, dict):
            result = target(**call_kwargs)
        elif isinstance(call_kwargs, (list, tuple)):
            result = target(*call_kwargs)
        else:
            result = target(call_kwargs)


        # Recurse into sub-functions using the result as obj
        # if sub_funcs and result is not None:
        #     print(result)
        #     _, _, sub_results = apply_plot_params(fig, ax, sub_funcs, obj=result)

    return fig, ax, results


def __apply_near_real_time(ax, **kwargs):
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

def __apply_datetime_axis(ax):
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
