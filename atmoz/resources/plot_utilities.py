# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:34:20 2025

@author: Magnolia
"""

import matplotlib.image as image
from atmoz.resources import useful_functions
import matplotlib.dates as mdates

def __apply_water_mark(fig, filepath, **kwargs):
    params = {
        "add_axes": {
            "rect": [0.12, 0.73, 0.3, 0.15],
            "anchor": "SW",
            "zorder": 10
        }
    }

    params = useful_functions.merge_dicts(params, kwargs)
    with open(filepath, "rb") as file:
            im = image.imread(file)
    ax_wm = fig.add_axes(**params["add_axes"])
    ax_wm.imshow(im, alpha=0.7)
    ax_wm.axis('off')
    return fig

def __add_plot_params(ax, params: dict):
    for func_name, kwargs in params.items():
        target = getattr(ax, func_name, None)
        if callable(target):
            if isinstance(kwargs, dict):
                target(**kwargs)
            elif isinstance(kwargs, (list, tuple)):
                target(*kwargs)
            else:
                target(kwargs)  # single scalar
    return 

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
    
    return 

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
    return
