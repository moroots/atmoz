# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:34:20 2025

@author: Maurice Roots
"""

import numpy as np

def tolnet():
    ncolors = [
        np.array([255, 140, 255]) / 255.0,
        np.array([221, 111, 242]) / 255.0,
        np.array([187, 82, 229]) / 255.0,
        np.array([153, 53, 216]) / 255.0,
        np.array([119, 24, 203]) / 255.0,
        np.array([0, 0, 187]) / 255.0,
        np.array([0, 44, 204]) / 255.0,
        np.array([0, 88, 221]) / 255.0,
        np.array([0, 132, 238]) / 255.0,
        np.array([0, 165, 255]) / 255.0,
        np.array([0, 235, 255]) / 255.0,
        np.array([39, 255, 215]) / 255.0,
        np.array([99, 255, 150]) / 255.0,
        np.array([163, 255, 91]) / 255.0,
        np.array([211, 255, 43]) / 255.0,
        np.array([255, 255, 0]) / 255.0,
        np.array([250, 200, 0]) / 255.0,
        np.array([255, 159, 0]) / 255.0,
        np.array([255, 111, 0]) / 255.0,
        np.array([255, 63, 0]) / 255.0,
        np.array([255, 0, 0]) / 255.0,
        np.array([216, 0, 15]) / 255.0,
        np.array([178, 0, 31]) / 255.0,
        np.array([140, 0, 47]) / 255.0,
        np.array([102, 0, 63]) / 255.0,
        np.array([200, 200, 200]) / 255.0,
        np.array([140, 140, 140]) / 255.0,
        np.array([80, 80, 80]) / 255.0,
        np.array([52, 52, 52]) / 255.0,
        np.array([0, 0, 0]),
    ]
    cmap = mpl.colors.ListedColormap(ncolors)
    cmap.set_under([1, 1, 1])
    cmap.set_over([0, 0, 0])
    bounds = [0.001, *np.arange(5, 121, 5), 150, 200, 300, 600]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

# def apply_colorbar(ax, mappable, name: str, **kwargs):
#     """Attach a registered colorbar to an axes."""
#     cmap, norm = get_colorbar(name)
#     mappable.set_cmap(cmap)
#     mappable.set_norm(norm)
#     return ax.figure.colorbar(mappable, ax=ax, **kwargs)