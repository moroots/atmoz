import numpy as np

from atmoz.plotting.styles import styles

# The curtain look now lives in atmoz.plotting.styles (fonts that exist everywhere, a 11 x 4.4 in figure).
# TOLNET.plot_curtain still reads these names.
curtain_plot_theme = {**styles.curtain.rc, "figure.figsize": styles.curtain.figsize}

tolnet_plot_params = {
    "ax.set_ylabel": {
        "ylabel": "Altitude (km ASL)"
        },

    "ax.set_yticks": {
        "ticks": np.arange(0, 16, 1)
        },

    "ax.set_xlabel": {
        "xlabel": "Time"
        },

    "ax.set_title": {
        "label": r"Ozone Mixing Ratio Profile",
        "fontsize": 13
        },

    "ax.grid": {
        "visible": True,
        "color": "gray",
        "linestyle": "--",
        "linewidth": 0.5
        },

    "ax.set_ylim": [0, 15],

    # no "fig.tight_layout": the curtain style uses constrained layout, which handles the colorbar

    "fig.colorbar": {
        "pad": 0.01, 
        "ticks": [0.001, *np.arange(10, 121, 10), 150, 200, 300, 600],
        "sub_functions": {
            "set_label": {
                "label": "Ozone Mixing Ratio ($ppb_v$)",
                "weight": "bold",
                "size": 11
                },
                    
            "ax.tick_params": {
                "labelsize": 9
                }
            }
        }
    }
