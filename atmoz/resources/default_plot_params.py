import numpy as np

curtain_plot_theme = {
            # ime_Tick label font sizes
            "xtick.labelsize": 12,      # previously params["fontsize_ticks"]
            "ytick.labelsize": 12,

            # Axes labels
            "axes.labelsize": 14,       # previously params["fontsize_label"]

            # Title font
            "axes.titlesize": 16,       # previously params["title"]["fontsize"]
            "axes.titleweight": "bold",

            # Grid style
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.grid.which": "both",

            # Minor ticks
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,

            # Major ticks
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,

            # Figure size (optional)
            "figure.figsize": (20, 6),

            # Fonts
            "font.family": "Courier New",
            "font.size": 14
            }

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
        "fontsize": 16
        },

    "ax.grid": {
        "visible": True,
        "color": "gray",
        "linestyle": "--",
        "linewidth": 0.5
        },

    "ax.set_ylims": [0, 15],

    "fig.savefig": {
        "fname": "test.png",
        "dpi": 300,
        "transparent": True,
        "format": "png",
        "bbox_inches": "tight"
        },

    "fig.layout": "tight",

    "fig.colorbar": {
        "pad": 0.01, 
        "ticks": [0.001, *np.arange(10, 121, 10), 150, 200, 300, 600],
        "sub_functions": {
            "set_label": {
                "label": "Ozone Mixing Ration ($ppb_v$)",
                "weight": "bold",
                "size": 16
                },
                    
            "ax.tick_params": {
                "labelsize": 16
                }
            }
        }
    }