"""TEMPO maps: lowest-layer ozone and tropospheric NO2 column around a site, on a cartopy basemap.

Streams TEMPO L3 granules for one UTC day (only the pixels in the box are transferred) and maps the mean over all scans that
observed each pixel.

* ozone: ``O3PROF_L3`` bottom retrieval layer (ground to about 2 km above it), ppb. A BETA product; the layer is 2 km thick, so this
  is not a surface concentration.
* NO2: ``NO2_L3`` tropospheric vertical column, 1e15 molecules/cm2, pixels with quality flag 0 only.

Needs ``pip install "atmoz[satellite,plots]"`` and an Earthdata login (see docs/satellite.md). Cartopy downloads its Natural Earth
layers the first time (cached afterwards).

    python examples/tempo_map.py --day 2025-07-28 --lat 39.0 --lon -76.8 --box-km 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import atmoz
from atmoz.satellite import TEMPO

NO2 = "vertical_column_troposphere"


def box_around(lat: float, lon: float, side_km: float) -> tuple:
    """``(west, south, east, north)`` of a square of *side_km* centred on a site."""
    half = side_km / 2.0
    dlat = half / 110.574
    dlon = half / (111.320 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def pixel_means(frame: atmoz.AtmozFrame, scale: float = 1.0) -> pd.DataFrame:
    """Mean ``value`` per pixel over the scans of a bbox frame, as a lat x lon table (NaN where never observed)."""
    mean = frame.df.groupby(["lat", "lon"])["value"].mean() * scale
    return mean.unstack("lon").sort_index()


def get_ozone(day: str, bbox: tuple) -> atmoz.AtmozFrame:
    ds = TEMPO("O3PROF_L3").download(day, bbox=bbox, lowest_layer=True, max_workers=4, mode="process")
    return ds["bbox/ozone_profile"]


def get_no2(day: str, bbox: tuple) -> atmoz.AtmozFrame:
    ds = TEMPO("NO2_L3").download(day, bbox=bbox, variables=[NO2], max_quality=0, max_workers=4, mode="process")
    return ds[f"bbox/{NO2}"]


def draw(ax, grid: pd.DataFrame, label: str, title: str, site: tuple, cmap: str):
    """One pixel-mean map on a cartopy axes."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    plate = ccrs.PlateCarree()
    mesh = ax.pcolormesh(grid.columns.to_numpy(), grid.index.to_numpy(), np.ma.masked_invalid(grid.to_numpy()),
                         transform=plate, cmap=cmap, shading="nearest")
    ax.add_feature(cfeature.STATES, edgecolor="#444444", linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAKES, edgecolor="#4a90c2", facecolor="none", linewidth=0.5)
    ax.plot(site[1], site[0], marker="*", ms=14, color="magenta", mec="black", transform=plate, label="site")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.6, linestyle=":")
    gl.top_labels = gl.right_labels = False
    ax.set_title(title)
    ax.figure.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85, label=label)


def make_figure(ozone: atmoz.AtmozFrame, no2: atmoz.AtmozFrame, bbox: tuple, site: tuple, day: str):
    import cartopy.crs as ccrs

    west, south, east, north = bbox
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True)
    for ax in axes:
        ax.set_extent((west, east, south, north), crs=ccrs.PlateCarree())
    n_o3, n_no2 = ozone.t.nunique(), no2.t.nunique()
    draw(axes[0], pixel_means(ozone), f"Ozone, lowest layer ({ozone.units['value']})",
         f"TEMPO O3PROF L3 (BETA), mean of {n_o3} scan(s)", site, "viridis")
    draw(axes[1], pixel_means(no2, 1e-15), "Tropospheric NO2 column (1e15 molecules/cm2)",
         f"TEMPO NO2 L3, mean of {n_no2} scan(s)", site, "magma_r")
    fig.suptitle(f"TEMPO, {day} (UTC)")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--day", default="2025-07-28", help="UTC day (default 2025-07-28)")
    parser.add_argument("--lat", type=float, default=39.0, help="site latitude")
    parser.add_argument("--lon", type=float, default=-76.8, help="site longitude")
    parser.add_argument("--box-km", type=float, default=150.0, help="side of the box, km")
    parser.add_argument("--out", default="tempo_map.png", help="figure path")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    bbox = box_around(args.lat, args.lon, args.box_km)
    fig = make_figure(get_ozone(args.day, bbox), get_no2(args.day, bbox), bbox, (args.lat, args.lon), args.day)
    fig.savefig(Path(args.out), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
