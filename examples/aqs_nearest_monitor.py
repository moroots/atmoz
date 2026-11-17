"""The nearest EPA AQS monitor to a site, for each parameter, from the pre-generated hourly files.

Not every monitor measures everything, so the nearest monitor is chosen per parameter; its distance to the site is in each panel title.
Uses ``EPA_PreGen.download`` (hourly files, cached as parquet), ``AtmozDataset.inventory()`` for the monitor positions and
``convert_units`` to show ozone in ppb. The hourly files are nationwide annual CSVs of hundreds of MB, so the first run takes a while;
the bbox and dates filter the rows before anything is kept in memory.

Needs only the core install.

    python examples/aqs_nearest_monitor.py --lat 33.43 --lon -112.12 --start 2025-06-15 --end 2025-06-17
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import atmoz
from atmoz.surface import EPA_PreGen

# panel name, AQS pre-generated file, AQS parameter code
MEASUREMENTS = [
    ("Ozone", "ozone", "44201"),
    ("NO2", "no2", "42602"),
    ("PM2.5", "pm25", "88101"),
    ("PM10", "pm10", "81102"),
]
MIN_HOURS = 24          # a monitor must report at least this many hours in the window


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def download(lat: float, lon: float, start: str, end: str, radius_deg: float = 1.5) -> atmoz.AtmozDataset:
    """Hourly AQS frames (``"{site}/{parameter code}"``) inside a box around the site."""
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    return EPA_PreGen.download(
        [m[1] for m in MEASUREMENTS], "hourly", list(years),
        bbox=(lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg),
        start_date=start, end_date=end, parameter=[m[2] for m in MEASUREMENTS],
    )


def nearest_monitors(ds: atmoz.AtmozDataset, lat: float, lon: float) -> dict:
    """``{parameter code: (frame key, distance km)}``: the nearest frame with at least MIN_HOURS hours, per parameter code."""
    inv = ds.inventory()
    inv["code"] = inv["key"].str.split("/").str[1]
    inv["km"] = haversine_km(lat, lon, inv["lat"], inv["lon"])
    picked = {}
    for code, group in inv[inv["rows"] >= MIN_HOURS].groupby("code"):
        best = group.loc[group["km"].idxmin()]
        picked[code] = (best["key"], float(best["km"]))
    return picked


def to_display_units(frame: atmoz.AtmozFrame) -> atmoz.AtmozFrame:
    """Ozone and NO2 in ppb for display (AQS ozone is in ppm); other units stay as the file declares them."""
    unit = frame.units.get("value")
    return frame.convert_units("value", "ppb") if unit in ("ppm", "ppt") else frame


def make_figure(ds: atmoz.AtmozDataset, picked: dict, site: tuple, start: str, end: str):
    panels = [(name, picked[code]) for name, _f, code in MEASUREMENTS if code in picked]
    if not panels:
        raise SystemExit("No monitor near the site reported enough hours in the window.")
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(12, 2.8 * len(panels) + 1), squeeze=False)
    for ax, (name, (key, km)) in zip(axes[:, 0], panels):
        frame = to_display_units(ds[key])
        ax.plot(frame.t, frame.df["value"], lw=1.1)
        ax.set_ylabel(f"{name} ({frame.units.get('value', '?')})")
        ax.set_title(f"monitor {key.split('/')[0]}, {km:.1f} km from the site", fontsize=10, loc="left")
        ax.grid(True, alpha=0.4)
    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle(f"Nearest EPA AQS monitors to ({site[0]:.2f}, {site[1]:.2f}), {start} to {end}")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lat", type=float, default=33.43)
    parser.add_argument("--lon", type=float, default=-112.12)
    parser.add_argument("--start", default="2025-06-15")
    parser.add_argument("--end", default="2025-06-17")
    parser.add_argument("--out", default="aqs_nearest_monitor.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    ds = download(args.lat, args.lon, args.start, args.end)
    picked = nearest_monitors(ds, args.lat, args.lon)
    for code, (key, km) in sorted(picked.items()):
        print(f"{code}: {key.split('/')[0]}  {km:6.1f} km  {len(ds[key])} h")
    fig = make_figure(ds, picked, (args.lat, args.lon), args.start, args.end)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
