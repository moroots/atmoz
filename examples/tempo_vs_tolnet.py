"""TEMPO ozone profiles against a TOLNet lidar: curtains side by side, then a 0-2 km layer comparison with ``atmoz.collocate``.

1. TOLNet ozone profiles for an instrument and a few days (JSON, no login).
2. TEMPO ``O3PROF_L3`` profiles at the nearest pixel to the lidar (streamed; needs an Earthdata login).
3. Curtains of both with ``atmoz.plot.curtain`` (same colour scale).
4. The mean ozone between the ground and 2 km above it (AGL), from each source: ``integrate_layers`` for TEMPO,
   ``atmoz.vertical.layer_mean`` for the lidar, put on the TEMPO times with ``atmoz.collocate`` and summarised with ``atmoz.stats.compare``.

TEMPO's lowest layer is about 2 km thick and TEMPO is BETA; the layer mean is the only fair comparison with a 30 m lidar grid.
TEMPO scans cover a strip of the grid, so the site is observed in only some scans of a day.

Needs ``pip install "atmoz[lidar,satellite]"``.

    python examples/tempo_vs_tolnet.py --start 2025-06-15 --end 2025-06-17 --instrument "NASA JPL SMOL-5"
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

import atmoz
from atmoz.lidar import TOLNET
from atmoz.satellite import TEMPO, integrate_layers

LAYER_KM = (0.0, 2.0)      # above ground level


def concat(frames: list) -> atmoz.AtmozFrame:
    """Join frames of one source (e.g. one per day) into one, keeping the first one's units and metadata."""
    first = frames[0]
    df = pd.concat([f.df for f in frames]).sort_index()
    return atmoz.AtmozFrame(df, units=first.units, lat=first.axes["lat"], lon=first.axes["lon"], alt=first.axes["alt"],
                            metadata=first.metadata)


def get_tolnet(start: str, end: str, instrument: str) -> atmoz.AtmozFrame:
    """All HIRES, centrally processed profiles of one instrument between *start* and *end* as one frame."""
    tolnet = TOLNET()
    groups = tolnet.instrument_groups
    hits = groups.index[groups["instrument_group_name"] == instrument]
    if not len(hits):
        raise SystemExit(f"Unknown TOLNet instrument {instrument!r}. Options: {sorted(groups['instrument_group_name'])}")
    ds = TOLNET.download(start, end, instrument_group=[int(hits[0])], product_type=[4], processing_type=[1])
    return concat([ds[key] for key in ds.keys()])


def get_tempo(start: str, end: str, lat: float, lon: float) -> atmoz.AtmozFrame:
    """TEMPO ozone profiles at the pixel nearest to (lat, lon): long, one row per scan and layer."""
    ds = TEMPO("O3PROF_L3").download(start, end, points={"site": (lat, lon)}, max_workers=4, mode="process")
    return ds["site/ozone_profile"]


def layer_comparison(tempo: atmoz.AtmozFrame, lidar: atmoz.AtmozFrame, layer=LAYER_KM, max_gap: str = "1h"):
    """Layer-mean ozone (ppb) of both sources on the TEMPO times (the anchor), with comparison statistics."""
    tempo_layer = integrate_layers(tempo, *layer, reference="agl")
    lidar_layer = atmoz.vertical.layer_mean(lidar, *layer, reference="agl")
    out = atmoz.collocate([tempo_layer, lidar_layer], names=["tempo", "tolnet"], max_gap=max_gap)
    return out, atmoz.stats.compare(out, "tolnet", "tempo")      # obs = lidar, model slot = TEMPO


def make_figure(lidar: atmoz.AtmozFrame, tempo: atmoz.AtmozFrame, out: atmoz.AtmozFrame, instrument: str):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    atmoz.plot.curtain(lidar, ax=axes[0], title=f"TOLNet {instrument}")
    atmoz.plot.curtain(tempo, ax=axes[1], title="TEMPO O3PROF L3 V04 (BETA), nearest pixel")
    for ax in axes[:2]:
        ax.set_ylim(0, 15)
    df = out.df
    axes[2].plot(df.index, df["tolnet"], label=f"TOLNet {LAYER_KM[0]:g}-{LAYER_KM[1]:g} km AGL mean", lw=1.2)
    axes[2].plot(df.index, df["tempo"], "o", ms=4, label="TEMPO, same layer")
    axes[2].set_ylabel(f"Ozone ({out.units['tempo']})")
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.4)
    axes[2].set_xlabel("Time (UTC)")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default="2025-06-15")
    parser.add_argument("--end", default="2025-06-17")
    parser.add_argument("--instrument", default="NASA JPL SMOL-5")
    parser.add_argument("--out", default="tempo_vs_tolnet.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    lidar = get_tolnet(args.start, args.end, args.instrument)
    lat, lon = lidar.location
    print(f"TOLNet {args.instrument}: {lat:.2f}, {lon:.2f}, {len(lidar.t.unique())} profiles")
    tempo = get_tempo(args.start, args.end, lat, lon)
    print(f"TEMPO: {len(tempo.t.unique())} scans observed the site (pixel {tempo.location[0]:.3f}, {tempo.location[1]:.3f})")

    out, stats = layer_comparison(tempo, lidar)
    print(stats.round(2).to_string())
    fig = make_figure(lidar, tempo, out, args.instrument)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
