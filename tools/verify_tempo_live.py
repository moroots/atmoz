"""Check, against the live NASA archive, the assumptions atmoz makes about TEMPO L3 files.

atmoz reads TEMPO granules with h5py, and was written from the product layout rather than from a live test. This script makes a
small real request (one search, a few granule reads, about a few hundred KB transferred, nothing saved) and checks:

  1. the Earthdata login and the (short name, version) search, and that every returned scan starts on the requested day;
  2. NO2_L3: the nearest-pixel read (units, quality flag, scan time in UTC on the right day);
  3. O3PROF_L3: 24 layers ordered top -> bottom, altitudes in km, the bottom layer starting at the ground, pressure bounds in hPa,
     ozone in DU converted to a plausible ppb mixing ratio, and the lowest-layer shortcut equal to the last layer of the full profile;
  4. ``integrate_layers`` over the bottom layer equals that layer.

Usage (needs a free Earthdata account, https://urs.earthdata.nasa.gov/ ; credentials in ~/.netrc, see docs/satellite.md):

    pip install "atmoz[satellite]"
    python tools/verify_tempo_live.py [YYYY-MM-DD]

The default day is 2025-07-28 and the default point is NASA Goddard (38.98 N, 76.83 W). Exit code 0 when every check passes,
1 when one fails (the message says which assumption to revisit), 2 when the check could not be run (login, no granule).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from atmoz.satellite import TEMPO, integrate_layers
from atmoz.satellite import tempo as tp

POINT = {"GSFC": (38.98, -76.83)}
MAX_PROBES = 8          # granules tried before giving up on finding one that observed the point

results: list = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    return ok


def probe(client: TEMPO, granules: list, spec: dict):
    """Read granules (afternoon scans first) until one holds a pixel for the point; returns the granule result or None."""
    def key(g):
        hour = TEMPO._scan_start(g).hour
        return abs(hour - 18)
    for granule in sorted(granules, key=key)[:MAX_PROBES]:
        name = tp._granule_name(granule)
        print(f"  reading {name} ...")
        out = tp._read_granule(granule, spec)
        if out["rows"] is not None:
            return name, out
    return None


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else "2025-07-28"
    start = pd.Timestamp(day)
    print(f"TEMPO live check for {day}, point {POINT['GSFC']}")

    print("1. search")
    no2 = TEMPO("NO2_L3")
    try:
        granules = no2.search(day)
    except Exception as exc:        # login failure, network
        print(f"Could not search: {type(exc).__name__}: {exc}\nCheck ~/.netrc (see docs/satellite.md) and your connection.")
        return 2
    if not granules:
        print(f"No NO2_L3 V04 granules for {day}; pick another day.")
        return 2
    starts = [TEMPO._scan_start(g) for g in granules]
    check(all(start <= s < start + pd.Timedelta(days=1) for s in starts), f"{len(granules)} granules, all scans start on {day}")
    check(len(set(starts)) == len(starts), "no duplicate scans (a version-less search would double count V03 and V04)")

    print("2. NO2_L3 point read")
    spec = no2._validate(POINT, None, None, 0, False)
    found = probe(no2, granules, spec)
    if found is None:
        print("None of the probed granules observed the point; try another day or point.")
        return 2
    name, out = found
    rows = out["rows"]
    check(str(out["scan"].tz) == "UTC" and out["scan"].normalize().tz_localize(None) == start,
          "scan time is UTC and on the requested day", str(out["scan"]))
    check("molec" in out["units"]["vertical_column_troposphere"].lower(),
          "tropospheric column unit", out["units"]["vertical_column_troposphere"])
    val = rows["vertical_column_troposphere"].iloc[0]
    check(bool(np.isnan(val)) or 0 < val < 1e18, "column is NaN (masked) or a plausible 0-1e18 molecules/cm2", f"{val:.3g}")
    check(rows["quality_flag"].iloc[0] in (0, 1, 2) or pd.isna(rows["quality_flag"].iloc[0]), "quality flag is 0, 1 or 2")

    print("3. O3PROF_L3 point read")
    o3 = TEMPO("O3PROF_L3")
    o3_granules = o3.search(day)
    if not o3_granules:
        print(f"No O3PROF_L3 V04 granules for {day}.")
        return 2
    full_spec = o3._validate(POINT, None, None, None, False)
    found = probe(o3, o3_granules, full_spec)
    if found is None:
        print("None of the probed O3PROF granules observed the point.")
        return 2
    name, out = found
    ds = o3._assemble({name: out}, full_spec, (day, day))
    frame = ds["GSFC/ozone_profile"].df
    check(len(frame) == 24, "24 layers", str(len(frame)))
    check(frame["layer"].tolist() == list(range(24)) and frame["alt"].is_monotonic_decreasing,
          "layers are ordered top -> bottom (layer 0 is the highest)")
    check(frame["alt"].max() < 100 and frame["alt"].min() > -1, "layer altitudes are km", f"{frame['alt'].min():.2f}..{frame['alt'].max():.2f}")
    ground = frame["alt_lower"].iloc[-1]
    check(frame["alt_lower"].iloc[-1] == frame["alt_lower"].min() and -0.5 < ground < 5,
          "the bottom layer starts at the ground (site elevation, km asl)", f"{ground:.3f}")
    check((frame["pressure_thickness"] > 0).all() and frame["pressure_thickness"].sum() < 1100,
          "pressure thicknesses are positive hPa and sum to less than the surface pressure", f"{frame['pressure_thickness'].sum():.0f} hPa")
    ppb = frame["value"]
    check(ppb.between(5, 5000).all(), "layer ozone is a plausible 5-5000 ppb", f"{ppb.min():.0f}..{ppb.max():.0f}")
    check(0.5 < frame["partial_column"].sum() / frame["total_ozone_column"].iloc[0] < 1.5,
          "layer partial columns add up to roughly the total ozone column",
          f"{frame['partial_column'].sum():.0f} vs {frame['total_ozone_column'].iloc[0]:.0f} DU")

    low_spec = o3._validate(POINT, None, None, None, True)
    low = o3._assemble({name: tp._read_granule(dict_granule(o3_granules, name), low_spec)}, low_spec, (day, day))["GSFC/ozone_profile"].df
    check(len(low) == 1 and np.isclose(low["value"].iloc[0], ppb.iloc[-1]),
          "lowest_layer=True equals the last layer of the full profile", f"{low['value'].iloc[0]:.1f} vs {ppb.iloc[-1]:.1f} ppb")
    print(f"  bottom layer spans {low['alt_lower'].iloc[0]:.2f}-{low['alt_upper'].iloc[0]:.2f} km (documented: ground to about 2 km above it)")

    print("4. integrate_layers")
    top = float(low["alt_upper"].iloc[0] - low["alt_lower"].iloc[0])
    layer = integrate_layers(ds["GSFC/ozone_profile"], 0.0, top, reference="agl")
    check(np.isclose(layer.df["column"].iloc[0], low["partial_column"].iloc[0], rtol=1e-6),
          "0-to-layer-top AGL column equals the bottom layer", f"{layer.df['column'].iloc[0]:.2f} DU")

    failed = results.count(False)
    print(f"\nRESULT: {'OK, every assumption holds.' if not failed else f'{failed} CHECK(S) FAILED: revisit atmoz/satellite/tempo.py.'}")
    return 1 if failed else 0


def dict_granule(granules: list, name: str):
    return next(g for g in granules if tp._granule_name(g) == name)


if __name__ == "__main__":
    raise SystemExit(main())
