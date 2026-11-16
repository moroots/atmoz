# Collocation

`atmoz.collocate` puts several sources on the grid of the first one: its times, its altitudes and its unit. The result has one
column per source, so they can be compared row by row (and passed to [`atmoz.stats.compare`](analysis.md)).

```python
import atmoz
out = atmoz.collocate([lidar, model, sonde], names=["lidar", "geos_cf", "sonde"])
out = atmoz.collocate(lidar, model)                       # varargs work too; two sources are enough
```

The sources are `AtmozFrame`s or `AtmozDataset`s (see [Conventions](conventions.md#containers)). The **first is the anchor**: the output has
its rows, its `alt` / `lat` / `lon` axes and its unit; every other source is brought onto those rows. At least two sources are needed.

| Argument | Meaning |
|----------|---------|
| `names` | Output column names, one per source, distinct. Default: `obs`, `model`, `src3`, `src4`, ... |
| `cols` | The column compared in each source (default `"value"`): one name for all, a list, or `{name: column}`. |
| `max_gap` | Time tolerance for sources made of profiles at discrete times (a model, a lidar): NaN when the anchor time is farther than this from either neighbouring time. Default `"1h"`. |
| `window` | Half-width of the time window averaged for a trajectory source (a sonde). Default `"1h"`. |
| `method` | Vertical method: `"auto"` (default), `"interp"` or `"mean"`. |
| `alt` | With surface and profile sources mixed: the altitude (km) at which profiles are read. Default: the lowest valid level. |
| `max_distance_km` | Datasets only: the farthest a frame may be from the anchor frame to be paired with it. Default 25. |
| `assume_same_reference` | Skip the altitude-reference check (see below). |

## What it does

* **Time.** A source of profiles at discrete times is interpolated linearly between its two nearest times. If either neighbour is
  farther than `max_gap` away the result is NaN (use `"2h"` to bridge a missing hour). A source that is one continuous trajectory,
  where every row has its own time (a [sonde](sonde.md)), is averaged over its rows within `window` of the anchor time. A balloon takes
  an hour or more to ascend, so the default window of one hour covers the troposphere around the anchor time; widen it to cover the
  whole ascent, or narrow it to compare only the altitudes the balloon reached near that time.
* **Altitude.** Each profile is brought onto the anchor's levels using its own `alt` column, which may differ per time (GEOS-CF layer
  heights do). `method="interp"` interpolates linearly in altitude, `"mean"` averages the points that fall into each anchor level's bin, and `"auto"`
  uses `"mean"` when the other profile is finer than the anchor and `"interp"` when it is coarser. Outside the other profile's altitude range
  the result is NaN. A trajectory source always uses the bin mean unless you pass `method="interp"`.
* **Altitude reference.** Profile altitudes must say what they are measured from (`frame.alt_reference`, `"asl"` or `"agl"`; the ground elevation is
  `frame.metadata["site_elevation_km"]`). When the references differ, the sea-level side is converted to above ground
  (`frame.with_alt_reference("agl")`) and the result is in agl. If a reference or elevation is missing, `collocate` raises instead of guessing; pass
  `assume_same_reference=True` to skip the check. The model's terrain height at its grid cell is not subtracted, so over hilly terrain
  the two ground levels can differ by the terrain difference.
* **Units.** Every column is converted to the anchor's unit when both are declared and differ (for example ppm to ppb).
* **Surface and profile together.** If any source is a surface series (no `alt`), the comparison is made at the surface: profiles are reduced
  to their lowest valid level at each time, or to `alt=` (km). The result is a surface-style frame on the surface anchor's times;
  `metadata["sources"][name]["reduced_to_surface"]` records which sources were reduced.
* **Datasets.** See below.

The result is an `AtmozFrame` with a UTC time index, the anchor's `alt` / `lat` / `lon` columns, one column per source and `units` for
each. `metadata` holds `sources` (every input's metadata by name), `columns`, `max_gap`, `window`, `method` and, for profiles,
`alt_reference` and `site_elevation_km`.

## Lidar, model and sonde

```python
import pandas as pd
import atmoz
from atmoz import Sonde
from atmoz.lidar import TOLNET
from atmoz.models import GEOS_CF

# One TOLNet file: a long AtmozFrame (UTC time, alt in km above sea level, ozone in ppb)
tol = TOLNET.download("2025-07-28", "2025-07-28", product_type=[4], processing_type=[1])
lidar = tol[next(iter(tol))]
lat, lon = lidar.location                                  # the instrument

# GEOS-CF ozone profiles at the same place; end one day later so there is a model time after the last profile
cf = GEOS_CF.download(lat, lon, lidar.t.min().date(), lidar.t.max().date() + pd.Timedelta("1D"),
                      collection="chm", variables=["o3"])
model = cf["o3"]                                           # long: value (ppb), alt (km above ground), lev

sonde = Sonde.read("data/Sondes/flight.ict").frame        # trajectory: alt above sea level, value in ppb

out = atmoz.collocate(lidar, model, sonde, names=["lidar", "geos_cf", "sonde"], window="1h")
out.df[["alt", "lidar", "geos_cf", "sonde"]]
out.units                                                  # {"lidar": "ppb", "geos_cf": "ppb", "sonde": "ppb", "alt": "km"}
out.metadata["alt_reference"]                              # "agl": the lidar and sonde were converted from sea level

atmoz.stats.compare(out, "lidar", "geos_cf", by="alt")     # bias, rmse, r by altitude
atmoz.plot.comparison_profile(out)                         # mean profiles with their spread
```

The lidar reports altitude above sea level and carries the instrument elevation (`site_elevation_km`); GEOS-CF layer heights are above the ground
(`alt_reference="agl"`). The three sources are therefore compared in km above ground. The sonde frame needs a `site_elevation_km` too: the
reader parses it from the file's `LOCATION` line when it is there; otherwise set it with `sonde.with_metadata(site_elevation_km=...)`.

## AQS and GEOS-CF datasets

With datasets the frames are paired by location: for every frame of the anchor dataset, the nearest frame of each other dataset within
`max_distance_km` (among frames at essentially the same place, the one that overlaps most in time). The result is an `AtmozDataset` keyed by
the anchor's keys.

```python
import atmoz
from atmoz.surface import AirNow
from atmoz.models import GEOS_CF

aqs = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025,
                      bbox=(-77.5, 38.5, -76.5, 39.5), start_date="2025-07-27", end_date="2025-07-30")
inv = aqs.inventory()
lat, lon = inv["lat"].median(), inv["lon"].median()
cf = GEOS_CF.download(lat, lon, "2025-07-27", "2025-07-31", collection="chm_sfc", variables=["o3"])   # one point near the sites

out = atmoz.collocate(aqs, cf, names=["aqs", "geos_cf"], max_distance_km=100)
out.metadata["pairs"]                                      # {aqs key: {"geos_cf": key of the paired frame}}
out.metadata["unpaired"]                                   # {aqs key: ["geos_cf"]} for sites with no partner in range
atmoz.stats.compare(out)                                   # one block per site, plus a pooled "all"
```

`AirNow.download` gives ozone in ppm and GEOS-CF in ppb: the model is converted to the anchor's unit (ppm here). Put the data in ppb
first with `frame.convert_units("value", "ppb")` on each frame if you prefer ppb. Anchor frames with no partner within range are left out of
the result and listed in `metadata["unpaired"]`. When only some sources are datasets, a plain frame is used for every pair.

Known caveat: GEOS-CF timestamps are hour-centred (`00:30`, `01:30`, ...), while AQS and AirNow hourly values are labelled at the hour
beginning (`00:00`, `01:00`, ...). The two series are therefore offset by half an hour, and the linear time interpolation matches an hourly surface value to the
model at its label time, half an hour before the middle of the hour the surface value averages. `max_gap` takes care of matching the
times (the default 1 hour covers a 30 minute offset), but the model is not shifted for you; shift the model's index by 30 minutes first
if the half hour matters for your analysis.
