# atmoz.satellite

NASA [TEMPO](https://tempo.si.edu/) (Tropospheric Emissions: Monitoring of Pollution) gridded Level 3 products over North America, read at points and
boxes **without downloading whole granules**.

```python
from atmoz.satellite import TEMPO, integrate_layers, du_to_ppbv
```

Install the extra first: `pip install "atmoz[satellite]"` (earthaccess, h5py, pydantic). TEMPO data need a free NASA Earthdata login, see
[Credentials](#credentials). `import atmoz.satellite` works without the extra; the dependencies are imported when you search or read.

---

## Quick start

```python
from atmoz.satellite import TEMPO, integrate_layers

# Tropospheric NO2 column at one point, one day (each scan is ~10 minutes; ~49 granules per day)
no2 = TEMPO("NO2_L3").download("2025-07-28", points={"GSFC": (38.98, -76.83)})
col = no2["GSFC/vertical_column_troposphere"]      # AtmozFrame: UTC time index, value, lat, lon, quality_flag
col.units                                           # {"value": "molecules/cm2"}

# Ozone profile at the same point: long format, one row per scan and layer
o3 = TEMPO("O3PROF_L3").download("2025-07-28", points={"GSFC": (38.98, -76.83)})
prof = o3["GSFC/ozone_profile"]                     # index t (UTC); alt (km asl), value (ppb), partial_column (DU), ...
prof.units["value"], prof.alt_reference             # ("ppb", "asl")

integrate_layers(o3, 0, 2, reference="agl")         # ground to 2 km: column (DU) and mean mixing ratio (ppb) per scan

# A box: every pixel inside, lowest retrieval layer only (much cheaper)
box = TEMPO("O3PROF_L3").download("2025-07-28", bbox=(-77.0, 38.9, -76.7, 39.1), lowest_layer=True)
box["bbox/ozone_profile"].df                        # index t; lat, lon, alt, value (ppb), ...
```

Like every atmoz source, `download()` returns an `AtmozDataset` of `AtmozFrame`s and `fetch()` only puts files on disk (see
[Conventions](conventions.md)).

---

## `TEMPO(product="NO2_L3", version="V04")`

| Product | Short name | Default variables | Shape |
|---------|------------|-------------------|-------|
| `"NO2_L3"` | `TEMPO_NO2_L3` | `vertical_column_troposphere`, `vertical_column_troposphere_uncertainty` | column: one number per pixel and scan |
| `"O3PROF_L3"` | `TEMPO_O3PROF_L3` | `ozone_profile` (also `ozone_profile_error`) | profile: 24 layers per pixel and scan (BETA product) |

Searches are pinned to one version: a TEMPO short name searched without a version returns every version together (V03 and V04 both exist), which
double counts. Any variable under the granule's `product/` or `support_data/` group can be asked for with `variables=[...]`
(`NO2_L3` also has the total and stratospheric columns, for instance).

Nothing is logged in when you create the object; the login happens when the first search or read needs it.

### `download(start_date, end_date=None, *, points=None, bbox=None, variables=None, max_quality=0, lowest_layer=False, only_scans=None, max_workers=4, mode="thread", retries=3)`

Searches CMR for the days (UTC, inclusive; granules are selected by **scan start time**: a scan that began at 23:58 the previous evening is dropped),
streams each granule with `earthaccess.open` and reads only the requested pixels.

| Argument | Description |
|----------|-------------|
| `points` | `{name: (lat, lon)}`, a list of `(lat, lon)` or one `(lat, lon)`. The nearest pixel is returned; a point more than one pixel outside the grid is skipped with a warning. Unnamed points are called `"lat,lon"` (3 decimals). |
| `bbox` | `(west, south, east, north)`. Every pixel inside, under the region name `"bbox"`; pixels that are all NaN are dropped. |
| `variables` | Names under `product/` or `support_data/`. |
| `max_quality` | Column products: keep pixels with `main_data_quality_flag <= max_quality` (0 normal, 1 suspicious, 2 bad); the others become NaN. `None` disables the mask. |
| `lowest_layer` | Profile products: read only the bottom layer (ground to about 2 km above it in V04). |
| `only_scans` | Scan times (UTC), e.g. `frame.t` of an earlier download: only granules that start within two minutes before one of them are read. |
| `mode` | `"thread"` (default) or `"process"`. h5py holds a global lock during remote reads, so threads do not overlap network waits; `"process"` is much faster for many granules. |

It is **all-or-nothing**: if a granule cannot be read after `retries` attempts (dropped connections are retried with a growing pause),
`DownloadError` names every failure and nothing is returned. `NoDataError` is raised when no granule exists for the days, or when none of the
granules holds a pixel for the request. Results are not cached; to work repeatedly on the same granules use `fetch` and `read`.

### `fetch(start_date, end_date=None, *, max_workers=3, retries=6, refresh=False, cache_dir=None)`

Downloads **whole granules** (every variable, the full grid) and returns `{file name: Path}`. Files go to the atmoz cache (namespace `tempo`) or
`cache_dir`. A granule is 400-550 MB for `O3PROF_L3`, so a dropped connection resumes with an HTTP `Range` request, a finished file (checked by size) is
kept and skipped next time, and an interrupted file never looks complete (`.part` until done).

### `TEMPO.read(paths, *, points=None, bbox=None, variables=None, max_quality=0, lowest_layer=False, product=None)`

The same subsetting on granules already on disk (a file, a folder or a list, e.g. the result of `fetch`); no login. The product and version come
from the file names unless `product=` is given.

```python
files = TEMPO("O3PROF_L3").fetch("2025-07-28", "2025-07-29")
ds = TEMPO.read(list(files.values()), points={"GSFC": (38.98, -76.83)})
```

### `search(start_date, end_date=None)`

The granules (earthaccess objects) whose scan starts in the window, oldest first.

---

## What you get

A dataset holds **one frame per region and variable**, keyed `"{region}/{variable}"`: `"GSFC/vertical_column_troposphere"`, `"bbox/ozone_profile"`.
`ds.metadata` records the product, version, the granule names, the request, units as the files give them and any skipped points.

### Column products (`NO2_L3`)

| Axis / column | Meaning |
|---------------|---------|
| index `t` | UTC time of the scan (the time stored in the file) |
| `lat`, `lon` | pixel centre (the nearest pixel for a point; `metadata["lat_requested"]` is what you asked for) |
| `value` | the variable, in the unit of the file (`frame.units["value"]`, `molecules/cm2`) |
| `quality_flag` | `main_data_quality_flag`, nullable integer |

There is no `alt` axis (a column is not a profile). Rows for a **point** are kept even when masked (NaN): each granule covers only a strip of the grid
(about ten minutes of scanning), so a point is NaN with a flag in scans that did not observe it. That is coverage, not an error; `frame.df.dropna()`
removes them. Bbox pixels that are all NaN are dropped.

### Profile products (`O3PROF_L3`)

Long format: one row per scan, pixel and layer, sorted by time and layer (layer 0 is the **top**).

| Column | Unit | Meaning |
|--------|------|---------|
| `value` | ppb | ozone mixing ratio of the layer, from the partial column and the layer's pressure thickness (hydrostatic; see `du_to_ppbv`) |
| `alt` | km | middle of the layer, **above sea level** |
| `alt_lower`, `alt_upper` | km | layer limits above sea level. The lower limit of the bottom layer is the ground |
| `layer` | | index in the file, 0 = top |
| `pressure_thickness` | hPa | `|p_bottom - p_top|` of the layer |
| `partial_column` | DU | the retrieved partial ozone column of the layer (what the file holds) |
| `total_ozone_column` | DU | total column of the pixel |
| `eff_cloud_fraction` | | effective cloud fraction of the pixel (a diagnostic; nothing is masked on it) |
| `lat`, `lon` | deg | pixel centre |

`metadata["alt_reference"]` is `"asl"`. Point frames also carry `metadata["site_elevation_km"]` (the ground, taken from the bottom layer limit), so
`frame.with_alt_reference("agl")` works and `atmoz.collocate` can convert; a box has no single ground height and no `site_elevation_km`.
Scans that did not observe the pixel are left out (no NaN rows). `ozone_profile_error` is an ordinary variable and gets its own frame.

`lowest_layer=True` reads the bottom layer only: the frame has one row per scan and pixel (`layer` = 23).

### Ozone helpers

* `du_to_ppbv(du, dp_hpa)`: partial column (DU) in a layer to a mixing ratio (ppb), from the layer's pressure thickness.
* `integrate_layers(frame_or_dataset, z_min=0, z_max=2, *, reference="asl")`: the partial column between two altitudes. Each layer contributes in
  proportion to its vertical overlap with the slab (a layer is assumed uniform). Returns, per scan and pixel, `column` (DU) and `value` (ppb, the
  mean mixing ratio of the slab) with `lat`/`lon` and no `alt` axis; `metadata["layer_km"]` and `["layer_reference"]` say which slab.
  `reference="agl"` measures the limits from each pixel's ground.

---

## Caveats

* **Vertical information is coarse.** TEMPO's lowest layer is 2-3 km thick (ground to about 2 km above it); a 0-2 km value is essentially that layer and
  carries little independent vertical information. Compare it with a lidar through a layer mean, not level by level
  (see `examples/tempo_vs_tolnet.py`). `O3PROF_L3` is a BETA product.
* **Altitude reference.** atmoz declares the `O3PROF_L3` altitudes as above sea level, from the product's altitude bounds, with the ground at the bottom of the
  lowest layer. `tools/verify_tempo_live.py` checks this on a real granule; run it once after installing.
* **Coverage.** A granule covers a strip of the grid, so the site is observed in only some scans of a day (a point is observed at most once per scan).
* **Speed.** Cost is network round trips, not data volume. A day is tens of granules; use `mode="process"` and `max_workers=4`, `lowest_layer=True`
  for boxes, `only_scans=` to skip granules known not to cover the site, and `fetch` when you will read the same days again.
* **Units.** Whatever the file reports (`molecules/cm2` for NO2); atmoz does not convert columns. Ozone `value` is ppb; the original DU is `partial_column`.
* **Pixel match.** Points match the nearest pixel centre within one grid spacing (about 4 km for O3PROF); beyond that the point is
  skipped with a warning.
* **Fill values** (the file's `_FillValue`, or anything beyond 1e29) become NaN.

---

## Credentials

TEMPO is access-controlled but free. Create an account at <https://urs.earthdata.nasa.gov/>, approve the "NASA GESDISC DATA ARCHIVE" application, and
either create `~/.netrc` (`machine urs.earthdata.nasa.gov login <user> password <pass>`, `chmod 600`) or let atmoz write it:

```python
from atmoz.data_access.NASA import EarthData
EarthData(username="...", password="...")      # writes ~/.netrc once, then logs in
```

See [Configuration](configuration.md#nasa-earthdata). `search`, `download` and `fetch` fail with a `FileNotFoundError` (no `.netrc`) or a
`PermissionError` (login refused) before any data is requested.

---

## Examples

* `examples/tempo_map.py`: maps of TEMPO lowest-layer ozone and tropospheric NO2 around a site on a cartopy basemap (`pip install "atmoz[satellite,plots]"`).
* `examples/tempo_vs_tolnet.py`: TEMPO ozone profiles next to a TOLNet lidar (curtains), then a 0-2 km AGL layer comparison with `atmoz.collocate` and `atmoz.stats.compare`.
* `examples/aqs_nearest_monitor.py`: the nearest EPA AQS monitor to a site for ozone, NO2 and particulates.
* `tools/verify_tempo_live.py`: a small live check of the assumptions above, run by hand (needs your Earthdata login).
