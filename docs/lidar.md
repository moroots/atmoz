# atmoz.lidar

Access and visualize NASA [TOLNet](https://tolnet.larc.nasa.gov/) (Tropospheric Ozone Lidar Network) data.

TOLNet is a network of ground-based ozone lidar instruments operated across North America by NASA and partner agencies. This module queries the TOLNet REST API, downloads JSON data files (cached on disk), and renders time-altitude curtain plots.

```python
from atmoz.lidar import TOLNET
```

Install the extra first: `pip install "atmoz[lidar]"`. JSON downloads need no login; HDF downloads need NASA EarthData credentials (see [Configuration](configuration.md#nasa-earthdata)).

---

## Quick Start

```python
from atmoz.lidar import TOLNET

# One week of high-resolution ozone profiles -> an AtmozDataset (cached on disk, all-or-nothing)
ds = TOLNET.download("2025-07-28", "2025-08-01", product_type=[4], processing_type=[1])
ds.inventory()                               # one row per file: site, altitude range, unit, time span
frame = ds[next(iter(ds))]                   # an AtmozFrame, keyed by file name
frame.df.head()                              # UTC time index; alt (km), value, lat, lon
frame.units                                  # {"alt": "km", "value": "ppb", ...} (read from the file)

groups = TOLNET.group(ds)                    # {(instrument, processing, "latxlon"): {"YYYY-MM-DD": AtmozFrame}}
TOLNET.plot_curtains(ds)                     # one curtain per instrument / site
TOLNET.plot_curtain(groups[next(iter(groups))], title="JPL SMOL-2")
```

---

## `TOLNET_DATA_QUERY` — Query Parameters

All query parameters are validated by a Pydantic model before the request is sent. These same fields are accepted as `**kwargs` by `download()`, `_get_files_list()`, and `fetch()`.

### Filter Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instrument_group` | `int \| str \| List[int\|str]` | `None` | Instrument group ID(s) or name(s). See `tolnet.instrument_groups` for valid values. |
| `product_type` | `int \| str \| List[int\|str]` | `None` | Product type ID(s). See `tolnet.products`. Common: `4` = HIRES, `1` = O3Lidar. |
| `processing_type` | `int \| str \| List[int\|str]` | `None` | Processing type ID(s). See `tolnet.processing_types`. Common: `1` = Centrally Processed. |
| `file_type` | `int \| str \| List[int\|str]` | `None` | File type ID(s). See `tolnet.file_types`. |
| `near_real_time` | `bool \| int \| str` | `None` | `True`/`1`/`"true"` for NRT files only; `False`/`0`/`"false"` to exclude NRT. |

### Date Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_date` | `str` | `None` | Start of data date range. Format: `YYYY-MM-DD`. |
| `max_date` | `str` | `None` | End of data date range. Format: `YYYY-MM-DD`. |
| `min_upload_date` | `str` | `None` | Filter by upload date range start. |
| `max_upload_date` | `str` | `None` | Filter by upload date range end. |

### Geospatial Fields

Use either a point + radius **or** a bounding box — not both.

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `latitude` | `float` | −90 to 90 | Site latitude for point query. Requires `longitude` and `radius`. |
| `longitude` | `float` | −180 to 180 | Site longitude for point query. |
| `radius` | `float` | > 0 | Search radius (km) around the point. |
| `minLatitude` | `float` | −90 to 90 | Bounding box south edge. All four bbox fields required together. |
| `maxLatitude` | `float` | −90 to 90 | Bounding box north edge. Must be > `minLatitude`. |
| `minLongitude` | `float` | −180 to 180 | Bounding box west edge. |
| `maxLongitude` | `float` | −180 to 180 | Bounding box east edge. Must be > `minLongitude`. |

### Sorting Fields

| Field | Options | Default | Description |
|-------|---------|---------|-------------|
| `order` | `"data_date"`, `"upload_date"`, `"instrument_group_name"`, `"product_type_name"`, `"processing_type_name"`, `"file_type_name"`, `"file_name"` | `"data_date"` | Sort field. |
| `order_direction` | `"asc"`, `"desc"` | `"desc"` | Sort direction. |

---

## `TOLNET` Class

Everything TOLNet lives in `atmoz/lidar/tolnet.py`: the API client, HDF reading and plotting. It follows the package
[conventions](conventions.md): `download()` returns an `AtmozDataset`, `fetch()` returns `{name: Path}`.

### `TOLNET.download(min_date, max_date, **query) → AtmozDataset`

Lists the matching files, downloads each JSON file (4 threads, cached), and parses them.

```python
ds = TOLNET.download(
    "2025-07-01", "2025-07-31",
    product_type=[4],              # HIRES only
    processing_type=[1],           # Centrally Processed only
    instrument_group=[7, 8],       # specific sites
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `min_date`, `max_date` | `str` | required | `YYYY-MM-DD` range of the data. |
| `**query` | | | Any `TOLNET_DATA_QUERY` field (see above). |
| `max_workers` | `int` | `4` | Concurrent downloads. |
| `use_cache`, `refresh`, `cache_dir` | | `True`, `False`, shared cache | Near-real-time files are re-checked hourly; final files never change. |
| `silent`, `show_traceback`, `session_handler` | | | The usual options. |

**Returns** an `AtmozDataset` (`source="TOLNet"`), **one frame per file**, keyed by the file name:

* `ds[file_name]` is an `AtmozFrame` in long format: UTC `time` index (repeated once per altitude), columns `alt` (km), `value` (ozone),
  constant `lat` / `lon` (the instrument), and any other per-point variable of the file with the same shape as `value` (uncertainties, say).
* Units are **read from the file**: `frame.units["value"]` (normally `ppb`; `ppbv` is normalised), altitude is converted to km from the unit
  the file declares (m or km). Nothing is assumed for `value` if the file does not say.
* `frame.metadata`: `instrument`, `processing`, `product`, `file`, `file_id`, `near_real_time`, `latitude`, `longitude`, `start_data_date`, `fileInfo`.
* `ds.metadata["info"]`: TOLNet's metadata per file (`LATITUDE.INSTRUMENT`, `LONGITUDE.INSTRUMENT`, `fileInfo`, ...).
* `ds.metadata["groups"]`: `"instrument|processing|latxlon"` → the file names of that instrument.
* `ds.metadata["file_list"]`: the API's file list as a DataFrame.

```python
frame = ds[next(iter(ds))]
wide = frame.df.pivot(columns="alt", values="value")      # time x altitude, the shape az.plot.curtain takes
lat, lon = frame.location                                 # the instrument, e.g. for GEOS_CF.download(lat, lon, ...)
```

**Raises** `NoDataError` if nothing matches, `DownloadError` if any file fails (nothing is returned; finished files stay cached).
To compare with GEOS-CF, call `GEOS_CF.download(...)` at the instrument's coordinates (`frame.location`) and use [`atmoz.collocate`](collocation.md).

### `TOLNET.group(ds) → dict`

`{(instrument, processing, "latxlon"): {"YYYY-MM-DD": AtmozFrame}}`, the shape the curtain plot takes.

### `TOLNET.fetch(min_date, max_date, *, file_type="json", dest_dir=None, **query) → {file_name: Path}`

Saves the raw files and returns their paths (`dest_dir` defaults to the atmoz cache). `file_type="hdf"` fetches the GEOMS HDF
files and needs NASA EarthData credentials (see [Configuration](configuration.md#nasa-earthdata)).

```python
paths = TOLNET.fetch("2025-07-01", "2025-07-31", file_type="hdf", dest_dir="./tolnet_hdf", product_type=4)
nodes = TOLNET.open_hdf(paths.values())                    # {filename: node}, opened lazily
nodes[next(iter(nodes))]["O3.MIXING.RATIO.VOLUME_DERIVED"][()]
```

### Plotting

```python
TOLNET.plot_curtain(groups[key])                           # same as az.plot.curtain, plus the TOLNet watermark
TOLNET.plot_curtain(groups[key], title="JPL SMOL-2", tz="US/Eastern", nrt=True)
figs = TOLNET.plot_curtains(ds)                            # {group key: (fig, ax)}, one per instrument
```

Both return `(fig, ax)`; options are those of [`curtain`](plotting.md) (`title`, `tz`, `altitude_units`, `sonde`, `style`, `figsize`, `nrt`,
`watermark`), plus `show=True` to call `plt.show()` in scripts. `plot_curtain` takes a long `AtmozFrame` (it pivots to time x altitude itself),
a wide DataFrame, or a dict of them. To plot a time window, slice the frame first (`frame.df.loc["2025-07-28":"2025-07-30"]`).

### Catalogue helpers (instance methods)

```python
t = TOLNET()
t.products; t.instrument_groups; t.processing_types; t.file_types      # DataFrames indexed by id, fetched once
t.search_by_file_name("2025-07", match_type="begins_with")
t.calendar(instrument_group=7, file_type=1)
```

`search_by_file_name` and `calendar` return DataFrames; `api_schema` is the parsed OpenAPI description.

---

## UCN: Unified Ceilometer Network

The [Unified Ceilometer Network](https://ucn.cas.hamptonu.edu/viewer/) (Hampton University) publishes one NetCDF file per
ceilometer per UTC day. atmoz reads two levels (**`L0b`**, the raw 16 s profiles at ~33 MB a day, is **planned but not supported yet**; asking for it raises a clear error, and `UCN.planned_levels` lists it):

| Level | Contents | Size per day |
|-------|----------|--------------|
| `L2` (default) | 10-minute attenuated backscatter profile (1540 heights, 10 m apart) + boundary-layer, cloud and QC retrievals | ~1.4 MB download, ~0.75 MB cached |
| `L3` | hourly boundary-layer statistics (UCN's own coverage rules) | ~1.3 MB download, ~11 KB cached |

An L3 file repeats L2's backscatter array exactly (checked on one day), so atmoz keeps only L3's hourly table.

```python
from atmoz.lidar import UCN

UCN.sites()                                       # ceilometer numbers and site names
UCN.available(9, "2026-10-01", "2026-10-31")      # what a request would fetch (and its size), no data downloaded

ds = UCN.download(9, "2026-11-01")                # McMillan Reservoir, DC; an AtmozDataset
frame = ds["CEILO-9_20261101_L2"]                 # one AtmozFrame per ceilometer, day and level
frame.df.head()                                   # long: UTC time index, alt (km), value, per-time scalars, lat, lon
frame.units["value"]                              # "km^-1.sr^-1": value is the attenuated backscatter
frame.df["nsl_ml_height"]                         # a per-time scalar, repeated along altitude
frame.metadata["altitude_m"]                      # heights in m, plus list labels and the file attributes

month = UCN.download(9, "2026-10-01", "2026-10-31", skip_missing=True)
both = UCN.stack(month, 9)                        # all days of ceilometer 9 joined into one AtmozFrame
UCN.backscatter(both).plot(x="time")              # (time x altitude) xarray DataArray for plotting

UCN.download(9, "2026-11-01", level="L3")         # hourly statistics instead
UCN.fetch(9, "2026-11-01")                        # {name: parquet path}: only put the files on disk
```

L3 frames have one row per hour, an all-NaN `alt` column (so L2 and L3 frames share one schema) and no `value`. Profile-like variables that are not
per altitude (cloud layers, say) are not in the frame; they are listed in `frame.metadata["not_in_frame"]` and stay in the cached parquet.
`lat` / `lon` columns are present when the file carries the ceilometer's position.

**Choosing data.** Ceilometer number(s), UTC day range (inclusive), `level`, and `skip_missing`. Everything else (time of day,
height range) is a slice of the DataFrame or the array from `UCN.backscatter()`. There is no selection by state or bounding box yet:
the ceilometer list carries no coordinates.

**Gentle on the server, by design**

* Files are downloaded **once**, then served from the cache. Days older than two are final; the newest two are re-checked hourly.
* 2 workers by default, a pause (`pause=0.5` s) between request starts, retries with backoff, and an `atmoz/<version>` User-Agent.
* `skip_missing=True` uses the viewer's list of available days, so days it does not list are never requested.
* A request of more than about 1 GB to download (about 700 L2 days) or 2 GB in memory (about 1,700 L2 days, ~1.2 MB each) raises `RequestTooLargeError` before anything is
  downloaded. Preview it with `UCN.available(...)`, split it up, or pass `allow_large=True`.

**The cache** (`~/.atmoz/cache/ucn`) holds one parquet file per ceilometer-day, not the NetCDF: about half the size for L2.
Profile variables are list columns, and heights, labels, units and attributes are stored in the file's metadata.
`fetch()` therefore returns parquet paths; `pandas.read_parquet` reads them (profile variables as one array per row, the wide form of the cache), while `download()` turns each file into a long `AtmozFrame`.

**Missing days.** UCN answers a missing day with a 404 (or an ordinary page). By default that raises `DownloadError`
(all-or-nothing); with `skip_missing=True` those days are left out and listed in `ds.metadata["missing"]`. A missing day is never cached.
L2 and L3 appear a day or more after the data is recorded.
