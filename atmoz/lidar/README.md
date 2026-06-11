# atmoz.lidar

Access and visualize NASA [TOLNet](https://tolnet.larc.nasa.gov/) (Tropospheric Ozone Lidar Network) data.

TOLNet is a network of ground-based ozone lidar instruments operated across North America by NASA and partner agencies. This module queries the TOLNet REST API, downloads JSON data files into memory, and renders time-altitude curtain plots.

```python
from atmoz.lidar import TOLNET
```

---

## Quick Start

```python
from atmoz.lidar import TOLNET

tolnet = TOLNET()

# Download one week of high-resolution ozone profiles
tolnet.import_data(
    min_date="2025-07-28",
    max_date="2025-08-01",
    product_type=[4],          # HIRES
    processing_type=[1],       # Centrally Processed
)

# Plot every loaded instrument
tolnet.tolnet_curtains()

# Or plot a specific key
key = list(tolnet.data.keys())[0]
tolnet.plot_curtain(tolnet.data[key])
```

---

## `TOLNET_DATA_QUERY` — Query Parameters

All query parameters are validated by a Pydantic model before the request is sent. These same fields are accepted as `**kwargs` by `import_data()`, `_get_files_list()`, and `download()`.

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

### `TOLNET()`

No arguments. Initializes the API client with:
- `self.base_url` = `"https://tolnet.larc.nasa.gov/api"`
- `self.data` = `{}` — populated by `import_data()`
- `self.meta_data` = `{}` — populated by `import_data()`
- `self._errors` = `[]` — failed file IDs with error messages

---

### Discovery Properties

These are `cached_property` attributes — fetched once on first access and cached for the session.

#### `tolnet.products → pd.DataFrame`

All available product types, indexed by ID.

```python
print(tolnet.products)
#     name
# id
# 1   O3Lidar
# 4   HIRES
# ...
```

#### `tolnet.instrument_groups → pd.DataFrame`

All instrument groups (sites), indexed by ID. Contains name, institution, location.

```python
print(tolnet.instrument_groups)
# Shows all TOLNet sites: NASA GSFC, NASA JPL, LaRC, NOAA CSL, UAH, ECCC, etc.
```

#### `tolnet.processing_types → pd.DataFrame`

Processing type IDs and names.

```python
print(tolnet.processing_types)
# 1: Centrally Processed (GLASS)
# 2: In-House
# 3: Unprocessed
```

#### `tolnet.file_types → pd.DataFrame`

File format type IDs and names (HDF GEOMS, JSON, Image, etc.).

#### `tolnet.api_schema → dict`

Raw OpenAPI YAML schema for the TOLNet API.

---

### `import_data(min_date, max_date, **kwargs)`

Primary data fetch method. Queries the file list, downloads each matching JSON file in parallel (2 threads), and stores results in `self.data`.

```python
tolnet.import_data(
    min_date="2025-07-01",
    max_date="2025-07-31",
    product_type=[4],              # HIRES only
    processing_type=[1],           # Centrally Processed only
    instrument_group=[7, 8],       # specific sites
)
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `min_date` | `str` | Yes | Start date `YYYY-MM-DD` |
| `max_date` | `str` | Yes | End date `YYYY-MM-DD` |
| `**kwargs` | | | Any `TOLNET_DATA_QUERY` field (see above). |
| `GEOS_CF` | `bool` | No | If `True`, also fetch co-located GEOS-CF model data and merge into `self.data`. |

**Returns:** `self` (chainable).

After calling `import_data()`:
- `tolnet.data` is populated (see [Data Structures](#data-structures))
- `tolnet.meta_data` is populated with file-level metadata
- `tolnet._errors` lists any files that failed to download/parse

---

### `download(dest_dir, file_type, threaded, max_workers, **params)`

Query the API file list and download matching files to disk.

```python
# Download JSON files (lightweight, no EarthData login needed)
tolnet.download(
    dest_dir="./tolnet_data",
    file_type="json",
    min_date="2025-07-01",
    max_date="2025-07-31",
    product_type=4,
)

# Download HDF GEOMS files (requires NASA EarthData credentials)
tolnet.download(
    dest_dir="./tolnet_hdf",
    file_type="hdf",
    max_workers=4,
    min_date="2025-07-01",
    max_date="2025-07-31",
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dest_dir` | `str` | required | Output directory (created if missing). |
| `file_type` | `str` | `"json"` | `"json"` or `"hdf"`. HDF requires NASA EarthData credentials. |
| `threaded` | `bool` | `True` | Use parallel download. |
| `max_workers` | `int` | `8` | Thread count for parallel download. |
| `**params` | | | Any `TOLNET_DATA_QUERY` fields to filter the file list. |

Files are saved as `{file_id}.json` or `{file_id}.hdf`.

---

### `plot_curtain(data, **kwargs)`

Render a time-altitude ozone curtain plot for one instrument/key.

```python
key = ("NASA JPL SMOL-2", "Centrally Processed (GLASS)", "39.24x-76.363")
tolnet.plot_curtain(tolnet.data[key])

# With options
tolnet.plot_curtain(
    tolnet.data[key],
    title="JPL SMOL-2 — July 2025",
    use_contourf=True,
    xlims=["2025-07-01", "2025-07-07"],
    time_resolution="30min",
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data` | `dict` | required | `{date_str: DataFrame}` — index = UTC datetime, columns = altitude (km) |
| `title` | `str` | `None` | Plot title |
| `use_contourf` | `bool` | `False` | Use `contourf` instead of `pcolormesh` |
| `xlims` | `"auto" \| List[str]` | `"auto"` | Date range to display: `["2025-07-01", "2025-07-07"]` |
| `time_resolution` | `"auto" \| str` | `"auto"` | Resample to a fixed resolution before plotting (e.g. `"30min"`, `"1h"`) |

The color scale uses the standard TOLNet ozone color map. A TOLNet watermark is applied automatically. Near-real-time data includes an NRT indicator.

---

### `tolnet_curtains(**kwargs)`

Plot one curtain for every key in `self.data`. Uses the key tuple as the title unless `title` is overridden.

```python
tolnet.tolnet_curtains()
tolnet.tolnet_curtains(use_contourf=True)
```

Accepts the same `**kwargs` as `plot_curtain()`.

---

### `search_by_file_name(file_name, match_type) → pd.DataFrame`

Search the API for files by name pattern.

```python
results = tolnet.search_by_file_name("JPL_2025")
results = tolnet.search_by_file_name("2025-07", match_type="begins_with")
results = tolnet.search_by_file_name(".hdf",    match_type="ends_with")
```

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `match_type` | `"exact_match"`, `"begins_with"`, `"ends_with"` | `"exact_match"` | How the name is matched |

Returns the same DataFrame schema as `_get_files_list()`.

---

### `calendar(instrument_group, file_type, product_type, processing_type) → pd.DataFrame`

Return a chronological listing of available files for a specific instrument and file type.

```python
cal = tolnet.calendar(instrument_group=7, file_type=1)
cal = tolnet.calendar(instrument_group="NASA JPL SMOL-2", file_type="HDF GEOMS", product_type=4)
```

---

### `change_timezone(timezone) → self`

Convert all timestamps in `self.data` to a new timezone in-place.

```python
tolnet.change_timezone("US/Eastern")
tolnet.change_timezone("America/Los_Angeles")
```

Uses `dateutil.tz` timezone strings. Returns `self` (chainable).

---

### `_get_files_list(**params) → pd.DataFrame`

Fetch the paginated file list from the API. Accepts all `TOLNET_DATA_QUERY` fields as kwargs. Returns a DataFrame with columns matching the file list dtypes:

`id`, `file_name`, `file_server_location`, `instrument_group_name`, `product_type_name`, `processing_type_name`, `file_type_name`, `start_data_date`, `end_data_date`, `upload_date`, `latitude`, `longitude`, `altitude`, `near_real_time`, `file_size`, `revision`, `public`, `isAccessible`, etc.

---

## Data Structures

### `tolnet.data`

```python
tolnet.data = {
    (instrument_group_name, processing_type_name, "latXlon"): {
        "YYYY-MM-DD": pd.DataFrame,   # index=UTC datetime, columns=altitude (km)
        "YYYY-MM-DD": pd.DataFrame,
        ...
    },
    ...
}
```

**Key format:** `(str, str, str)` — e.g. `("NASA JPL SMOL-2", "Centrally Processed (GLASS)", "39.24x-76.363")`

**Value DataFrame:** rows = UTC timestamped observations, columns = altitude levels in km, values = ozone mixing ratio (ppbv).

### `tolnet.meta_data`

```python
tolnet.meta_data = {
    (instrument_group_name, processing_type_name, "latXlon"): {
        "filename.json": {metadata_dict},
        ...
    }
}
```

Each metadata dict contains:
- `LATITUDE.INSTRUMENT`, `LONGITUDE.INSTRUMENT` — instrument coordinates
- `fileInfo` — dict with `start_data_date`, `end_data_date`, `instrument_group_name`, `processing_type_name`, etc.
- `altitude` — altitude grid metadata
- `datetime` — time axis metadata

### `tolnet._errors`

List of strings: `"Error processing file {filename}: {exception}"` for any files that failed during `import_data()`.

---

## Common Patterns

### Iterate over loaded data

```python
for key, daily_data in tolnet.data.items():
    inst, proc, latlon = key
    print(f"{inst} | {proc} | {latlon}: {len(daily_data)} days loaded")

    for date, df in daily_data.items():
        print(f"  {date}: {df.shape[0]} time steps, altitudes {df.columns.min():.1f}–{df.columns.max():.1f} km")
```

### Filter to a specific site

```python
target = [k for k in tolnet.data if "JPL" in k[0]]
for key in target:
    tolnet.plot_curtain(tolnet.data[key], title=str(key))
```

### Check available instrument IDs before querying

```python
tolnet = TOLNET()
print(tolnet.instrument_groups[["name", "city", "state"]])
print(tolnet.products)
print(tolnet.processing_types)
```
