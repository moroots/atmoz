# atmoz.models

Access NASA [GEOS-CF](https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/) (Goddard Earth Observing System Composition Forecast) chemistry-climate model output at any global site via OPeNDAP.

GEOS-CF provides hourly atmospheric chemistry, composition, and meteorology fields at 0.25° × 0.3125° resolution from 2018 onward, as both a replay (assimilation) product and a rolling 5-day forecast.

```python
from atmoz.models import GEOS_CF
```

---

## Quick Start

```python
from atmoz.models import GEOS_CF

geos = GEOS_CF()
geos.fetch(
    lat=39.055, lon=-76.878,
    start_date="2025-01-01", end_date="2025-01-31",
    collection="aqc", mode="assim",
    variables=["o3"],
    cache_dir=".geos_cf_cache",
)

key = next(iter(geos.data))
ds = geos.data[key]

# Convert mol/mol → ppb
o3_ppb = ds["o3"].squeeze("lev") * 1e9
print(f"O3 range: {float(o3_ppb.min()):.1f} – {float(o3_ppb.max()):.1f} ppb")
```

---

## Collections (`_COLLECTION_CATALOG`)

Three collections are available. Pass the collection name as the `collection` argument to `fetch()`.

| Collection | Description | Variables | Levels | Units |
|------------|-------------|-----------|--------|-------|
| `aqc` | Surface air quality — 1-hr avg | `o3`, `no2`, `co`, `so2`, `pm25_rh35_gcc` | 1 (surface only) | mol/mol (gas), µg/m³ (PM2.5) |
| `chm` | Chemistry profiles — 1-hr avg | `o3`, `no2`, `co`, `so2`, `oh`, `hno3` | 72 model levels | mol/mol |
| `met` | Surface meteorology — 1-hr avg | `t2m`, `u10m`, `v10m`, `q2m`, `zpbl`, `ps` | surface | K, m/s, kg/kg, Pa |

### Modes

| Mode | Description | Date range |
|------|-------------|------------|
| `assim` | Replay/assimilation — meteorology nudged to observations | 2018–2026 |
| `fcast` | 5-day rolling forecast — updated each day | Recent dates only |

`aqc` supports both `assim` and `fcast`. `chm` and `met` are `assim` only.

### Variable reference

| Variable | Collection | Units | Description |
|----------|------------|-------|-------------|
| `o3` | aqc, chm | mol/mol | Ozone |
| `no2` | aqc, chm | mol/mol | Nitrogen dioxide |
| `co` | aqc, chm | mol/mol | Carbon monoxide |
| `so2` | aqc, chm | mol/mol | Sulfur dioxide |
| `pm25_rh35_gcc` | aqc | µg/m³ | PM2.5 at 35% RH |
| `oh` | chm | mol/mol | Hydroxyl radical |
| `hno3` | chm | mol/mol | Nitric acid |
| `t2m` | met | K | 2-m air temperature |
| `u10m` | met | m/s | 10-m zonal wind |
| `v10m` | met | m/s | 10-m meridional wind |
| `q2m` | met | kg/kg | 2-m specific humidity |
| `zpbl` | met | m | Planetary boundary layer height |
| `ps` | met | Pa | Surface pressure |

### Inspect the catalog in code

```python
geos = GEOS_CF()
import json
print(json.dumps(geos.catalog, indent=2, default=str))
```

---

## Units

Gas-phase species (`o3`, `no2`, `co`, `so2`, `oh`, `hno3`) are in **mol/mol** from OPeNDAP.

```python
# mol/mol → ppb
o3_ppb = ds["o3"].squeeze("lev") * 1e9

# mol/mol → ppm
o3_ppm = ds["o3"].squeeze("lev") * 1e6

# PM2.5 is already in µg/m³ — no conversion needed
pm25 = ds["pm25_rh35_gcc"].squeeze("lev")

# Met variables are in SI units (K, m/s, kg/kg, Pa) — no conversion needed
t2m_celsius = ds["t2m"] - 273.15
```

**Note:** OPeNDAP values are typically very small (e.g., median O3 ≈ 4×10⁻⁸). If a variable's median is < 0.01, it is almost certainly in mol/mol and needs the ×1e9 conversion.

---

## `GEOS_CF_QUERY` — Query Validation

All parameters are validated by a Pydantic model before any network request is made.

| Field | Type | Constraint | Default | Description |
|-------|------|------------|---------|-------------|
| `lat` | `float` | −90 < lat < 90 | required | Site latitude (decimal degrees) |
| `lon` | `float` | −180 < lon < 180 | required | Site longitude (decimal degrees) |
| `start_date` | `str` | YYYY-MM-DD or YYYYMMDD | required | Start of date range |
| `end_date` | `str` | ≥ start_date | required | End of date range |
| `collection` | `"aqc" \| "chm" \| "met"` | | `"aqc"` | Data product |
| `mode` | `"assim" \| "fcast"` | | `"assim"` | Replay or forecast |
| `variables` | `List[str]` | lowercase | collection defaults | Variables to retrieve |

Date format: both `YYYY-MM-DD` and `YYYYMMDD` are accepted and normalized to `YYYY-MM-DD` internally.

If `variables` is omitted or `None`, all variables for the collection are fetched.

---

## `GEOS_CF` Class

### `GEOS_CF()`

No arguments. Initializes:
- `self.data` = `{}` — populated by `fetch()`
- `self._errors` = `[]` — error messages from failed fetches

### `catalog` (property)

Returns `_COLLECTION_CATALOG` — the full dict of available collections, variables, OPeNDAP URLs, and units.

---

### `fetch(lat, lon, start_date, end_date, ...)`

Primary data fetch method. Downloads a single-site time series or profile stack from OPeNDAP and stores it as an `xr.Dataset`.

```python
geos.fetch(
    lat=39.055, lon=-76.878,
    start_date="2025-01-01", end_date="2025-12-31",
    collection="aqc",
    mode="assim",
    variables=["o3", "no2"],
    cache_dir=".geos_cf_cache",
    force_download=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `lat` | `float` | required | Latitude in decimal degrees (−90 to 90) |
| `lon` | `float` | required | Longitude in decimal degrees (−180 to 180) |
| `start_date` | `str` | required | Start date: `"YYYY-MM-DD"` or `"YYYYMMDD"` |
| `end_date` | `str` | required | End date: same formats |
| `collection` | `str` | `"aqc"` | `"aqc"`, `"chm"`, or `"met"` |
| `mode` | `str` | `"assim"` | `"assim"` (historical) or `"fcast"` (forecast) |
| `variables` | `List[str]` | collection defaults | Variable name(s) to retrieve (lowercase). Pass `None` for all. |
| `cache_dir` | `str` | `None` | Path to local cache directory. If set, data is read from cache on repeat calls. |
| `force_download` | `bool` | `False` | Re-download even if a valid cache file exists. |

**Returns:** `self` (chainable). Result stored in `self.data[(lat_lon, mode, collection)]`.

#### How OPeNDAP monthly chunking works

The OPeNDAP server caps per-request time slice size. Requesting a full year in one call returns a `DAP failure` error. `fetch()` automatically splits the date range into monthly (chunk_start, chunk_end) pairs and fetches them in parallel using up to 4 worker threads.

```
Loading aqc/assim from OPeNDAP (1 variable(s), 12 monthly chunk(s), 4 workers)...
aqc/assim:  8%|▊  | 1/12 [01:30<16:34, 90.41s/it]
```

Each monthly chunk takes ~90 seconds depending on server load. A full year for one site typically takes 15–20 minutes.

#### Cache behavior

When `cache_dir` is provided:
- First call: downloads from OPeNDAP, saves a NetCDF file to `{cache_dir}/geos_cf_{hash}.nc`
- Subsequent calls with identical parameters: loads from NetCDF in < 1 second
- Cache key: MD5 hash of `{snap_lat}x{snap_lon}_{mode}_{collection}_{start}_{end}_{vars}` where lat/lon are snapped to the nearest 0.25° OPeNDAP grid cell — so sites within the same grid cell share one cache file
- Use `force_download=True` to bypass the cache and re-download

```python
# Recommended pattern for notebooks: always pass cache_dir
geos.fetch(..., cache_dir=".geos_cf_cache")
```

---

### `add_altitude_coords(key, met_key)`

Attach altitude (ZL, layer heights in km) from a MET dataset as a coordinate on a CHM profile dataset. Requires a prior `fetch()` call for `met` collection with a `zl` variable.

```python
# 1. Fetch both CHM and MET profiles
geos.fetch(lat=39.055, lon=-76.878,
           start_date="2025-07-01", end_date="2025-07-31",
           collection="chm", variables=["o3"])

geos.fetch(lat=39.055, lon=-76.878,
           start_date="2025-07-01", end_date="2025-07-31",
           collection="met", variables=["zl"])

# 2. Attach altitude coordinate to the CHM dataset
chm_key = ("39.05x-76.88", "assim", "chm")
geos.add_altitude_coords(chm_key)

# 3. Access CHM with altitude coordinate
ds_chm = geos.data[chm_key]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `key` | `tuple` | required | Key of the CHM dataset to update (e.g. `("39.05x-76.88", "assim", "chm")`) |
| `met_key` | `tuple` | `None` | Key of the MET dataset containing `zl`. Auto-inferred from `key` if `None`. |

**Returns:** `self` (chainable).

---

## Data Structure

### `geos.data`

```python
geos.data = {
    ("39.0x-77.0", "assim", "aqc"): xr.Dataset,
    ("39.0x-77.0", "assim", "chm"): xr.Dataset,
    ("39.0x-77.0", "assim", "met"): xr.Dataset,
    ...
}
```

**Key format:** `(lat_lon_str, mode, collection)` where `lat_lon_str` uses the raw requested coordinates (e.g. `"39.055x-76.878"`).

**Value:** `xr.Dataset` with:

- **`aqc` / `met` dimensions:** `(time,)` after squeeze, or `(time, lev)` before (lev=1 for surface collections)
- **`chm` dimensions:** `(time, lev)` with 72 model levels

**Dataset attributes:**
| Attribute | Description |
|-----------|-------------|
| `lat_requested` | Latitude passed to `fetch()` |
| `lon_requested` | Longitude passed to `fetch()` |
| `lat_actual` | Nearest grid cell latitude used by OPeNDAP |
| `lon_actual` | Nearest grid cell longitude used by OPeNDAP |
| `collection` | Collection name: `"aqc"`, `"chm"`, `"met"` |
| `mode` | `"assim"` or `"fcast"` |
| `start_date` | Requested start date |
| `end_date` | Requested end date |
| `units_note` | Units string for the collection |

### Working with the `lev` dimension

AQC and MET collections include a `lev` dimension of size 1 (surface level). Call `.squeeze("lev")` before converting to a pandas Series:

```python
key = next(iter(geos.data))
ds = geos.data[key]

# Wrong — MultiIndex error
o3_series = ds["o3"].to_series()

# Correct
o3_series = ds["o3"].squeeze("lev").to_series()
```

### Timestamp alignment with EPA AQS

GEOS-CF `tavg_1hr` timestamps mark the **center** of each 1-hour averaging window (`:30`), while EPA AQS timestamps mark the **start** of the hour (`:00`). Floor the GEOS-CF index before merging:

```python
o3_series.index = o3_series.index.floor("h")
```

### Timezone handling

OPeNDAP returns UTC timestamps without timezone info. Localize before merging with timezone-aware EPA data:

```python
if o3_series.index.tz is None:
    o3_series.index = o3_series.index.tz_localize("UTC")
```

---

## Performance Notes

| Scenario | Typical time |
|----------|-------------|
| One monthly chunk, one variable | ~90 seconds |
| Full year (12 chunks, 4 workers) | 15–20 minutes |
| Load from disk cache | < 1 second |

OPeNDAP server response time varies. Avoid re-downloading by always setting `cache_dir` in notebooks. Cache files are plain NetCDF (`.nc`) and can be opened independently with `xr.open_dataset()`.

---

## Common Patterns

### Compare GEOS-CF vs EPA AQS ozone

```python
import pandas as pd
from atmoz.surface import AirNow
from atmoz.surface import utilities as su
from atmoz.models import GEOS_CF

# 1. Download EPA AQS hourly ozone
data = AirNow.download(endpoint="aqs", parameters=["ozone"], resolutions=["hourly"], years=[2025])
df = data["hourly_44201_2025.csv"]
df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# 2. Filter to site of interest
meta = su.extract_metadata(df)
gdf  = su.to_gdf(meta)
gdf_roi = su.slice_bbox(gdf, bbox={"lon": [-77.5, -76.5], "lat": [38.5, 39.5]})
df_roi  = su.df_geo_slice(df, gdf_roi)

site_counts = df_roi.groupby(["State Code","County Code","Site Num"])["Sample Measurement"].count()
state, county, site = site_counts.idxmax()
df_site = df_roi[(df_roi["State Code"]==state) & (df_roi["County Code"]==county) & (df_roi["Site Num"]==site)].copy()

site_lat = float(df_site["Latitude"].iloc[0])
site_lon = float(df_site["Longitude"].iloc[0])

df_site["UTC"] = pd.to_datetime(df_site["Date GMT"] + " " + df_site["Time GMT"], utc=True)
df_site.set_index("UTC", inplace=True)
epa_o3 = df_site["Sample Measurement"] * 1000.0   # ppm → ppb

# 3. Fetch GEOS-CF (with disk cache)
geos = GEOS_CF()
geos.fetch(lat=site_lat, lon=site_lon,
           start_date="2025-01-01", end_date="2025-12-31",
           collection="aqc", mode="assim", variables=["o3"],
           cache_dir=".geos_cf_cache")

key = next(iter(geos.data))
geos_o3_raw = geos.data[key]["o3"].squeeze("lev").to_series()

# Unit detection: values in mol/mol will have median < 0.01
if geos_o3_raw.median() < 0.01:
    geos_o3 = geos_o3_raw * 1e9
else:
    geos_o3 = geos_o3_raw

# Align timestamps: GEOS-CF :30 → :00
geos_o3.index = geos_o3.index.floor("h")
if geos_o3.index.tz is None:
    geos_o3.index = geos_o3.index.tz_localize("UTC")

# 4. Merge and analyse
merged = pd.concat([epa_o3.rename("EPA_ppb"), geos_o3.rename("GEOS_ppb")], axis=1).dropna()
correlation = merged.corr().loc["EPA_ppb", "GEOS_ppb"]
print(f"N={len(merged)}, r={correlation:.3f}")
```

### Fetch multiple collections for the same site

```python
geos = GEOS_CF()
site = dict(lat=39.055, lon=-76.878, start_date="2025-07-01", end_date="2025-07-31")

geos.fetch(**site, collection="aqc", variables=["o3", "no2"], cache_dir=".cache")
geos.fetch(**site, collection="met", variables=["t2m", "zpbl"], cache_dir=".cache")

for key, ds in geos.data.items():
    print(key, ds.dims)
```

### Check for download errors

```python
if geos._errors:
    for err in geos._errors:
        print("ERROR:", err)
```
