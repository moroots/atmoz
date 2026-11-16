# atmoz.models

Access NASA [GEOS-CF](https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/) (Goddard Earth Observing System Composition Forecast) chemistry-climate model output at any global site via OPeNDAP.

GEOS-CF provides hourly atmospheric chemistry, composition, and meteorology fields at 0.25° × 0.3125° resolution from 2018 onward, as both a replay (assimilation) product and a rolling 5-day forecast.

```python
from atmoz.models import GEOS_CF
```

Install the extra first: `pip install "atmoz[models]"`.

---

## Quick Start

```python
from atmoz.models import GEOS_CF

# Surface air quality at a site -> an atmoz_dataset (cached on disk, all-or-nothing)
ds = GEOS_CF.download(39.055, -76.878, "2025-01-01", "2025-01-31", variables=["o3"])
o3 = ds.data["aqc_assim"]["o3"]              # pandas Series, UTC time index, ppbv
print(f"O3 range: {o3.min():.1f} - {o3.max():.1f} ppbv")

# Chemistry profiles: one time x level DataFrame per variable
prof = GEOS_CF.download(39.055, -76.878, "2025-07-28", "2025-08-01", collection="chm", variables=["o3"])
prof.data["chm_assim_o3"]                    # index: UTC time, columns: model level
```

`GEOS_CF.fetch(...)` takes the same request but only fills the cache and returns `{file_name: parquet_path}`.
See the [conventions](conventions.md): it works like every other atmoz source.

---

## Collections (`_COLLECTION_CATALOG`)

Three collections are available. Pass the collection name as the `collection` argument to `download()` / `fetch()`.

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
GEOS_CF.catalog["aqc"]["variables"]      # the full dict: collections, variables, OPeNDAP URLs
```

---

## Result shape and units

`download()` returns an `atmoz_dataset` (`datatype="model"`):

| Collection | `ds.data` | Each DataFrame |
|------------|-----------|----------------|
| `aqc`, `met` (surface) | `{"aqc_assim": DataFrame}` | UTC `time` index, one column per variable |
| `chm` (profiles) | `{"chm_assim_o3": DataFrame, ...}` | UTC `time` index, one column per model level (`lev`), one frame per variable |

`ds.metadata` has `lat` / `lon` (the 0.25° grid cell actually used), `lat_requested` / `lon_requested`, `units`, `start`, `end`,
`collection`, `mode`. Each DataFrame's `.attrs` repeats the grid cell and the units. A profile frame has the same shape the
TOLNet curtain plot takes (time index, level columns), so it can be plotted next to a lidar curtain.

**Units.** The server gives gas species (`o3`, `no2`, `co`, `so2`, `oh`, `hno3`) in mol/mol. `download()` converts them to
**ppbv** by default; pass `units="mol/mol"` for the raw values. PM2.5 (`pm25_rh35_gcc`, µg/m³) and the met variables
(K, m/s, kg/kg, m, Pa) are never converted. `fetch()` files are always raw.

---

## `download(lat, lon, start_date, end_date, ...)`

```python
ds = GEOS_CF.download(
    39.055, -76.878, "2025-01-01", "2025-12-31",
    collection="aqc", mode="assim", variables=["o3", "no2"],
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `lat`, `lon` | `float` | required | Site, decimal degrees. The nearest 0.25° grid cell is used. |
| `start_date`, `end_date` | `str \| date` | required | Inclusive UTC days: `"YYYY-MM-DD"`, `"YYYYMMDD"`, `date` or `datetime`. |
| `collection` | `str` | `"aqc"` | `"aqc"`, `"chm"` or `"met"`. |
| `mode` | `str` | `"assim"` | `"assim"` (historical) or `"fcast"` (forecast, `aqc` only). |
| `variables` | `list[str]` | collection defaults | Lower-case names; an unknown one fails the request. |
| `units` | `str` | `"ppbv"` | `"ppbv"` or `"mol/mol"` (see above). |
| `max_workers` | `int` | `1` | Months fetched at once. Server access is serialized anyway (the netCDF libraries are not thread-safe). |
| `use_cache`, `refresh`, `cache_dir` | | `True`, `False`, shared cache | See below. |
| `silent`, `show_traceback` | `bool` | `False` | Progress bar / full tracebacks. |

**Raises** `ValueError` for a bad request (before any network call) and `DownloadError` if any month fails; nothing is returned,
but finished months stay cached.

### Monthly requests

The OPeNDAP server caps the size of a time slice, so a range is split into calendar months, one request each. A month can take
around a minute depending on server load, so a full year for one site may take 10+ minutes the first time (and is instant afterwards).

### Cache

On by default, in the shared atmoz cache (`~/.atmoz/cache/geos_cf`; `atmoz.cache_info()` shows its size, `atmoz.clear_cache("geos_cf")`
empties it). One parquet file per month of data, keyed by the grid cell, collection, mode, dates and variables, so sites in the same
grid cell share files. Assimilation data older than two weeks is treated as final; newer data and all forecasts are re-downloaded
after three hours. `refresh=True` forces a download, `use_cache=False` ignores the cache.

### `GEOS_CF_QUERY`

The request is validated by a Pydantic model before any network call: `lat` in [-90, 90], `lon` in [-180, 180], `start_date <= end_date`,
and `mode` must exist for the collection (`chm` and `met` are `assim` only).

---

## Changes from the old API

`GEOS_CF().fetch(...)` (which stored xarray Datasets in `self.data`), `force_download` and `add_altitude_coords` are gone. Use
`GEOS_CF.download(...)`; convert to xarray yourself with `df.to_xarray()` if you want it. To compare with TOLNet, call
`GEOS_CF.download(...)` at the instrument's coordinates (`ds.metadata["info"]` of a TOLNet dataset has them).
