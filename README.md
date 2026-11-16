<p align="center">
  <img src="docs/assets/logo.svg" alt="atmoz" width="180">
</p>

# Atmoz

**Atmoz** is a Python toolkit for handling, analysing and visualising ground-based atmospheric datasets.
It builds on the earlier [DATAS](https://github.com/moroots/DATAS) package, with a more modular, faster
and lighter design. It is a dev / science tool: it makes common things fast and stays out of your way for the rest.

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Credentials and caching](#credentials-and-caching)
* [Documentation](#documentation)
* [Directory structure](#directory-structure)
* [Development](#development)
* [License](#license)

## Features

* **EPA surface air quality** (`atmoz.surface`): the live **AirNow** API and the bulk **EPA AQS** yearly files. `mda8` / `mda8_aqs` compute ozone 8-hour averages and the daily maximum (MDA8) per EPA rules.
  Streamed to a disk cache, parsed with Arrow's multi-threaded CSV reader (full-precision `float64`
  coordinates, `category` text columns), downloaded in parallel, all-or-nothing on failure.
* **TOLNet ozone lidar** (`atmoz.lidar`): query, download and plot NASA TOLNet profiles.
* **UCN ceilometers** (`atmoz.lidar`): Unified Ceilometer Network 10-minute backscatter and boundary-layer data, cached compactly as parquet. L0b (raw 16 s profiles) is planned.
* **GEOS-CF model output** (`atmoz.models`): point extractions through OPeNDAP, returned as frames (surface series or long time x level profiles) and cached.
* **Balloon sondes** (`atmoz.Sonde`): NASA LaRC STAQS ozonesonde and windsonde ICARTT files (downloaded by hand) read into an `AtmozDataset`, with a check of the declared ozone unit.
* **Data classes** (`atmoz.dataclasses`): every `download()` returns an `AtmozDataset` of `AtmozFrame`s (UTC time index, `lat` / `lon` / `alt` columns, units in `frame.units`, metadata).
* **Collocation** (`atmoz.collocate`): put a lidar, a model, a sonde or a surface network on the first source's times, altitudes and unit, with altitude references (asl / agl) reconciled.
* **Analysis** (`atmoz.stats`, `atmoz.vertical`): `compare` (bias, RMSE, r, by altitude / hour / month), `exceedances`, `diurnal`, `trend`; `layer_mean`, `mean_profile`.
* **Meteorology** (`atmoz.met`, extra `met`): potential temperature, dewpoint, wind components, boundary-layer height and Skew-T plots.
* **Plots** (`atmoz.plot`): curtain, profile, comparison profile, exceedances, wind rose, site map and Skew-T, with one built-in style.
* **Shared plumbing**: thread-safe HTTP sessions with retries, a file cache, a `.atmoz.env` secret store.

## Installation

Requires **Python 3.11+**.

```bash
pip install git+https://github.com/moroots/atmoz                       # core (AirNow / EPA AQS, data classes)
pip install "atmoz[surface,lidar] @ git+https://github.com/moroots/atmoz"   # add extras as needed
pip install "atmoz[met] @ git+https://github.com/moroots/atmoz"            # meteorology (MetPy): atmoz.met, plot.skewt
```

Extras: `surface` (maps / geo helpers), `lidar`, `models`, `plots`, `met`, `nasa`, `legacy`, `all`, `dev`.
See the [installation guide](docs/installation.md).

## Quick start

```python
from atmoz.surface import AirNow

# Bulk EPA AQS: hourly ozone for Maryland in 2025. An AtmozDataset with one AtmozFrame per site and parameter.
ds = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025, state="Maryland")
frame = ds["240338003/44201"]                 # key: "{state}{county}{site}/{parameter code}"
frame.df.head()                               # UTC time index; value, lat, lon, poc, local_time, ...
frame.units                                   # {"value": "ppm", ...}; frame.convert_units("value", "ppb") converts
ds.inventory()                                # one row per frame: lat, lon, unit, time span

# Only what you need: a bounding box and a date range
ds_roi = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025,
                         bbox=(-77.5, 38.5, -76.5, 39.5), start_date="2025-06-01", end_date="2025-08-31")

# Live AirNow API (free key from https://docs.airnowapi.org/): save it once, use it forever
AirNow.set_api_key()
ds = AirNow.download(endpoint="airnow", start_date="2025-08-01", end_date="2025-08-07", parameters=["OZONE", "PM25"])
ds.select("*/OZONE")                          # frames keyed "{IntlAQSCode}/{PARAMETER}"
ds.resample("1h").mean()                      # pandas methods apply to every frame
```

```python
import atmoz
from atmoz.lidar import TOLNET            # pip install "atmoz[lidar]"
from atmoz.models import GEOS_CF          # pip install "atmoz[models]"

ds = TOLNET.download("2025-07-28", "2025-07-28", product_type=[4], processing_type=[1])   # an AtmozDataset, one frame per file
TOLNET.plot_curtains(ds)                                                                      # one curtain per instrument

# Compare with the model at the instrument's location (the first source is the anchor: its times, altitudes and unit)
obs = ds[next(iter(ds))]
lat, lon = obs.location
model = GEOS_CF.download(lat, lon, "2025-07-28", "2025-07-30", collection="chm", variables=["o3"])["o3"]
out = atmoz.collocate(obs, model, names=["lidar", "geos_cf"])
out.df[["alt", "lidar", "geos_cf"]]
atmoz.stats.compare(out, by="alt")                  # n, bias, mae, rmse, nmb_percent, r per 1 km bin
atmoz.plot.comparison_profile(out)                  # mean profiles with their spread
```

```python
# A balloon sonde (STAQS .ict files are downloaded by hand), compared with the lidar and the model
sonde = atmoz.Sonde.read("data/Sondes/flight.ict").frame   # alt (km asl), value (ozone, ppb), lat / lon per row
atmoz.collocate(obs, model, sonde, names=["lidar", "geos_cf", "sonde"], window="1h")

# Ozone exceedances from the AQS dataset of the first example
table = atmoz.stats.exceedances(ds_roi)             # per site: days above 70 ppb MDA8, 4th highest, ...
atmoz.plot.exceedances(table)
```

## Credentials and caching

* **API keys** live in a `.atmoz.env` file (`~/.atmoz/.atmoz.env` for you, or `./.atmoz.env` per project) or the
  `ATMOZ_AIRNOW_API_KEY` environment variable. `AirNow.set_api_key()` writes it for you. atmoz never prompts for a
  missing key and never puts it in an error message. See [Configuration](docs/configuration.md).
* **Downloads** are cached in `~/.atmoz/cache` (`atmoz.cache_info()`, `atmoz.clear_cache()`). See [Caching](docs/caching.md).

## Documentation

Guides live in [`docs/`](docs) (build them with `mkdocs serve`):
[Installation](docs/installation.md) · [Configuration](docs/configuration.md) · [Caching](docs/caching.md) ·
[Conventions](docs/conventions.md) · [Surface](docs/surface.md) · [Lidar](docs/lidar.md) · [Models](docs/models.md) ·
[Sondes](docs/sonde.md) · [Collocation](docs/collocation.md) · [Analysis](docs/analysis.md) · [Meteorology](docs/met.md) ·
[Plotting](docs/plotting.md). Release notes: [CHANGELOG](CHANGELOG.md).

## Directory structure

<!-- START FILETREE -->
```
atmoz/
├── assets/
│   ├── package_info/
│   └── watermarks/
├── data_access/
│   └── NASA.py
├── collocation.py         # collocate(): several sources onto the first one's times, altitudes and unit
├── dataclasses/
│   ├── AtmozDataset.py
│   └── AtmozFrame.py
├── lidar/
│   ├── tolnet.py
│   └── ucn.py             # Unified Ceilometer Network
├── met.py                 # theta, dewpoint, wind components, PBL height, Skew-T (extra met)
├── models/
│   └── geos_cf.py
├── plotting/              # atmoz.plot: curtain, profile, comparison_profile, exceedances, windrose, site_map
├── resources/
│   ├── atmoz_dataclasses.py
│   ├── cache.py
│   ├── credentials.py
│   ├── icartt.py          # ICARTT (FFI 1001) reader
│   ├── parallelExecutor.py
│   ├── sessionHandler.py
│   ├── urls.py
│   ├── utilities.py       # shared download plumbing
│   └── ... (plotting, time and path helpers)
├── sonde.py               # Sonde.read(): STAQS ICARTT sondes
├── stats.py               # compare, exceedances, diurnal, trend
├── surface/
│   ├── AirNow.py          # AirNow facade: download() / fetch() / set_api_key()
│   ├── AirNow_API.py      # AirNowAPI (live API)
│   ├── AQS.py             # EPA_PreGen (yearly files), read_aqs_zip
│   └── utilities.py
├── vertical.py            # layer_mean, mean_profile
└── exceptions.py
```
<!-- END FILETREE -->

## Development

```bash
uv venv && uv pip install -e ".[all,dev]"
pytest                      # offline: HTTP is faked
ruff check atmoz tests
```

CI (GitLab and GitHub Actions) runs the tests on Python 3.11 - 3.13, lints, and checks the built wheel.

## License

Copyright (c) 2025 Maurice Roots

Atmoz is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).
