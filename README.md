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
* **Data classes** (`atmoz.dataclasses`): every `download()` returns an `AtmozDataset` of `AtmozFrame`s (UTC time index, `lat` / `lon` / `alt` columns, units in `frame.units`, metadata). `atmoz.collocate` puts model output on a lidar's times and altitudes.
* **Shared plumbing**: thread-safe HTTP sessions with retries, a file cache, a `.atmoz.env` secret store.

## Installation

Requires **Python 3.11+**.

```bash
pip install git+https://github.com/moroots/atmoz                       # core (AirNow / EPA AQS, data classes)
pip install "atmoz[surface,lidar] @ git+https://github.com/moroots/atmoz"   # add extras as needed
```

Extras: `surface` (maps / geo helpers), `lidar`, `models`, `nasa`, `legacy`, `all`, `dev`.
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

# Compare with the model at the instrument's location
obs = ds[next(iter(ds))]
lat, lon = obs.location
model = GEOS_CF.download(lat, lon, "2025-07-28", "2025-07-29", collection="chm", variables=["o3"])["o3"]
atmoz.collocate(obs, model).df[["alt", "obs", "model"]]
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
[Collocation](docs/collocation.md) · [Plotting](docs/plotting.md). Release notes: [CHANGELOG](CHANGELOG.md).

## Directory structure

<!-- START FILETREE -->
```
atmoz/
├── assets/
│   ├── package_info/
│   └── watermarks/
├── data_access/
│   └── NASA.py
├── collocation.py         # collocate(): model onto observation times and altitudes
├── dataclasses/
│   ├── AtmozDataset.py
│   └── AtmozFrame.py
├── lidar/
│   ├── tolnet.py
│   └── ucn.py             # Unified Ceilometer Network
├── models/
│   └── geos_cf.py
├── resources/
│   ├── atmoz_dataclasses.py
│   ├── cache.py
│   ├── credentials.py
│   ├── parallelExecutor.py
│   ├── sessionHandler.py
│   ├── urls.py
│   ├── utilities.py       # shared download plumbing
│   └── ... (plotting, time and path helpers)
├── surface/
│   ├── AirNow.py          # AirNow facade: download() / fetch() / set_api_key()
│   ├── AirNow_API.py      # AirNowAPI (live API)
│   ├── AQS.py             # EPA_PreGen (yearly files), read_aqs_zip
│   └── utilities.py
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
