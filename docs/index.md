# atmoz

**atmoz** is a Python toolkit for working with ground-based atmospheric datasets: EPA surface air quality
(AirNow, AQS), NASA TOLNet ozone lidar, ceilometers, balloon sondes and GEOS-CF model output, with common data classes,
collocation, statistics and plotting helpers.
It is a dev / science tool: it makes the common things fast, and gets out of your way for everything else
(every download can be fetched as plain files or used as plain pandas DataFrames through `frame.df`, every default can be overridden).

It builds on the earlier [DATAS](https://github.com/moroots/DATAS) package.

## Install

```bash
pip install git+https://github.com/moroots/atmoz                       # core: data classes + AirNow / EPA AQS
pip install "atmoz[lidar] @ git+https://github.com/moroots/atmoz"     # + TOLNet
pip install "atmoz[met] @ git+https://github.com/moroots/atmoz"       # + meteorology (MetPy)
```

See [Installation](installation.md) for every extra.

## What is in the box

| Module | What it does | Extra |
|--------|--------------|-------|
| [`atmoz.surface`](surface.md) | EPA AQS bulk files and the AirNow API: streamed, cached, parallel | core (`surface` for maps / helpers) |
| [`atmoz.lidar`](lidar.md) | NASA TOLNet ozone-lidar queries, downloads and curtain plots | `lidar` |
| [`atmoz.models`](models.md) | NASA GEOS-CF point extractions via OPeNDAP | `models` |
| `atmoz.data_access` | NASA EarthData login helper | `nasa` |
| [`atmoz.Sonde`](sonde.md) | Balloon sondes (NASA LaRC STAQS ICARTT files) read into an `AtmozDataset` | core |
| `atmoz.dataclasses` | `AtmozFrame` / `AtmozDataset` containers, returned by every `download()` and by `Sonde.read` | core |
| [`atmoz.collocate`](collocation.md) | Put several sources (lidar, model, sonde, surface) on the first one's times, altitudes and unit | core |
| [`atmoz.stats`, `atmoz.vertical`](analysis.md) | `compare`, `exceedances`, `diurnal`, `trend`; `layer_mean`, `mean_profile` | core |
| [`atmoz.met`](met.md) | Potential temperature, dewpoint, wind components, PBL height, Skew-T | `met` |
| [`atmoz.plot`](plotting.md) | Opinionated plots: curtain, profile, comparison profile, exceedances, wind rose, site map, Skew-T | core (`plots`, `met` for some) |

## Good to know

* **Credentials**: no keyring and no prompts. See [Configuration](configuration.md).
* **Caching**: downloads land in `~/.atmoz/cache`. See [Caching](caching.md).
* **One data shape**: every `download()` returns an `AtmozDataset` of `AtmozFrame`s: UTC time index, `lat` / `lon` / `alt` columns, units in `frame.units`. See [Conventions](conventions.md).
* **All-or-nothing downloads**: a failed request raises `DownloadError`; you never get a silently partial dataset.

## Project

* Source and issues: <https://github.com/moroots/atmoz>
* [Changelog](https://github.com/moroots/atmoz/blob/main/CHANGELOG.md)
* License: GPL-3.0
