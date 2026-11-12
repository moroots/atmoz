# atmoz

**atmoz** is a Python toolkit for working with ground-based atmospheric datasets: EPA surface air quality
(AirNow, AQS), NASA TOLNet ozone lidar, and GEOS-CF model output, with common data classes and plotting helpers.
It is a dev / science tool: it makes the common things fast, and gets out of your way for everything else
(every download can return plain files or plain DataFrames, every default can be overridden).

It builds on the earlier [DATAS](https://github.com/moroots/DATAS) package.

## Install

```bash
pip install git+https://github.com/moroots/atmoz                       # core: data classes + AirNow / EPA AQS
pip install "atmoz[lidar] @ git+https://github.com/moroots/atmoz"     # + TOLNet
```

See [Installation](installation.md) for every extra.

## What is in the box

| Module | What it does | Extra |
|--------|--------------|-------|
| [`atmoz.surface`](surface.md) | EPA AQS bulk files and the AirNow API: streamed, cached, parallel | core (`surface` for maps / helpers) |
| [`atmoz.lidar`](lidar.md) | NASA TOLNet ozone-lidar queries, downloads and curtain plots | `lidar` |
| [`atmoz.models`](models.md) | NASA GEOS-CF point extractions via OPeNDAP | `models` |
| `atmoz.data_access` | NASA EarthData login helper | `nasa` |
| `atmoz.dataclasses` | `AtmoFrame` / `AtmoDataset` containers | core |

## Good to know

* **Credentials**: no keyring and no prompts. See [Configuration](configuration.md).
* **Caching**: downloads land in `~/.atmoz/cache`. See [Caching](caching.md).
* **All-or-nothing downloads**: a failed request raises `DownloadError`; you never get a silently partial dataset.

## Project

* Source and issues: <https://github.com/moroots/atmoz>
* [Changelog](https://github.com/moroots/atmoz/blob/main/CHANGELOG.md)
* License: GPL-3.0
