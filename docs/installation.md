# Installation

atmoz needs **Python 3.11 or newer**.

```bash
pip install git+https://github.com/moroots/atmoz
```

The core install is deliberately light (pandas, numpy, pyarrow, requests, matplotlib, pint, tqdm). It already
contains the AirNow / EPA AQS downloaders and the data classes. Add extras for the rest:

| Extra | Adds | Needed for |
|-------|------|-----------|
| `surface` | geopandas, shapely, plotly | `atmoz.surface.utilities` (GeoDataFrames, site maps) |
| `lidar` | pydantic, pyyaml, python-dateutil, earthaccess | `atmoz.lidar.TOLNET` |
| `models` | pydantic, xarray, netcdf4 | `atmoz.models.GEOS_CF` |
| `nasa` | pydantic, earthaccess | `atmoz.data_access.NASA` |
| `legacy` | geopandas, h5py, windrose, beautifulsoup4, ... | DATAS-era helpers in `atmoz.resources` |
| `all` | everything above | |
| `dev` | pytest, ruff, build, mkdocs | working on atmoz |

```bash
pip install "atmoz[surface,lidar] @ git+https://github.com/moroots/atmoz"
```

If you import a module whose extra is missing, the error tells you which one to install.

## Development

```bash
git clone <repo> && cd atmoz
uv venv && uv pip install -e ".[all,dev]"
pytest            # no network needed: HTTP is faked
ruff check atmoz tests
mkdocs serve      # docs at http://127.0.0.1:8000
```

`uv.lock` is not tracked; the supported dependency ranges live in `pyproject.toml`.
