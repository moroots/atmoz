# Sondes

`atmoz.Sonde` reads balloon-sonde files in the ICARTT format (format index 1001), such as the ozonesondes and windsondes of the
NASA LaRC **STAQS** campaign, into an `AtmozDataset` (see [Conventions](conventions.md#containers)).

```python
import atmoz as az

ds = az.Sonde.read("data/Sondes")                  # a .ict file, a directory of .ict files, or a list of paths
flight = ds["staqs-..._ozonesonde.ict"]            # one AtmozFrame per file (a flight), keyed by the file name
flight.df[["alt", "value", "lat", "lon"]]
flight.units                                       # {"alt": "km", "value": "ppb", "Pressure_hPa": "hPa", ...}
ds.inventory()                                     # one row per flight
```

## Getting the files

The STAQS sondes are listed at <https://www-air.larc.nasa.gov/cgi-bin/ArcView/staqs?SONDE=1>. Download the `.ict` files by hand
and point `Sonde.read` at the file or folder: atmoz has no download automation for them. `Sonde.read` takes a path, a directory
(all `*.ict` files in it) or a list of paths, and raises `FileNotFoundError` when a directory holds no `.ict` file and `ValueError`
for a file that is not ICARTT 1001 or has no time or altitude column.

## What a frame holds

| Item | Content |
|------|---------|
| index | UTC time of every reading. A sonde is a trajectory: each row has its own time (seconds since the file's start date; a clock that wraps past midnight is continued). |
| `alt` | altitude in km **above sea level** (`Altitude_km`, or `Altitude_m_MSL` converted); `frame.alt_reference == "asl"` |
| `alt_agl` | altitude above ground in km, only when the file has `Altitude_m_AGL` |
| `value` | ozone in **ppb**, for ozonesonde files (from `Ozone_ppmv` / `Ozone_ppbv`). Windsonde files have no `value`. |
| `lat`, `lon` | position per row (the balloon drifts); NaN columns when the file has no GPS. `frame.location` is the median position, or `None` without GPS. |
| other columns | every other variable keeps its file name and unit (`Pressure_hPa`, `Temp_degC`, `RH_percent`, `WindSpeed_mps`, `Ozone_mPa`, ...), declared in `frame.units`. |
| `metadata` | `file`, `pi`, `platform`, `dataset`, `mission`, `location`, `revision`, `start_date`, `launch` (first time), `alt_reference="asl"`, `site_elevation_km` (parsed from the file's `LOCATION` line when it gives a height AMSL, else `None`), `variables` and `comments` (the header) |

Missing and limit-of-detection flags in the file become NaN, and rows without an altitude are dropped.

## Ozone unit check

Some files label a ppb column as ppmv. The declared ozone unit is checked against the file's own ozone partial pressure
(`Ozone_mPa`) and pressure (`Pressure_hPa`): ppb is `Ozone_mPa / Pressure_hPa * 1e4`. If the label and the physics disagree,
the unit implied by the physics wins and a `UserWarning` says so. The check needs both columns and at least 10 usable rows;
without them the declared unit is trusted.

## Using sondes

A sonde is a profile at a point in time and space, so it can be collocated with a lidar or a model:

```python
sonde = ds.frame                                           # when the dataset holds one flight
out = az.collocate(lidar, model, sonde, names=["lidar", "geos_cf", "sonde"], window="1h")
```

See [Collocation](collocation.md) for how a trajectory is matched to a profile (the `window` option). Wind and met
variables of a sonde feed [`atmoz.met`](met.md) (`theta`, `dewpoint`, `wind_components`, `pbl_height`, `skewt`).
For a windsonde, name the column to compare: `cols={"sonde": "WindSpeed_mps"}`.
