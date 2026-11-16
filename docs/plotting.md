# Plotting

atmoz plots look right with no arguments. Style is built in, so figures from different datasets and authors look
alike.

```python
import atmoz as az

fig, ax = az.plot("curtain", df)        # by name ...
fig, ax = az.plot.curtain(df)           # ... or as a function; they are the same
az.plot                                 # lists every plot
az.plotting.styles                      # lists every style
```

Matplotlib plots return `(fig, ax)`, so you can annotate, save (`fig.savefig("out.png")`) or put the figure in a
report. In a notebook, write `fig, ax = az.plot.curtain(df)` and the figure is displayed once. Plotly plots (the
interactive ones) return the plotly `Figure`.

## Plots

| Plot | Backend | What it draws |
|---|---|---|
| `curtain` | matplotlib | Time-altitude curtain from a DataFrame (time index, altitude columns) or a dict of them |
| `profile` | matplotlib | Vertical profiles with optional uncertainty bands |
| `comparison_profile` | matplotlib | Mean profiles of every source of a collocated frame, with the spread (1 standard deviation) shaded |
| `exceedances` | matplotlib | Days above the ozone standard per site, as bars, from the table of `atmoz.stats.exceedances` |
| `skewt` | matplotlib | Skew-T log-p diagram of a sonde or model profile (also `atmoz.met.skewt`; needs `atmoz[met]`) |
| `windrose` | matplotlib (windrose) | Wind rose from direction and speed |
| `site_map` | plotly | Interactive map of site locations |

`curtain` takes the wide form, a DataFrame with time as the index and altitude as the columns, or a long profile `AtmozFrame` (what
TOLNet returns: `alt` and `value` columns), which it pivots for you:

```python
frame = ds[next(iter(ds))]                                  # TOLNet frame: UTC time index; alt, value
fig, ax = az.plot.curtain(frame)                            # same as pivoting yourself:
wide = frame.df.pivot(columns="alt", values="value")
fig, ax = az.plot.curtain(wide)
```

## Comparing sources

```python
out = az.collocate(lidar, model, sonde, names=["lidar", "geos_cf", "sonde"])
fig, ax = az.plot.comparison_profile(out)                   # mean profile per source, shaded standard deviation
fig, ax = az.plot.comparison_profile(out, columns=["lidar", "geos_cf"], bins=0.5, ylim=(0, 6))

table = az.stats.exceedances(aqs_ds)                        # per-site days above 70 ppb MDA8
fig, ax = az.plot.exceedances(table, top=15)                # bars, most exceedance days first
```

`comparison_profile` takes the result of [`atmoz.collocate`](collocation.md) for profiles (a frame, or a dataset of them, pooled); `columns=` picks
the sources (default: all recorded in `metadata["columns"]`), `bins=` is the altitude bin width in km (default 1) or the edges, and the axis is
labelled with the altitude reference. `exceedances` takes the table from [`atmoz.stats.exceedances`](analysis.md); `top=` limits the bars
(default 25, `None` for all) and `level_ppb=` only sets the default title.

## Skew-T

```python
fig, ax = az.plot.skewt(sonde, dewpoint="dewpoint", wind=("u", "v"))      # same as az.met.skewt
```

Needs `pip install "atmoz[met]"`; the columns and units it expects are described in [Meteorology](met.md).

`TOLNET.plot_curtain(frame)` and `TOLNET.plot_curtains(ds)` do the pivot for you (and add the TOLNet watermark). GEOS-CF layer heights differ
per time, so pivot its profiles on `lev` or collocate them onto a lidar's altitudes with [`atmoz.collocate`](collocation.md) first.

`windrose` and `site_map` need the optional plotting packages: `pip install "atmoz[plots]"`.

## Styles

```python
az.plot.curtain(df, style=az.plotting.styles.curtain)       # a Style object (this is the default)
az.plot.curtain(df, style="curtain")                        # the same, by name
az.plot.curtain(df, style={"axes.titlesize": 18})           # your rcParams, merged over the default
az.plot.curtain(df, style=None)                             # no atmoz styling: your matplotlib settings apply
az.plot.curtain(df, figsize=(8, 3))                         # size only
az.plotting.styles.curtain.replace(figsize=(8, 3))          # a derived Style you can reuse
```

Style is applied **per call**. atmoz never changes your global `matplotlib.rcParams`, so your own plots are not
affected by importing atmoz or calling its plots.

A `Style` bundles rcParams (`rc`), the colour cycle (`colors`, colour-blind-safe), a default `figsize`, artist-level
`params`, and `plotly` layout options. Styles are immutable; use `replace`, `with_rc` or `with_params` to derive one.

## Fonts

atmoz ships **Inter** (SIL Open Font License, license file included in `atmoz/assets/fonts/`) and registers it with
matplotlib the first time you plot, so plots look the same on every machine, Colab included. DejaVu Sans, which comes
with matplotlib, is the fallback. Any other `.ttf` / `.otf` placed in that folder is registered too.

## Adding a variable to `curtain`

`curtain` picks its colour scale, labels and title from a small registry (`_VARIABLES` in
`atmoz/plotting/curtain.py`). Add an entry to teach it a new variable; `variable=None` falls back to viridis.
