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
| `curtain` | matplotlib | Time-altitude curtain from a DataFrame (time index, altitude columns) or a dict of them such as `TOLNET.data[key]` |
| `profile` | matplotlib | Vertical profiles with optional uncertainty bands |
| `windrose` | matplotlib (windrose) | Wind rose from direction and speed |
| `site_map` | plotly | Interactive map of site locations |

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
