"""Plot styles: one object per kind of plot, applied per call, never globally.

A :class:`Style` bundles everything that decides how a plot *looks*:

``rc``
    matplotlib rcParams, applied with ``rc_context`` while the figure is built.
    Importing or using atmoz never changes ``matplotlib.rcParams`` for your own plots.
``colors``
    the colour cycle for lines and bars.
``figsize``
    default figure size in inches (override per call with ``figsize=``).
``params``
    artist-level settings, e.g. ``{"ax.set_ylim": [0, 15]}``. Keys starting with ``ax.`` / ``fig.`` call that
    method; other keys are read by the plot function that owns them (colorbar options, for example).
``plotly``
    layout options for the interactive (plotly) plots.

This module is plain Python on purpose: ``import atmoz.plotting`` does not import matplotlib or plotly.

    >>> from atmoz.plotting import styles
    >>> styles.curtain                       # a Style
    >>> styles.curtain.replace(figsize=(8, 3))
    >>> sorted(styles)                       # names of every registered style
"""

from __future__ import annotations

import copy
import difflib
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as _replace
from typing import Any, Iterator, Mapping, Optional

__all__ = ["Style", "styles", "resolve_style", "PALETTE"]

# Okabe-Ito: distinguishable for the common kinds of colour-vision deficiency.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#000000",  # black
)

# Inter ships with atmoz (``atmoz/assets/fonts``, SIL Open Font License) and is registered before the first plot.
# DejaVu Sans ships with matplotlib and is the fallback, so text always renders. This is an explicit list, not the
# "sans-serif" alias, because matplotlib expands the alias when the figure is *drawn* (after the style context has
# ended); an explicit list is fixed when the text is created.
_FONT_STACK = ["Inter", "DejaVu Sans"]

_BASE_RC: dict[str, Any] = {
    "font.family": _FONT_STACK,
    "font.size": 10.5,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    # Layout is fixed when the figure is created (inside the style context), so titles, colorbars and legends
    # stay inside the figure however and wherever it is saved afterwards.
    "figure.constrained_layout.use": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "axes.labelcolor": "#222222",
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "image.cmap": "viridis",
}

_OZONE_COLORBAR_TICKS = [0.001, *range(10, 121, 10), 150, 200, 300, 600]


@dataclass(frozen=True)
class Style:
    """How a plot looks. Immutable: use :meth:`replace` / :meth:`with_rc` to derive a variant."""

    name: str
    rc: Mapping[str, Any] = field(default_factory=dict)
    colors: tuple = ()
    figsize: Optional[tuple] = None
    params: Mapping[str, Any] = field(default_factory=dict)
    plotly: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def replace(self, **changes) -> "Style":
        """Copy of this style with some fields changed, e.g. ``style.replace(figsize=(8, 3))``."""
        return _replace(self, **changes)

    def with_rc(self, rc: Mapping[str, Any], name: Optional[str] = None) -> "Style":
        """Copy of this style with extra rcParams merged over the existing ones."""
        return _replace(self, name=name or f"{self.name}+custom", rc={**self.rc, **rc})

    def with_params(self, params: Mapping[str, Any]) -> "Style":
        """Copy of this style with artist-level params merged (nested dicts merge, others replace)."""
        return _replace(self, params=_deep_merge(self.params, params))

    def rc_params(self, figsize: Optional[tuple] = None) -> dict[str, Any]:
        """The rcParams dict to hand to ``matplotlib.rc_context``. Needs matplotlib (imported here, lazily)."""
        out = dict(self.rc)
        if isinstance(out.get("font.family"), (list, tuple)):
            from atmoz.plotting.fonts import available_families
            out["font.family"] = available_families(out["font.family"])
        size = figsize or self.figsize
        if size:
            out["figure.figsize"] = tuple(size)
        if self.colors:
            from cycler import cycler
            out["axes.prop_cycle"] = cycler(color=list(self.colors))
        return out

    @contextmanager
    def context(self, figsize: Optional[tuple] = None) -> Iterator[None]:
        """Apply this style to matplotlib for the duration of a ``with`` block only."""
        import matplotlib as mpl

        from atmoz.plotting.fonts import register_fonts
        register_fonts()
        with mpl.rc_context(self.rc_params(figsize)):
            yield


def _deep_merge(base: Mapping, over: Mapping) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class _Styles:
    """Registry of named styles: ``styles.curtain``, ``styles["curtain"]``, ``sorted(styles)``."""

    def __init__(self) -> None:
        object.__setattr__(self, "_items", {})

    def register(self, style: Style) -> Style:
        self._items[style.name] = style
        return style

    def __getitem__(self, name: str) -> Style:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(_unknown("style", name, self._items)) from None

    def __getattr__(self, name: str) -> Style:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(_unknown("style", name, self._items)) from None

    def __setattr__(self, name, value):
        raise AttributeError("styles is read-only; use styles.register(Style(...)) to add one")

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, name) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __dir__(self):
        return sorted(self._items) + ["register"]

    def __repr__(self) -> str:
        return "styles(" + ", ".join(sorted(self._items)) + ")"


def _unknown(kind: str, name: str, options) -> str:
    close = difflib.get_close_matches(str(name), list(options), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"Unknown {kind} {name!r}.{hint} Available: {', '.join(sorted(options))}."


styles = _Styles()

# ---- the styles ---------------------------------------------------------------------------------

atmoz_style = styles.register(Style(
    name="atmoz",
    rc=_BASE_RC,
    colors=PALETTE,
    figsize=(7.5, 4.2),
    description="Base look: clean axes, light grid, colour-blind-safe cycle.",
))

styles.register(atmoz_style.replace(
    name="curtain",
    figsize=(11, 4.4),
    rc={**_BASE_RC, "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.55,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5},
    params={
        "ax.set_ylabel": {"ylabel": "Altitude (km ASL)"},
        "ax.set_ylim": [0, 15],
        "ax.set_yticks": {"ticks": list(range(0, 16))},
        "colorbar": {
            "pad": 0.012,
            "ticks": _OZONE_COLORBAR_TICKS,
            "label": {"weight": "bold", "size": 11},
            "labelsize": 9,
        },
    },
    description="Time-altitude curtains (lidar, ceilometer, model profiles).",
))

styles.register(atmoz_style.replace(
    name="profile",
    figsize=(4.2, 5.6),
    params={"ax.set_ylabel": {"ylabel": "Altitude (km ASL)"}},
    description="Vertical profiles: x = quantity, y = altitude.",
))

styles.register(atmoz_style.replace(
    name="exceedances",
    figsize=(8, 4.2),
    description="Bar charts of exceedance days per site.",
))

styles.register(atmoz_style.replace(
    name="skewt",
    figsize=(6.5, 6.5),
    description="Skew-T log-p diagrams (MetPy).",
))

styles.register(atmoz_style.replace(
    name="windrose",
    figsize=(7.2, 5.6),
    rc={**_BASE_RC, "axes.grid": True, "axes.spines.top": True, "axes.spines.right": True},
    description="Wind roses (windrose library).",
))

styles.register(atmoz_style.replace(
    name="timeseries",
    figsize=(9, 3.6),
    description="Time series of one or more variables.",
))

styles.register(atmoz_style.replace(
    name="sitemap",
    figsize=None,
    plotly={
        "height": 650,
        "width": None,
        "font": {"family": "Inter, Arial, sans-serif", "size": 14},
        "title_font_size": 20,
        "map_style": "carto-positron",
        "marker_size": 11,
        "legend": {"bgcolor": "rgba(255,255,255,0.8)", "x": 0.01, "y": 0.99, "itemsizing": "constant"},
    },
    description="Interactive (plotly) map of site locations.",
))


def resolve_style(style, default: str) -> Style:
    """Turn the ``style=`` argument of a plot function into a :class:`Style`.

    ``default`` (not passed)  -> the plot's own named style
    a name                    -> that registered style
    a :class:`Style`          -> used as is
    a dict                    -> rcParams merged over the plot's own style
    ``None``                  -> no atmoz styling: your current matplotlib settings apply
    """
    if style is _DEFAULT:
        return styles[default]
    if style is None:
        return Style(name="none")
    if isinstance(style, Style):
        return style
    if isinstance(style, str):
        return styles[style]
    if isinstance(style, Mapping):
        return styles[default].with_rc(style)
    raise TypeError(f"style must be a Style, a style name, a dict of rcParams or None, not {type(style).__name__}")


class _Default:
    def __repr__(self) -> str:
        return "<plot's own style>"


_DEFAULT = _Default()
