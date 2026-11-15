"""Opinionated plots for atmospheric observations.

    import atmoz as az

    fig, ax = az.plot("curtain", df)                       # by name ...
    fig, ax = az.plot.curtain(df)                          # ... or as a function
    fig, ax = az.plot.curtain(df, style=az.plotting.styles.curtain)
    az.plot                                                # lists every plot
    az.plotting.styles                                     # lists every style

Every plot looks right with no arguments. Style is applied per call (never to your global matplotlib settings) and
can be changed with ``style=`` (a name, a ``Style``, a dict of rcParams, or ``None`` for your own settings) and
``figsize=``. Matplotlib plots return ``(fig, ax)``; interactive plotly plots return the plotly figure.
"""

from atmoz.plotting.core import plot, styled
from atmoz.plotting.styles import PALETTE, Style, resolve_style, styles

__all__ = ["plot", "styles", "Style", "styled", "resolve_style", "PALETTE"]
