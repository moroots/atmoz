"""The ``plot`` entry point: ``plot("curtain", df)`` and ``plot.curtain(df)`` run the same function.

Plot functions are listed by name and imported on first use, so ``import atmoz.plotting`` stays light and the
optional backends (plotly, windrose) are only needed by the plots that use them.

Matplotlib plots return ``(fig, ax)``; plotly plots return the plotly ``Figure``.
"""

from __future__ import annotations

import difflib
import functools
from importlib import import_module
from typing import Callable

from atmoz.plotting.styles import _DEFAULT, Style, resolve_style

# plot name -> (module, function, backend, one-line description)
_PLOTS: dict[str, tuple[str, str, str, str]] = {
    "curtain": ("atmoz.plotting.curtain", "curtain", "matplotlib", "Time-altitude curtain (lidar, ceilometer, model)."),
    "profile": ("atmoz.plotting.profile", "profile", "matplotlib", "Vertical profiles with optional uncertainty."),
    "windrose": ("atmoz.plotting.windrose", "windrose", "matplotlib", "Wind rose from direction and speed."),
    "skewt": ("atmoz.met", "skewt", "matplotlib", "Skew-T log-p diagram of a sonde or model profile (needs atmoz[met])."),
    "site_map": ("atmoz.plotting.sitemap", "site_map", "plotly", "Interactive map of site locations."),
}


def styled(default_style: str, backend: str = "matplotlib") -> Callable:
    """Decorator for plot functions.

    Adds the shared ``style=`` and ``figsize=`` arguments, resolves ``style`` to a :class:`Style`, and (for
    matplotlib plots) builds the figure inside ``rc_context`` so the style never leaks into the caller's session.
    The wrapped function receives the resolved ``style`` as a keyword argument.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, style=_DEFAULT, figsize=None, **kwargs):
            resolved = resolve_style(style, default_style)
            if backend == "matplotlib":
                with resolved.context(figsize=figsize):
                    return func(*args, style=resolved, figsize=figsize, **kwargs)
            return func(*args, style=resolved, figsize=figsize, **kwargs)

        wrapper.__atmoz_default_style__ = default_style
        wrapper.__atmoz_backend__ = backend
        return wrapper

    return decorator


class _Plot:
    """Callable namespace of plot functions. See the module docstring."""

    def _load(self, name: str) -> Callable:
        try:
            module, func, _backend, _doc = _PLOTS[name]
        except KeyError:
            close = difflib.get_close_matches(str(name), _PLOTS, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise AttributeError(f"Unknown plot {name!r}.{hint} Available: {', '.join(_PLOTS)}.") from None
        try:
            return getattr(import_module(module), func)
        except ImportError as err:
            raise ImportError(
                f"The {name!r} plot needs an optional package that is not installed ({err}). "
                f"Install it with:  pip install \"atmoz[plots]\""
            ) from err

    def __call__(self, kind: str, *args, **kwargs):
        """``plot("curtain", df, ...)``: same as ``plot.curtain(df, ...)``."""
        return self._load(kind)(*args, **kwargs)

    def __getattr__(self, name: str) -> Callable:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._load(name)

    def __dir__(self):
        return sorted(_PLOTS)

    def __iter__(self):
        return iter(_PLOTS)

    def available(self) -> dict[str, str]:
        """``{name: description}`` of every plot."""
        return {name: f"[{backend}] {doc}" for name, (_m, _f, backend, doc) in _PLOTS.items()}

    def __repr__(self) -> str:
        lines = [f"  plot.{name:<9} {desc}" for name, desc in self.available().items()]
        return "atmoz plots:\n" + "\n".join(lines)


plot = _Plot()

__all__ = ["plot", "styled", "Style"]
