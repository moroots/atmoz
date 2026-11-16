"""atmoz: atmospheric science toolkit focused on ground-based datasets.

Heavy submodules are imported lazily, so ``import atmoz`` is fast and works with
only the core install. Optional pieces need their extra, e.g.
``pip install "atmoz[lidar]"``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("atmoz")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

# public name -> (module, attribute)
_LAZY = {
    "AirNow": ("atmoz.surface.AirNow", "AirNow"),
    "AtmozFrame": ("atmoz.dataclasses.AtmozFrame", "AtmozFrame"),
    "AtmozDataset": ("atmoz.dataclasses.AtmozDataset", "AtmozDataset"),
    "collocate": ("atmoz.collocation", "collocate"),
    "set_secret": ("atmoz.resources.credentials", "set_secret"),
    "get_secret": ("atmoz.resources.credentials", "get_secret"),
    "delete_secret": ("atmoz.resources.credentials", "delete_secret"),
    "plot": ("atmoz.plotting", "plot"),
    "plotting": ("atmoz.plotting", None),   # the submodule itself: atmoz.plotting.styles.curtain
    "cache_dir": ("atmoz.resources.cache", "cache_dir"),
    "cache_info": ("atmoz.resources.cache", "cache_info"),
    "clear_cache": ("atmoz.resources.cache", "clear_cache"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'atmoz' has no attribute {name!r}") from None
    loaded = import_module(module)
    value = loaded if attr is None else getattr(loaded, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
