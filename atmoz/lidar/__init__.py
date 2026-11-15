"""Lidar and ceilometer data access (NASA TOLNet, Unified Ceilometer Network). Requires ``pip install "atmoz[lidar]"``."""

from __future__ import annotations

__all__ = ["TOLNET", "UCN"]

_MODULES = {"TOLNET": "atmoz.lidar.tolnet", "UCN": "atmoz.lidar.ucn"}


def __getattr__(name: str):
    if name in _MODULES:
        import importlib

        try:
            module = importlib.import_module(_MODULES[name])
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"atmoz.lidar needs optional dependencies ({exc.name!r} is missing). "
                'Install them with: pip install "atmoz[lidar]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module 'atmoz.lidar' has no attribute {name!r}")
