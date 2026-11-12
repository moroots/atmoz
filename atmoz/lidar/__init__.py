"""NASA TOLNet ozone-lidar access. Requires ``pip install "atmoz[lidar]"``."""

from __future__ import annotations

__all__ = ["TOLNET"]


def __getattr__(name: str):
    if name == "TOLNET":
        try:
            from atmoz.lidar.tolnet import TOLNET
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"atmoz.lidar needs optional dependencies ({exc.name!r} is missing). "
                'Install them with: pip install "atmoz[lidar]"'
            ) from exc
        return TOLNET
    raise AttributeError(f"module 'atmoz.lidar' has no attribute {name!r}")
