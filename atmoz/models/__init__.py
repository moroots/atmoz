"""Model output access (NASA GEOS-CF). Requires ``pip install "atmoz[models]"``."""

from __future__ import annotations

__all__ = ["GEOS_CF"]


def __getattr__(name: str):
    if name == "GEOS_CF":
        try:
            from atmoz.models.geos_cf import GEOS_CF
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"atmoz.models needs optional dependencies ({exc.name!r} is missing). "
                'Install them with: pip install "atmoz[models]"'
            ) from exc
        return GEOS_CF
    raise AttributeError(f"module 'atmoz.models' has no attribute {name!r}")
