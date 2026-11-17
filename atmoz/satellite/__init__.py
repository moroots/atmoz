"""Satellite data access (NASA TEMPO gridded L3 products). Heavy readers need ``pip install "atmoz[satellite]"``."""

from __future__ import annotations

__all__ = ["TEMPO", "du_to_ppbv", "integrate_layers"]

_MODULES = {name: "atmoz.satellite.tempo" for name in __all__}


def __getattr__(name: str):
    if name in _MODULES:
        import importlib

        try:
            module = importlib.import_module(_MODULES[name])
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"atmoz.satellite needs optional dependencies ({exc.name!r} is missing). "
                'Install them with: pip install "atmoz[satellite]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module 'atmoz.satellite' has no attribute {name!r}")
