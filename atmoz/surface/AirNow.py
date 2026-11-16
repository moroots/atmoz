# -*- coding: utf-8 -*-
"""
Created on 2025-10-02 09:24:37

@author: Maurice Roots

Description:
     - ``AirNow``: the single entry point for EPA surface air-quality data.

Two backends sit behind it, each in its own module with the same process and names:

* :class:`atmoz.surface.AirNow_API.AirNowAPI`: the live AirNow web API (near-real-time).
* :class:`atmoz.surface.AQS.EPA_PreGen`: EPA's pre-generated yearly files (history).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from atmoz.resources.atmoz_dataclasses import atmoz_dataset
from atmoz.surface.AirNow_API import AirNowAPI
from atmoz.surface.AQS import EPA_PreGen


class AirNow:
    """Pick a backend with ``endpoint=`` and get the same kind of result either way.

    >>> ds = AirNow.download(endpoint="aqs", parameters="ozone", resolutions="hourly", years=2025)
    >>> ds = AirNow.download(endpoint="airnow", start_date="2025-08-01", end_date="2025-08-07")
    """

    backends = {"airnow": AirNowAPI, "aqs": EPA_PreGen}

    @classmethod
    def _backend(cls, endpoint: str):
        try:
            return cls.backends[endpoint]
        except KeyError:
            raise ValueError(f"Unsupported endpoint: {endpoint}. Valid options are 'airnow' and 'aqs'.") from None

    @classmethod
    def download(cls, endpoint: str = "airnow", **kwargs) -> atmoz_dataset:
        """
        The single entry point. Always returns an :class:`~atmoz.atmoz_dataset` (``datatype="surface"``), the same shape
        for both backends: ``ds.data == {name: DataFrame}`` and ``ds.metadata`` is a dict
        (``endpoint``, ``files``, ``rows``; the AirNow API also adds ``start``, ``end`` and a per-site table ``sites``).
        ``ds.frame`` returns the DataFrame when there is only one.

        * ``endpoint="airnow"``: arguments: see :meth:`AirNowAPI.download`.
        * ``endpoint="aqs"``: arguments: see :meth:`EPA_PreGen.download`.
        """
        return cls._backend(endpoint).download(**kwargs)

    @classmethod
    def fetch(cls, endpoint: str = "airnow", **kwargs) -> Dict[str, Path]:
        """Same arguments as :meth:`download`, but only save raw files to disk and return ``{name: path}``."""
        return cls._backend(endpoint).fetch(**kwargs)

    @staticmethod
    def set_api_key(key: Optional[str] = None, *, scope: Literal["user", "project"] = "user") -> Path:
        """Save your AirNow API key once (hidden prompt if *key* is omitted). See :meth:`AirNowAPI.set_api_key`."""
        return AirNowAPI.set_api_key(key, scope=scope)
