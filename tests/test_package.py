import subprocess
import sys

import pytest

import atmoz


def test_version_is_exposed():
    assert atmoz.__version__ and atmoz.__version__ != "0+unknown"


def test_lazy_public_names_resolve():
    from atmoz.surface import AirNow
    assert atmoz.AirNow is AirNow
    assert callable(atmoz.set_secret) and callable(atmoz.clear_cache)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        atmoz.does_not_exist


def test_import_is_light():
    """`import atmoz` and the surface downloaders must not pull in the optional heavy dependencies."""
    code = (
        "import sys, atmoz, atmoz.surface\n"
        "heavy = {'geopandas', 'plotly', 'xarray', 'earthaccess', 'h5py', 'pydantic'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
