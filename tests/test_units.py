import numpy as np
import pytest

from atmoz.resources.atmoz_dataclasses import dataframe


def test_convert_ppbv_to_ppmv():
    out = dataframe(np.array([50.0, 100.0]), units="ppbv").convert_units("ppmv")
    assert out.units == "ppmv"
    assert np.allclose(out.data.to_numpy().ravel(), [0.05, 0.1])


def test_add_converts_other_units():
    a = dataframe(np.array([1.0]), units="ppmv")
    b = dataframe(np.array([500.0]), units="ppbv")
    assert np.allclose((a + b).data.to_numpy().ravel(), [1.5])


def test_incompatible_units_raise():
    with pytest.raises(ValueError):
        _ = dataframe(np.array([1.0]), units="ppbv") + dataframe(np.array([1.0]), units="meter")
