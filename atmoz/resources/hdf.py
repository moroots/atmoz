"""Lazy, NumPy-friendly access to HDF5 files (TOLNet GEOMS files, HSRL-2, ...). Needs ``h5py`` (the ``lidar`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Union

import h5py as h5
import numpy as np


def read_h5(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    def recursive_load(h5_obj):
        """Recursively walk through the HDF5 structure and return nested dicts."""
        result = {}
        for key, item in h5_obj.items():
            if isinstance(item, h5.Dataset):
                # Convert dataset to a NumPy array
                result[key] = item[()]
            elif isinstance(item, h5.Group):
                # Recurse into subgroups
                result[key] = recursive_load(item)
        # Add attributes if present
        if h5_obj.attrs:
            result["_attrs"] = {k: v for k, v in h5_obj.attrs.items()}
        return result

    with h5.File(filepath, "r") as f:
        data = recursive_load(f)

    return data, filepath.name

class H5Dataset:
    """Wrap h5py.Dataset for lazy loading with full NumPy-like behavior."""
    def __init__(self, dataset: h5.Dataset):
        self._dataset = dataset

    def __array__(self, dtype=None, **kwargs):
        arr = self._dataset[()]
        if dtype is not None:
            arr = arr.astype(dtype, copy=kwargs.get("copy", True))
        return arr

    # Arithmetic operators
    def __mul__(self, other):
        return np.array(self) * other

    def __rmul__(self, other):
        return other * np.array(self)

    def __add__(self, other):
        return np.array(self) + other

    def __radd__(self, other):
        return other + np.array(self)

    def __sub__(self, other):
        return np.array(self) - other

    def __rsub__(self, other):
        return other - np.array(self)

    def __truediv__(self, other):
        return np.array(self) / other

    def __rtruediv__(self, other):
        return other / np.array(self)

    # Indexing/slicing
    def __getitem__(self, key):
        return self._dataset[key]

    def __len__(self):
        return len(self._dataset)

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def dtype(self):
        return self._dataset.dtype

    @property
    def attrs(self):
        return self._dataset.attrs

    def __repr__(self):
        return f"<H5Dataset {self._dataset.name} shape={self._dataset.shape} dtype={self._dataset.dtype}>"


class H5Node:
    """Wrap h5py.Group to allow dot-access, tab completion, and lazy dataset access."""
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        item = self._obj[key]
        if isinstance(item, h5.Group):
            return H5Node(item)
        elif isinstance(item, h5.Dataset):
            return H5Dataset(item)
        return item

    def __getattr__(self, name):
        if name in self._obj:
            return self[name]
        if name == "attrs":
            return self._obj.attrs
        if name == "keys":
            return self.keys
        raise AttributeError(f"{name} not found in HDF5 group '{self._obj.name}'")

    def keys(self):
        return list(self._obj.keys())

    def __dir__(self):
        # Only include child **groups** for tab completion
        # group_keys = [k for k in self._obj.keys() if isinstance(self._obj[k], h5.Group)]
        return list(self._obj.keys()) + list(super().__dir__())

    def __repr__(self):
        cls = type(self._obj).__name__
        return f"<H5Node {cls} '{self._obj.name}'>"


def read_h5_lazy(filepath):
    filepath = Path(filepath)
    f = h5.File(filepath, "r", swmr=True)
    return H5Node(f), filepath.name


def open_hdf(filepaths: Union[str, Path, Iterable[Union[str, Path]]]) -> Dict[str, "H5Node"]:
    """Open HDF5 files lazily: ``{filename: H5Node}``. Close each with ``node._obj.close()`` when done."""
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    return dict(reversed(read_h5_lazy(f)) for f in filepaths)
