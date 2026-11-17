"""Lazy, NumPy-friendly access to HDF5 files (TOLNet GEOMS files, HSRL-2, TEMPO, ...). Needs ``h5py`` (``lidar``/``satellite`` extra).

>>> with open_h5("TOLNet.h5") as f:
...     f["O3.MIXING.RATIO.VOLUME_DERIVED"].array(fill_value="auto")     # read now, bad values -> NaN
...     f.attr("DATA_LOCATION"), "ALTITUDE" in f

HDF4 files (older NASA products) are not HDF5: convert them first with :mod:`atmoz.resources.h4toh5`
(``prepare_h5``), whose converter is installed only on request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import h5py as h5
import numpy as np

from atmoz.resources.h4toh5 import detect_format

PathLike = Union[str, Path]


def _decode(value: Any) -> Any:
    """bytes / ``np.bytes_`` -> str, and 1-element arrays of them -> str; anything else is returned untouched."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype.kind in ("S", "O"):
        return _decode(value.reshape(-1)[0])
    return value


class H5Dataset:
    """Wrap an ``h5py.Dataset``. Nothing is read until you slice it or turn it into an array."""

    def __init__(self, dataset: h5.Dataset):
        self._dataset = dataset

    def __array__(self, dtype=None, copy=None):
        arr = self._dataset[()]
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __getitem__(self, key):
        return self._dataset[key]

    def __len__(self):
        return len(self._dataset)

    def __float__(self):
        arr = np.asarray(self._dataset[()])
        if arr.size != 1:
            raise TypeError(f"only size-1 datasets convert to float (shape={arr.shape})")
        return float(arr.reshape(-1)[0])

    # Arithmetic gives plain NumPy arrays
    def __neg__(self):
        return -np.asarray(self)

    def __mul__(self, other):
        return np.asarray(self) * other

    def __rmul__(self, other):
        return other * np.asarray(self)

    def __add__(self, other):
        return np.asarray(self) + other

    def __radd__(self, other):
        return other + np.asarray(self)

    def __sub__(self, other):
        return np.asarray(self) - other

    def __rsub__(self, other):
        return other - np.asarray(self)

    def __truediv__(self, other):
        return np.asarray(self) / other

    def __rtruediv__(self, other):
        return other / np.asarray(self)

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def dtype(self):
        return self._dataset.dtype

    @property
    def name(self) -> str:
        return self._dataset.name

    @property
    def attrs(self):
        return self._dataset.attrs

    def attr(self, name: str, default: Any = None) -> Any:
        """One attribute, with bytes decoded to ``str``."""
        if name not in self._dataset.attrs:
            return default
        return _decode(self._dataset.attrs[name])

    def array(self, dtype=None, fill_value=None, key=()) -> np.ndarray:
        """Read the data (or a window of it) into memory.

        ``key`` is an index or tuple of slices (default: everything). ``dtype`` casts the result (needed before
        NaN-masking integer data). ``fill_value`` is replaced by NaN; ``"auto"`` uses the dataset's own
        ``VAR_FILL_VALUE`` / ``_FillValue`` / ``missing_value`` attribute if it has one.
        """
        arr = self._dataset[key]
        if dtype is not None or fill_value is not None:
            arr = arr.astype(float if dtype is None else dtype)
        if isinstance(fill_value, str) and fill_value == "auto":
            fill_value = None
            for attr in ("VAR_FILL_VALUE", "_FillValue", "missing_value"):
                if attr in self._dataset.attrs:
                    fill_value = np.asarray(self._dataset.attrs[attr]).reshape(-1)[0]
                    break
            if fill_value is not None and arr.dtype.kind != "f":
                arr = arr.astype(float)
        if fill_value is not None:
            arr = np.where(arr == fill_value, np.nan, arr)
        return arr if np.ndim(arr) else np.asarray(arr)

    def __repr__(self):
        return f"<H5Dataset {self._dataset.name} shape={self._dataset.shape} dtype={self._dataset.dtype}>"


class H5Node:
    """Wrap an ``h5py.Group``: dot-access, tab completion and lazy dataset access."""

    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        item = self._obj[key]
        if isinstance(item, h5.Group):
            return H5Node(item)
        if isinstance(item, h5.Dataset):
            return H5Dataset(item)
        return item

    def __getattr__(self, name):
        # Private/dunder lookups must not touch self._obj (it may not exist yet, e.g. while copying or
        # unpickling) or this recurses forever.
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._obj:
            return self[name]
        raise AttributeError(f"{name!r} not found in HDF5 group {self._obj.name!r}")

    def __contains__(self, key) -> bool:
        return key in self._obj

    def keys(self) -> List[str]:
        return list(self._obj.keys())

    @property
    def attrs(self):
        return self._obj.attrs

    def attr(self, name: str, default: Any = None) -> Any:
        """One attribute, with bytes decoded to ``str``."""
        if name not in self._obj.attrs:
            return default
        return _decode(self._obj.attrs[name])

    def attrs_dict(self) -> Dict[str, Any]:
        """All attributes as a plain dict, with bytes decoded to ``str``."""
        return {k: _decode(v) for k, v in self._obj.attrs.items()}

    def __dir__(self):
        return self.keys() + list(super().__dir__())

    def __repr__(self):
        return f"<H5Node {type(self._obj).__name__} {self._obj.name!r}>"


class H5File(H5Node):
    """An open HDF5 file. Use as a context manager, or call :meth:`close` when done."""

    def __init__(self, source, **h5_kwargs):
        """``source`` is a path or an open binary file-like object (e.g. a remote file from ``earthaccess.open``)."""
        if hasattr(source, "read"):
            label = str(getattr(source, "path", None) or getattr(source, "name", None) or "stream")
            self.filepath = Path(label.rstrip("/").split("/")[-1])
        else:
            self.filepath = Path(source)
        super().__init__(h5.File(source if hasattr(source, "read") else self.filepath, "r", **h5_kwargs))

    def close(self) -> None:
        if self._obj:
            self._obj.close()

    def __enter__(self) -> "H5File":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self):
        return f"<H5File {self.filepath.name!r}>"


def open_h5(filepath, **h5_kwargs) -> H5File:
    """Open an HDF5 file lazily. A clear error for HDF4, zipped or non-HDF5 input (see ``h4toh5.prepare_h5``).

    ``filepath`` may also be an open binary file-like object, read as is. Extra keyword arguments go to
    ``h5py.File`` (e.g. ``rdcc_nbytes`` for a larger chunk cache when making many small reads).
    """
    if hasattr(filepath, "read"):
        return H5File(filepath, **h5_kwargs)
    filepath = Path(filepath)
    fmt = detect_format(filepath)
    if fmt != "hdf5":
        hint = ("Convert it first with atmoz.resources.h4toh5.prepare_h5() (HDF4 needs the converter, installed on "
                "request with install_h4toh5())." if fmt in ("hdf4", "zip") else "Is it the right file?")
        raise ValueError(f"{filepath.name} is {fmt!r}, not HDF5. {hint}")
    return H5File(filepath, **h5_kwargs)


def read_h5_lazy(filepath: PathLike):
    """Returns ``(H5File, filename)``; the file stays open until you close it."""
    node = open_h5(filepath)
    return node, node.filepath.name


def read_h5(filepath: PathLike):
    """Eagerly load a whole HDF5 file into nested dicts. Returns ``(dict, filename)``."""
    filepath = Path(filepath)

    def recursive_load(obj):
        result = {}
        for key, item in obj.items():
            if isinstance(item, h5.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5.Group):
                result[key] = recursive_load(item)
        if obj.attrs:
            result["_attrs"] = {k: _decode(v) for k, v in obj.attrs.items()}
        return result

    with h5.File(filepath, "r") as f:
        return recursive_load(f), filepath.name


def open_hdf(filepaths: Union[PathLike, Iterable[PathLike]], *, convert: bool = False) -> Dict[str, H5Node]:
    """Open HDF5 files lazily: ``{filename: H5File}``. Close each with ``node.close()`` when done.

    ``convert=True`` first turns HDF4 / zipped files into HDF5 (``h4toh5.prepare_h5``; the converter must be installed,
    nothing is downloaded for you). HDF5 files are opened as they are.
    """
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    if convert:
        from atmoz.resources.h4toh5 import prepare_h5
        filepaths = [prepare_h5(f) for f in filepaths]
    return dict(reversed(read_h5_lazy(f)) for f in filepaths)
