# HDF4 files (NASA archives)

Many NASA products, including older TOLNet files, are **HDF4**. `h5py` (and so atmoz) reads HDF5 only. The HDF Group's
`h4toh5convert` converts HDF4 to HDF5. atmoz does **not** install it with the package: it stays in the GitHub repository
(`extras/h4toh5/`) and you pull it when you need it.

## Install the converter (once, on request)

```python
from atmoz.resources import h4toh5

h4toh5.install_h4toh5()        # downloads, verifies, installs to ~/.atmoz/bin
```

or from a shell: `python -m atmoz.resources.h4toh5 install`.

What happens:

* The file is fetched over HTTPS from `extras/h4toh5/<platform>/` in the atmoz GitHub repo (`ref="main"` by default; pass a tag such as `"v0.1.1"` to pin).
* Its SHA-256 is compared with the value pinned inside the installed atmoz package. A mismatch raises an error and installs nothing.
* Only after the check is it made executable and moved to `~/.atmoz/bin` (`$ATMOZ_HOME/bin` if set).
* **Nothing is downloaded automatically.** If a conversion needs the tool and it is missing, you get an error that names this one-liner.

A prebuilt converter is provided for **Linux x86-64**. On macOS or Windows download the h4h5tools build from the
[HDF Group](https://support.hdfgroup.org/ftp/HDF5/tools/h4toh5/bin/) and set `ATMOZ_H4TOH5` to the executable (or put it on `PATH`).

`find_h4toh5()` looks in this order: the `exe=` argument, `$ATMOZ_H4TOH5`, the installed copy, then `PATH`.

## Convert files

```python
h4toh5.prepare_h5("tolnet_file.hdf")                    # -> .../tolnet_file.h5 (HDF5 in: returned unchanged)
h4toh5.h4_to_h5("in.hdf", "out.h5")                     # one file
h4toh5.convert_hdf4_to_hdf5("hdf4_dir/", "h5_dir/")     # every *.hdf in a folder
```

`prepare_h5` looks at the file's first bytes, not its extension, because some servers send zipped HDF4 under a `.hdf` name.
HDF5 is returned as is, HDF4 is converted, and a zip has its HDF member extracted (to a name chosen by atmoz, never the
path stored in the archive) and handled the same way. Inputs are never modified.

## Licence

The binary is the HDF Group's own tool, distributed under their BSD-style licence, and is only run as a separate program.
See `extras/h4toh5/README.md` for provenance and how to update it.

## Reading the result: the lazy HDF5 reader

`atmoz.resources.hdf` opens HDF5 files lazily: nothing is read until you slice a dataset or ask for an array. It works on
local files, on open file objects (for example `earthaccess.open` streams), and on what `prepare_h5` returns.

```python
from atmoz.resources.hdf import open_h5

with open_h5("tolnet_file.h5") as f:                       # closed on exit
    f.attrs_dict()                                         # all file attributes, bytes decoded to str
    "ALTITUDE" in f                                        # membership test
    o3 = f["O3.MIXING.RATIO.VOLUME_DERIVED"]
    o3.attr("VAR_UNITS")                                   # one attribute, decoded
    o3.array(fill_value="auto")                            # read now; the file's own fill value becomes NaN
    o3.array(key=(slice(0, 10), slice(None)))              # or just a window
```

Pass a HDF4 or zipped file to `open_h5` and you get an error that names `prepare_h5`. `open_hdf(paths, convert=True)` and
`TOLNET.fetch(..., file_type="hdf", convert=True)` convert first, so the whole path is download, convert, read.
