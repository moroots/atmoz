# h4toh5convert (HDF Group)

`h4toh5convert` converts HDF4 files (older NASA products, TOLNet `.hdf`, ...) to HDF5 so `h5py` can read them.

**This folder is not part of the `atmoz` wheel or sdist.** The binary lives in the GitHub repository only and is
downloaded when you ask for it:

```python
from atmoz.resources import h4toh5
h4toh5.install_h4toh5()          # downloads from this folder, verifies the SHA-256, installs under ~/.atmoz/bin
```

Nothing is downloaded automatically. See `docs/hdf4.md`.

| Platform | File | SHA-256 |
|---|---|---|
| Linux x86-64 | `linux-x86_64/h4toh5convert` | `10cc85f7cc8a648150fc1b6bdfaa6406edc6ba2d6b2933b63a23e79e0befce4d` |

The checksum is pinned inside `atmoz/resources/h4toh5.py`, so a file that does not match is refused.

## Provenance and licence

The binary is The HDF Group's own `h4toh5convert` (HDF5 1.10 build, CentOS 7 release build; the build paths are
embedded in the file). It is distributed under the HDF Group's BSD-style licence:
<https://support.hdfgroup.org/ftp/HDF5/releases/COPYING> and <https://github.com/HDFGroup/hdf4/blob/master/COPYING>.
atmoz itself is GPL-3.0; this tool is only *run* as a separate program, never linked.

Other platforms (macOS, Windows): get the h4h5tools build for your system from the HDF Group
(<https://support.hdfgroup.org/ftp/HDF5/tools/h4toh5/bin/>) and point `ATMOZ_H4TOH5` at the executable.

## Updating the binary

Replace the file, update `_BINARIES` (path, size, SHA-256) in `atmoz/resources/h4toh5.py` and this table in the same commit.
