"""Download helpers for test data, with retries and zip unpacking."""

import pathlib
import time
import urllib.request
import zipfile

# Pinned to a commit of sameeul/polus-test-data so the data cannot change.
DATA_REPO = (
    "https://raw.githubusercontent.com/sameeul/polus-test-data/"
    "7ef91e6ca3c6202bd81c105440471049c0977d38/bfio"
)
# zarr v2 (NGFF 0.4), 2 levels: (Y,X,Z,C) = (300,302,1,4) and (150,151,1,4)
PYRAMID_ZARR = "small_5d_pyramid.zarr"
# zarr v3 (NGFF 0.5), same shapes
V3_ZARR = "small_5d_v3.ome.zarr"
ZARR_URLS = {n: f"{DATA_REPO}/{n}.zip" for n in (PYRAMID_ZARR, V3_ZARR)}


def fetch(url, dest, tries=4, timeout=60):
    """Download url to dest, retrying with backoff; raises on final failure."""
    dest = pathlib.Path(dest)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2**i)


def fetch_zarr(name, test_dir):
    """Download the zipped zarr `name` and unpack it into test_dir."""
    test_dir = pathlib.Path(test_dir)
    z = fetch(ZARR_URLS[name], test_dir / f"{name}.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(test_dir)
    z.unlink()
