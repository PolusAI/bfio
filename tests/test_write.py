# -*- coding: utf-8 -*-
import unittest
import requests, io, pathlib, shutil, logging, sys
import bfio
import numpy as np
from ome_zarr.utils import download as zarr_download

TEST_IMAGES = {
    "Plate1-Blue-A-12-Scene-3-P3-F2-03.czi": "https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-12-Scene-3-P3-F2-03.czi",
}

TEST_DIR = pathlib.Path(__file__).with_name("data")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.test")

if "-v" in sys.argv:
    logger.setLevel(logging.INFO)


def setUpModule():
    """Download images for testing"""
    TEST_DIR.mkdir(exist_ok=True)

    for file, url in TEST_IMAGES.items():
        logger.info(f"setup - Downloading: {file}")

        if not file.endswith(".ome.zarr"):
            if TEST_DIR.joinpath(file).exists():
                continue

            r = requests.get(url)

            with open(TEST_DIR.joinpath(file), "wb") as fw:
                fw.write(r.content)
        else:
            zarr_download(url, str(TEST_DIR))


class TestOmeTiffWrite(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Load the czi image, and save as a npy file for further testing."""

        if TEST_DIR.joinpath("4d_array.npy").exists():
            return

        with bfio.BioReader(
            TEST_DIR.joinpath("Plate1-Blue-A-12-Scene-3-P3-F2-03.czi")
        ) as br:
            np.save(TEST_DIR.joinpath("4d_array.npy"), br[:])

    def test_write_python(self):
        with bfio.BioWriter("4d_array.ome.tif") as bw:
            image = np.load(TEST_DIR.joinpath("4d_array.npy"))

            bw.shape = image.shape[:3]
            bw.dtype = image.dtype

            bw[:] = image[:, :, :, 0]

        with bfio.BioReader("4d_array.ome.tif", backend="python") as br:
            reconstructed = br[:]

        assert np.array_equal(image[..., 0], reconstructed)

    def test_write_channels_python(self):
        with bfio.BioWriter("4d_array_python.ome.tif") as bw:
            image = np.load(TEST_DIR.joinpath("4d_array.npy"))

            bw.shape = image.shape
            bw.dtype = image.dtype
            bw[:] = image[:]
        with bfio.BioReader("4d_array_python.ome.tif") as br:
            image = np.load(TEST_DIR.joinpath("4d_array.npy"))
            assert image.shape == br.shape
            assert np.array_equal(image[:], br[:])

    @unittest.skipIf(sys.platform.startswith("darwin"), "Does not work in Mac")
    def test_write_java(self):
        with bfio.BioWriter("4d_array_bf.ome.tif", backend="bioformats") as bw:
            image = np.load(TEST_DIR.joinpath("4d_array.npy"))

            bw.shape = image.shape
            bw.dtype = image.dtype

            bw[:] = image
        with bfio.BioReader("4d_array_bf.ome.tif") as br:
            assert image.shape == br.shape
            assert np.array_equal(image[:], br[:])
