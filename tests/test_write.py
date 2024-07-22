# -*- coding: utf-8 -*-
import unittest
import requests, pathlib, shutil, logging, sys, os
import bfio
import numpy as np
import tempfile
from ome_zarr.utils import download as zarr_download

TEST_IMAGES = {
    "Plate1-Blue-A-12-Scene-3-P3-F2-03.czi": "https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-12-Scene-3-P3-F2-03.czi",
    "5025551.zarr": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0054A/5025551.zarr",
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

        if not file.endswith(".zarr"):
            if TEST_DIR.joinpath(file).exists():
                continue

            r = requests.get(url)

            with open(TEST_DIR.joinpath(file), "wb") as fw:
                fw.write(r.content)
        else:
            if TEST_DIR.joinpath(file).exists():
                shutil.rmtree(TEST_DIR.joinpath(file))
            zarr_download(url, str(TEST_DIR))

# def tearDownModule():
#     """Remove test images"""

#     logger.info("teardown - Removing test images...")
#     shutil.rmtree(TEST_DIR)

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


class TestOmeZarrWriter(unittest.TestCase):

    def test_write_zarr_tensorstore(self):

        with bfio.BioReader(str(TEST_DIR.joinpath("5025551.zarr"))) as br:

            actual_shape = br.shape
            actual_dtype = br.dtype
            
            actual_image = br[:]
            
            actual_mdata = br.metadata
        
        with tempfile.TemporaryDirectory() as dir:

            # Use the temporary directory
            test_file_path = os.path.join(dir, 'out/test.ome.zarr')

            with bfio.BioWriter(test_file_path, metadata=actual_mdata, backend="tensorstore") as bw:

                expanded = np.expand_dims(actual_image, axis=-1)
                bw[:] = expanded

            with bfio.BioReader(test_file_path) as br:


                assert br.shape == actual_shape
                assert br.dtype == actual_dtype

                assert br[:].sum() == actual_image.sum()
