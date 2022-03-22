import unittest
import requests, io, pathlib, shutil, logging, sys
import bfio
import numpy as np

TEST_IMAGES = {
    # "1884807.ome.zarr": "https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/1884807.zarr/",
    "Plate1-Blue-A-12-Scene-3-P3-F2-03.czi": "https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-12-Scene-3-P3-F2-03.czi",
    "0.tif": "https://osf.io/j6aer/download",
    "img_r001_c001.ome.tif": "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/img_r001_c001.ome.tif",
    "Leica-1.scn": "https://downloads.openmicroscopy.org/images/Leica-SCN/openslide/Leica-1/Leica-1.scn",
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

            base_path = TEST_DIR.joinpath(file)
            base_path.mkdir()
            base_path.joinpath("0").mkdir()

            units = [
                ".zattrs",
                ".zgroup",
                "0/.zarray",
                "0/0.0.0.0.0",
                "0/0.1.0.0.0",
                "0/0.2.0.0.0",
            ]

            for u in units:

                if base_path.joinpath(u).exists():
                    continue

                with open(base_path.joinpath(u), "wb") as fw:

                    fw.write(requests.get(url + u).content)


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

    @unittest.expectedFailure
    def test_write_channels_python(self):
        # Cannot write an image with channel information using python backend

        with bfio.BioWriter("4d_array.ome.tif") as bw:

            image = np.load(TEST_DIR.joinpath("4d_array.npy"))

            bw.shape = image.shape
            bw.dtype = image.dtype

    def test_write_java(self):
        # Cannot write an image with channel information using python backend

        with bfio.BioWriter("4d_array.ome.tif", backend="java") as bw:

            image = np.load(TEST_DIR.joinpath("4d_array.npy"))

            bw.shape = image.shape
            bw.dtype = image.dtype

            bw[:] = image

        with bfio.BioReader("4d_array.ome.tif", backend="python") as br:

            reconstructed = br[:]

        assert np.array_equal(image, reconstructed)
