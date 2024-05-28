# -*- coding: utf-8 -*-
import unittest
import requests, pathlib, shutil, logging, sys
import bfio
import numpy as np
import random
import zarr
from ome_zarr.utils import download as zarr_download

TEST_IMAGES = {
    "5025551.zarr": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0054A/5025551.zarr",
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

    """Load the czi image, and save as a npy file for further testing."""
    with bfio.BioReader(
        TEST_DIR.joinpath("Plate1-Blue-A-12-Scene-3-P3-F2-03.czi")
    ) as br:
        np.save(TEST_DIR.joinpath("4d_array.npy"), br[:])
        zf = zarr.open(
            str(TEST_DIR.joinpath("4d_array.zarr")),
            mode="w",
            shape=(1, br.C, br.Z, br.Y, br.X),
            dtype=br.dtype,
            chunks=(1, 1, 1, 1024, 1024),
        )
        for t in range(1):
            for c in range(br.C):
                for z in range(br.Z):
                    zf[t, c, z, :, :] = br[:, :, z, c, t]


def tearDownModule():
    """Remove test images"""

    logger.info("teardown - Removing test images...")
    shutil.rmtree(TEST_DIR)


class TestSimpleRead(unittest.TestCase):
    def test_bioformats(self):
        """test_bioformats - Fails if Java/JPype improperly configured"""

        bfio.start()

    def test_read_czi(self):
        """test_read_czi

        This test will fail if JPype and Java are not installed or improperly
        configured.

        """

        with bfio.BioReader(
            TEST_DIR.joinpath("Plate1-Blue-A-12-Scene-3-P3-F2-03.czi")
        ) as br:
            self.assertEqual(br._backend_name, "bioformats")

            I = br[:]

            np.save(TEST_DIR.joinpath("4d_array.npy"), br[:])

            self.assertEqual(br.shape[0], 512)
            self.assertEqual(br.shape[1], 672)
            self.assertEqual(br.shape[2], 21)
            self.assertEqual(br.shape[3], 3)
            self.assertEqual(br.dtype, np.uint16)
            assert all(i == b for i, b in zip(I.shape, br.shape))
            assert br.dtype == I.dtype

    def test_read_tif_strip_auto(self):
        """test_read_tif_strip_auto - Read tiff saved in strips, should load bioformats backend"""
        with bfio.BioReader(TEST_DIR.joinpath("0.tif")) as br:
            self.assertEqual(br._backend_name, "bioformats")

            I = br[:]

    def test_read_zarr_auto(self):
        """test_read_zarr_auto - Read ome zarr, should load zarr backend"""
        with bfio.BioReader(TEST_DIR.joinpath("4d_array.zarr")) as br:
            self.assertEqual(br._backend_name, "zarr")

            I = br[:]

            logger.info(I.shape)

    def test_read_ome_tif_strip_auto(self):
        """test_read_ome_tif_strip_auto - Read tiff using Java backend"""
        with bfio.BioReader(TEST_DIR.joinpath("img_r001_c001.ome.tif")) as br:
            I = br[:]

    def test_read_tif_strip_bioformats(self):
        """test_read_tif_strip_bioformats - Read tiff using Java backend"""
        with bfio.BioReader(
            TEST_DIR.joinpath("img_r001_c001.ome.tif"), backend="bioformats"
        ) as br:
            self.assertEqual(br._backend_name, "bioformats")

            I = br[:]

    def test_read_tif_strip_python(self):
        """test_read_tif_strip_python - Issue warning and switch to Java backend"""
        with bfio.BioReader(
            TEST_DIR.joinpath("img_r001_c001.ome.tif"), backend="python"
        ) as br:
            I = br[:]

    def test_read_unaligned_tile_boundary_python(self):
        # create a 2D numpy array filled with random integer form 0-255
        img_height = 8000
        img_width = 7500
        source_data = np.random.randint(
            0, 256, (img_height, img_width), dtype=np.uint16
        )
        with bfio.BioWriter(
            str(TEST_DIR.joinpath("test_output.ome.tiff")),
            X=img_width,
            Y=img_height,
            dtype=np.uint16,
        ) as bw:
            bw[0:img_height, 0:img_width, 0, 0, 0] = source_data

        x_max = source_data.shape[0]
        y_max = source_data.shape[1]

        with bfio.BioReader(str(TEST_DIR.joinpath("test_output.ome.tiff"))) as test_br:
            for i in range(100):
                x_start = random.randint(0, x_max)
                y_start = random.randint(0, y_max)
                x_step = random.randint(1, 2 * 1024)
                y_step = random.randint(1, 3 * 1024)

                x_end = x_start + x_step
                y_end = y_start + y_step

                if x_end > x_max:
                    x_end = x_max

                if y_end > y_max:
                    y_end = y_max

                test_data = test_br[x_start:x_end, y_start:y_end, ...]
                assert (
                    np.sum(source_data[x_start:x_end, y_start:y_end] - test_data) == 0
                )

    def test_read_unaligned_tile_boundary_tensorstore(self):
        # create a 2D numpy array filled with random integer form 0-255
        img_height = 8000
        img_width = 7500
        source_data = np.random.randint(
            0, 256, (img_height, img_width), dtype=np.uint16
        )
        with bfio.BioWriter(
            str(TEST_DIR.joinpath("test_output.ome.tiff")),
            X=img_width,
            Y=img_height,
            dtype=np.uint16,
        ) as bw:
            bw[0:img_height, 0:img_width, 0, 0, 0] = source_data

        x_max = source_data.shape[0]
        y_max = source_data.shape[1]

        with bfio.BioReader(
            str(TEST_DIR.joinpath("test_output.ome.tiff")), backend="tensorstore"
        ) as test_br:
            for i in range(100):
                x_start = random.randint(0, x_max)
                y_start = random.randint(0, y_max)
                x_step = random.randint(1, 2 * 1024)
                y_step = random.randint(1, 3 * 1024)

                x_end = x_start + x_step
                y_end = y_start + y_step

                if x_end > x_max:
                    x_end = x_max

                if y_end > y_max:
                    y_end = y_max

                test_data = test_br[x_start:x_end, y_start:y_end, ...]
                assert (
                    np.sum(source_data[x_start:x_end, y_start:y_end] - test_data) == 0
                )


# Metadata tests to run on each backend


def get_dims(reader):
    """Get all dimension attributes"""
    for dim in "xyzct":
        logger.info("image.{} = {}".format(dim, getattr(reader, dim)))
    for dim in "xyzct".upper():
        logger.info("image.{} = {}".format(dim, getattr(reader, dim)))
    logger.info("image.shape = {}".format(reader.shape))


def get_pixel_size(reader):
    """Get all pixel size attributes"""
    for dim in "xyz":
        attribute = "physical_size_{}".format(dim)
        logger.info(
            "image.physical_size_{} = {}".format(dim, getattr(reader, attribute))
        )
    for dim in "xyz":
        attribute = "ps_{}".format(dim)
        logger.info("image.ps_{} = {}".format(dim, getattr(reader, attribute)))


def get_pixel_info(reader):
    """Get pixel information (type, samples per pixel, etc)"""
    logger.info("image.samples_per_pixel={}".format(reader.samples_per_pixel))
    logger.info("image.spp={}".format(reader.spp))
    logger.info("image.bytes_per_pixel={}".format(reader.bytes_per_pixel))
    logger.info("image.bpp={}".format(reader.bpp))
    logger.info("image.dtype={}".format(reader.dtype))


def get_channel_names(reader):
    """Get channel names attribute"""
    logger.info("image.channel_names={}".format(reader.channel_names))
    logger.info("image.cnames={}".format(reader.cnames))


# Test classes (where the testing actually happens)


class TestVersion(unittest.TestCase):
    def test_bfio_version(self):
        """Ensure bfio version is properly loaded"""
        logger.info("__version__ = {}".format(bfio.__version__))
        assert bfio.__version__ != "0.0.0"

    def test_jar_version(self):
        """Load loci-tools.jar and get version"""
        logger.info("JAR_VERSION = {}".format(bfio.JAR_VERSION))
        assert bfio.__version__ != None


class TestJavaReader(unittest.TestCase):
    def test_get_dims(self):
        """Testing metadata dimension attributes"""
        with bfio.BioReader(TEST_DIR.joinpath("0.tif")) as br:
            get_dims(br)

    def test_get_pixel_size(self):
        """Testing metadata pixel sizes"""
        with bfio.BioReader(TEST_DIR.joinpath("0.tif")) as br:
            get_pixel_size(br)

    def test_get_pixel_info(self):
        """Testing metadata pixel information"""
        with bfio.BioReader(TEST_DIR.joinpath("0.tif")) as br:
            get_pixel_info(br)

    def test_get_channel_names(self):
        """Testing metadata channel names"""
        with bfio.BioReader(TEST_DIR.joinpath("0.tif")) as br:
            get_channel_names(br)

    def test_sub_resolution_read(self):
        """Testing multi-resolution read"""
        with bfio.BioReader(TEST_DIR.joinpath("Leica-1.scn")) as br:
            get_dims(br)
            self.assertEqual(br.shape, (4668, 1616, 1, 3))
        with bfio.BioReader(TEST_DIR.joinpath("Leica-1.scn"), level=1) as br:
            get_dims(br)
            self.assertEqual(br.shape, (1167, 404, 1, 3))


class TestZarrReader(unittest.TestCase):
    def test_get_dims(self):
        """Testing metadata dimension attributes"""
        with bfio.BioReader(TEST_DIR.joinpath("4d_array.zarr"), backend="zarr") as br:
            get_dims(br)
            self.assertEqual(br.shape, (512, 672, 21, 3))

    def test_get_pixel_size(self):
        """Testing metadata pixel sizes"""
        with bfio.BioReader(TEST_DIR.joinpath("4d_array.zarr")) as br:
            get_pixel_size(br)

    def test_get_pixel_info(self):
        """Testing metadata pixel information"""
        with bfio.BioReader(TEST_DIR.joinpath("4d_array.zarr")) as br:
            get_pixel_info(br)

    def test_get_channel_names(self):
        """Testing metadata channel names"""
        with bfio.BioReader(TEST_DIR.joinpath("4d_array.zarr")) as br:
            get_channel_names(br)

    def test_sub_resolution_read(self):
        """Testing multi-resolution read"""
        with bfio.BioReader(TEST_DIR.joinpath("5025551.zarr")) as br:
            get_dims(br)
            self.assertEqual(br.shape, (2700, 2702, 1, 27))
        with bfio.BioReader(TEST_DIR.joinpath("5025551.zarr"), level=1) as br:
            get_dims(br)
            self.assertEqual(br.shape, (1350, 1351, 1, 27))


class TestZarrTSReader(unittest.TestCase):
    def test_get_dims(self):
        """Testing metadata dimension attributes"""
        with bfio.BioReader(
            TEST_DIR.joinpath("4d_array.zarr"), backend="tensorstore"
        ) as br:
            get_dims(br)
            self.assertEqual(br.shape, (512, 672, 21, 3))

    def test_get_pixel_size(self):
        """Testing metadata pixel sizes"""
        with bfio.BioReader(
            TEST_DIR.joinpath("4d_array.zarr"), backend="tensorstore"
        ) as br:
            get_pixel_size(br)

    def test_get_pixel_info(self):
        """Testing metadata pixel information"""
        with bfio.BioReader(
            TEST_DIR.joinpath("4d_array.zarr"), backend="tensorstore"
        ) as br:
            get_pixel_info(br)

    def test_get_channel_names(self):
        """Testing metadata channel names"""
        with bfio.BioReader(
            TEST_DIR.joinpath("4d_array.zarr"), backend="tensorstore"
        ) as br:
            get_channel_names(br)

    def test_sub_resolution_read(self):
        """Testing multi-resolution read"""
        with bfio.BioReader(
            TEST_DIR.joinpath("5025551.zarr"), backend="tensorstore"
        ) as br:
            get_dims(br)
            self.assertEqual(br.shape, (2700, 2702, 1, 27))
        with bfio.BioReader(
            TEST_DIR.joinpath("5025551.zarr"), backend="tensorstore", level=1
        ) as br:
            get_dims(br)
            self.assertEqual(br.shape, (1350, 1351, 1, 27))


class TestZarrMetadata(unittest.TestCase):
    def test_set_metadata(self):
        """Testing metadata dimension attributes"""
        cname = ["test"]

        image = np.load(TEST_DIR.joinpath("4d_array.npy"))

        with bfio.BioWriter(TEST_DIR.joinpath("test_cname.ome.zarr")) as bw:
            bw.cnames = cname
            bw.ps_x = (100, "nm")
            bw.shape = image.shape
            bw[:] = image

        with bfio.BioReader(TEST_DIR.joinpath("test_cname.ome.zarr")) as br:
            logger.info(br.cnames)
            logger.info(br.ps_x)
            self.assertEqual(br.cnames[0], cname[0])


class TestZarrTesnsorstoreMetadata(unittest.TestCase):
    def test_set_metadata(self):
        """Testing metadata dimension attributes"""
        cname = ["test"]

        image = np.load(TEST_DIR.joinpath("4d_array.npy"))

        with bfio.BioWriter(TEST_DIR.joinpath("test_cname.ome.zarr")) as bw:
            bw.cnames = cname
            bw.ps_x = (100, "nm")
            bw.shape = image.shape
            bw[:] = image

        with bfio.BioReader(
            TEST_DIR.joinpath("test_cname.ome.zarr"), backend="tensorstore"
        ) as br:
            logger.info(br.cnames)
            logger.info(br.ps_x)
            self.assertEqual(br.cnames[0], cname[0])
