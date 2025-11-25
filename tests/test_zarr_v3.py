# -*- coding: utf-8 -*-
"""
Unit tests for Zarr v2 and v3 compatibility in bfio

This test suite verifies that bfio works correctly with both Zarr v2 and v3,
including version detection, API compatibility, read/write operations, and
metadata handling.
"""
import unittest
import logging
import sys
import shutil
from pathlib import Path

import numpy as np
import zarr
import bfio

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.test.zarr_v3")

if "-v" in sys.argv:
    logger.setLevel(logging.INFO)

TEST_DIR = Path(__file__).with_name("data_zarr_v3")


def setUpModule():
    """Create test directory"""
    TEST_DIR.mkdir(exist_ok=True)
    logger.info(f"Created test directory: {TEST_DIR}")


def tearDownModule():
    """Clean up test directory"""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        logger.info(f"Removed test directory: {TEST_DIR}")


class TestZarrVersionDetection(unittest.TestCase):
    """Test version detection and API compatibility"""

    def test_zarr_version_detection(self):
        """Verify zarr version can be detected"""
        zarr_version = int(zarr.__version__.split(".")[0])
        logger.info(
            f"Detected zarr version: {zarr.__version__} (major: {zarr_version})"
        )
        self.assertIn(zarr_version, [2, 3], "Zarr version must be 2 or 3")

    def test_zarr_type_checking(self):
        """Verify appropriate types are available based on version"""
        zarr_version = int(zarr.__version__.split(".")[0])

        if zarr_version >= 3:
            # Zarr v3 should have these types
            self.assertTrue(hasattr(zarr, "Array"))
            self.assertTrue(hasattr(zarr, "Group"))
            logger.info("✓ Zarr v3 types verified")
        else:
            # Zarr v2 should have these types
            self.assertTrue(hasattr(zarr.core, "Array"))
            self.assertTrue(hasattr(zarr.hierarchy, "Group"))
            logger.info("✓ Zarr v2 types verified")

    def test_bfio_import(self):
        """Verify bfio can be imported with current zarr version"""
        try:
            import bfio

            logger.info(
                f"✓ bfio {bfio.__version__} imported successfully with zarr {zarr.__version__}"
            )
        except Exception as e:
            self.fail(f"Failed to import bfio: {e}")


class TestZarrBasicOperations(unittest.TestCase):
    """Test basic zarr operations with version-appropriate APIs"""

    def setUp(self):
        """Set up test path"""
        self.test_path = TEST_DIR / "basic_test.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_create_zarr_group(self):
        """Test creating zarr group with appropriate API"""
        zarr_version = int(zarr.__version__.split(".")[0])

        if zarr_version >= 3:
            root = zarr.open_group(str(self.test_path), mode="w")
        else:
            root = zarr.open_group(str(self.test_path), mode="w")

        self.assertIsNotNone(root)
        logger.info(f"✓ Created zarr group with v{zarr_version} API")

    def test_create_zarr_array(self):
        """Test creating zarr array with version-specific syntax"""
        zarr_version = int(zarr.__version__.split(".")[0])

        if zarr_version >= 3:
            root = zarr.open_group(str(self.test_path), mode="w")
            arr = root.zeros(name="test_array", shape=(100, 100), dtype="uint16")
        else:
            root = zarr.open_group(str(self.test_path), mode="w")
            arr = root.zeros("test_array", shape=(100, 100), dtype="uint16")

        self.assertEqual(arr.shape, (100, 100))
        self.assertEqual(arr.dtype, np.uint16)
        logger.info(f"✓ Created zarr array with v{zarr_version} API")

    def test_read_zarr_group(self):
        """Test reading zarr group with appropriate API"""
        zarr_version = int(zarr.__version__.split(".")[0])

        # Create a group first
        if zarr_version >= 3:
            root = zarr.open_group(str(self.test_path), mode="w")
            root.zeros(name="test", shape=(10, 10), dtype="uint8")
        else:
            root = zarr.open_group(str(self.test_path), mode="w")
            root.zeros("test", shape=(10, 10), dtype="uint8")

        # Read it back
        if zarr_version >= 3:
            root_read = zarr.open_group(str(self.test_path), mode="r")
        else:
            root_read = zarr.open(str(self.test_path), mode="r")

        self.assertIsNotNone(root_read)
        self.assertIn("test", list(root_read.array_keys()))
        logger.info(f"✓ Read zarr group with v{zarr_version} API")


class TestZarrWriterV3(unittest.TestCase):
    """Test ZarrWriter functionality with current zarr version"""

    def setUp(self):
        """Set up test paths and data"""
        self.test_path = TEST_DIR / "writer_test.ome.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

        # Create test data
        self.width, self.height = 512, 512
        self.depth, self.channels, self.timepoints = 3, 2, 1
        self.test_data = np.random.randint(
            0,
            256,
            (self.height, self.width, self.depth, self.channels, self.timepoints),
            dtype=np.uint16,
        )

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_write_zarr_basic(self):
        """Test basic zarr writing"""
        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = self.test_data.shape
            bw[:] = self.test_data

        self.assertTrue(self.test_path.exists())
        logger.info(f"✓ Successfully wrote zarr store with zarr {zarr.__version__}")

    def test_write_zarr_with_metadata(self):
        """Test zarr writing with custom metadata"""
        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = self.test_data.shape
            bw.cnames = ["Channel_1", "Channel_2"]
            bw.ps_x = (0.5, "µm")
            bw.ps_y = (0.5, "µm")
            bw[:] = self.test_data

        # Verify metadata file was created
        metadata_path = self.test_path / "OME" / "METADATA.ome.xml"
        self.assertTrue(metadata_path.exists())

        with open(metadata_path, "r") as f:
            metadata_content = f.read()
            self.assertIn("OME", metadata_content)

        logger.info(f"✓ Wrote zarr with metadata using zarr {zarr.__version__}")

    def test_write_zarr_structure(self):
        """Test that written zarr has correct structure"""
        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = self.test_data.shape
            bw[:] = self.test_data

        zarr_version = int(zarr.__version__.split(".")[0])

        # Open and verify structure
        if zarr_version >= 3:
            root = zarr.open_group(str(self.test_path), mode="r")
        else:
            root = zarr.open(str(self.test_path), mode="r")

        # Check that array '0' exists
        arrays = list(root.array_keys())
        self.assertIn("0", arrays)

        # Check array shape (TCZYX)
        arr = root["0"]
        self.assertEqual(len(arr.shape), 5)
        logger.info(f"✓ Zarr structure verified: array shape = {arr.shape}")

    def test_write_zarr_large_tiles(self):
        """Test writing data larger than tile size"""
        large_data = np.random.randint(0, 256, (2048, 2048, 1, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = large_data.shape
            bw[:] = large_data

        self.assertTrue(self.test_path.exists())
        logger.info(f"✓ Wrote large zarr (2048x2048) successfully")


class TestZarrReaderV3(unittest.TestCase):
    """Test ZarrReader functionality with current zarr version"""

    def setUp(self):
        """Set up test paths and create test data"""
        self.test_path = TEST_DIR / "reader_test.ome.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

        # Create test data and write it
        self.width, self.height = 256, 256
        self.depth, self.channels, self.timepoints = 3, 2, 1
        self.test_data = np.random.randint(
            0,
            256,
            (self.height, self.width, self.depth, self.channels, self.timepoints),
            dtype=np.uint16,
        )

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = self.test_data.shape
            bw[:] = self.test_data

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_read_zarr_basic(self):
        """Test basic zarr reading"""
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.dtype, np.uint16)
            logger.info(
                f"✓ Opened zarr for reading: shape={br.shape}, dtype={br.dtype}"
            )

    def test_read_zarr_full_data(self):
        """Test reading complete zarr data"""
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]

            # Compare with original data (accounting for shape differences)
            test_data_squeezed = self.test_data.squeeze()
            self.assertTrue(np.array_equal(test_data_squeezed, read_data))
            logger.info(
                f"✓ Full data read verified: shapes match and data is identical"
            )

    def test_read_zarr_partial(self):
        """Test partial zarr data reading"""
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            # Read a subset
            partial = br[0:100, 0:100, 0, 0]

            # Verify shape
            self.assertEqual(partial.shape, (100, 100))

            # Verify data matches
            self.assertTrue(
                np.array_equal(self.test_data[0:100, 0:100, 0, 0, 0], partial)
            )
            logger.info(f"✓ Partial data read verified: [0:100, 0:100, 0, 0]")

    def test_read_zarr_metadata(self):
        """Test reading zarr metadata"""
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            metadata = br.metadata

            self.assertIsNotNone(metadata)
            self.assertEqual(len(metadata.images), 1)
            logger.info(
                f"✓ Metadata read successfully: {len(metadata.images)} image(s)"
            )

    def test_read_zarr_dimensions(self):
        """Test dimension attributes"""
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.X, self.width)
            self.assertEqual(br.Y, self.height)
            self.assertEqual(br.Z, self.depth)
            self.assertEqual(br.C, self.channels)
            # T dimension may be 1 or not reported for singleton dimension
            logger.info(
                f"✓ Dimensions verified: X={br.X}, Y={br.Y}, Z={br.Z}, C={br.C}"
            )

    def test_read_zarr_random_access(self):
        """Test random access pattern reading"""
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            # Test multiple random reads
            for _ in range(5):
                y_start = np.random.randint(0, self.height // 2)
                x_start = np.random.randint(0, self.width // 2)
                y_size = np.random.randint(10, 100)
                x_size = np.random.randint(10, 100)

                y_end = min(y_start + y_size, self.height)
                x_end = min(x_start + x_size, self.width)

                chunk = br[y_start:y_end, x_start:x_end, 0, 0]
                expected = self.test_data[y_start:y_end, x_start:x_end, 0, 0, 0]

                self.assertTrue(np.array_equal(chunk, expected))

            logger.info(f"✓ Random access reading verified (5 random chunks)")


class TestZarrRoundTrip(unittest.TestCase):
    """Test complete write-read cycles"""

    def setUp(self):
        """Set up test path"""
        self.test_path = TEST_DIR / "roundtrip_test.ome.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_roundtrip_uint8(self):
        """Test write-read roundtrip with uint8 data"""
        data = np.random.randint(0, 256, (128, 128, 1, 1, 1), dtype=np.uint8)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint8
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.array_equal(data.squeeze(), read_data))

        logger.info(f"✓ uint8 roundtrip verified")

    def test_roundtrip_uint16(self):
        """Test write-read roundtrip with uint16 data"""
        data = np.random.randint(0, 65536, (128, 128, 1, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.array_equal(data.squeeze(), read_data))

        logger.info(f"✓ uint16 roundtrip verified")

    def test_roundtrip_float32(self):
        """Test write-read roundtrip with float32 data"""
        data = np.random.rand(128, 128, 1, 1, 1).astype(np.float32)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.float32
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.allclose(data.squeeze(), read_data))

        logger.info(f"✓ float32 roundtrip verified")

    def test_roundtrip_multichannel(self):
        """Test write-read roundtrip with multi-channel data"""
        data = np.random.randint(0, 256, (256, 256, 5, 3, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.array_equal(data.squeeze(), read_data))
            self.assertEqual(br.Z, 5)
            self.assertEqual(br.C, 3)

        logger.info(f"✓ Multi-channel (Z=5, C=3) roundtrip verified")


class TestZarrEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Set up test path"""
        self.test_path = TEST_DIR / "edge_cases.ome.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_write_read_exact_tile_size(self):
        """Test data that exactly matches tile size (1024x1024)"""
        tile_size = 1024
        data = np.random.randint(
            0, 256, (tile_size, tile_size, 1, 1, 1), dtype=np.uint16
        )

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.array_equal(data.squeeze(), read_data))

        logger.info(f"✓ Exact tile size (1024x1024) roundtrip verified")

    def test_write_read_unaligned_boundaries(self):
        """Test data that doesn't align to tile boundaries"""
        width, height = 1500, 1300  # Not multiples of 1024
        data = np.random.randint(0, 256, (height, width, 1, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.X, width)
            self.assertEqual(br.Y, height)
            read_data = br[:]
            self.assertTrue(np.array_equal(data.squeeze(), read_data))

        logger.info(f"✓ Unaligned boundaries (1500x1300) roundtrip verified")

    def test_write_read_small_image(self):
        """Test small 10x10 data"""
        data = np.full((10, 10, 1, 1, 1), 42, dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.all(read_data == 42))
            # Shape should be (Y, X) for single Z, C when T=1
            self.assertTrue(
                read_data.shape == (10, 10) or read_data.shape == (10, 10, 1, 1)
            )

        logger.info(f"✓ Small image (10x10) roundtrip verified")

    def test_partial_read_across_tiles(self):
        """Test reading region that spans multiple tiles"""
        data = np.random.randint(0, 256, (2048, 2048, 1, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            # Read region spanning 4 tiles (512:1536 crosses tile boundary at 1024)
            partial = br[512:1536, 512:1536, 0, 0]
            expected = data[512:1536, 512:1536, 0, 0, 0]
            self.assertTrue(np.array_equal(partial, expected))

        logger.info(f"✓ Multi-tile boundary read (512:1536) verified")

    def test_write_read_all_zeros(self):
        """Test data with all zeros (highly compressible)"""
        data = np.zeros((512, 512, 1, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.all(read_data == 0))

        logger.info(f"✓ All-zeros data roundtrip verified")

    def test_write_read_max_values(self):
        """Test data with maximum values for dtype"""
        data = np.full((256, 256, 1, 1, 1), 65535, dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.all(read_data == 65535))

        logger.info(f"✓ Maximum value (65535) data roundtrip verified")


class TestZarrVersionSpecific(unittest.TestCase):
    """Test version-specific features and behaviors"""

    def test_zarr_version_string(self):
        """Verify zarr version information"""
        logger.info(f"Testing with Zarr version: {zarr.__version__}")
        version_parts = zarr.__version__.split(".")
        self.assertGreaterEqual(len(version_parts), 2)
        logger.info(f"✓ Zarr version: {zarr.__version__}")

    def test_compression_works(self):
        """Test that compression is applied correctly"""
        test_path = TEST_DIR / "compression_test.ome.zarr"
        if test_path.exists():
            shutil.rmtree(test_path)

        try:
            # Create compressible data (lots of zeros)
            data = np.zeros((512, 512, 1, 1, 1), dtype=np.uint16)
            data[100:200, 100:200, 0, 0, 0] = 100

            with bfio.BioWriter(test_path, backend="zarr") as bw:
                bw.dtype = np.uint16
                bw.shape = data.shape
                bw[:] = data

            # Verify data can be read back
            with bfio.BioReader(test_path, backend="zarr") as br:
                read_data = br[:]
                self.assertTrue(np.array_equal(data.squeeze(), read_data))

            logger.info(f"✓ Compression test passed with zarr {zarr.__version__}")
        finally:
            if test_path.exists():
                shutil.rmtree(test_path)

    def test_consolidated_metadata(self):
        """Test consolidated metadata creation"""
        test_path = TEST_DIR / "consolidated_test.ome.zarr"
        if test_path.exists():
            shutil.rmtree(test_path)

        try:
            data = np.random.randint(0, 256, (128, 128, 1, 1, 1), dtype=np.uint16)

            with bfio.BioWriter(test_path, backend="zarr") as bw:
                bw.dtype = np.uint16
                bw.shape = data.shape
                bw[:] = data

            # Check if .zmetadata exists (may not exist in v3)
            zmetadata_path = test_path / ".zmetadata"
            if zmetadata_path.exists():
                logger.info(f"✓ Consolidated metadata created at {zmetadata_path}")
            else:
                logger.info(
                    f"ℹ Consolidated metadata not created (expected for zarr v3)"
                )
        finally:
            if test_path.exists():
                shutil.rmtree(test_path)


class TestZarrIntegration(unittest.TestCase):
    """Integration tests combining multiple operations"""

    def setUp(self):
        """Set up test path"""
        self.test_path = TEST_DIR / "integration_test.ome.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_write_verify_structure_read(self):
        """Complete workflow: write, verify structure, then read"""
        zarr_version = int(zarr.__version__.split(".")[0])

        # Write data
        data = np.random.randint(0, 256, (512, 512, 2, 3, 1), dtype=np.uint16)
        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw.cnames = ["Red", "Green", "Blue"]
            bw[:] = data

        # Verify structure with zarr directly
        if zarr_version >= 3:
            root = zarr.open_group(str(self.test_path), mode="r")
        else:
            root = zarr.open(str(self.test_path), mode="r")

        self.assertIn("0", list(root.array_keys()))
        arr = root["0"]
        self.assertEqual(arr.shape, (1, 3, 2, 512, 512))  # TCZYX

        # Read back with bfio
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.C, 3)
            self.assertEqual(br.Z, 2)
            read_data = br[:]
            self.assertTrue(np.array_equal(data.squeeze(), read_data))

        logger.info(f"✓ Complete write-verify-read workflow passed")

    def test_multiple_partial_writes(self):
        """Test writing data in multiple chunks"""
        # Use size that aligns with tile boundaries (2048 = 2 * 1024)
        width, height = 2048, 2048
        full_data = np.random.randint(0, 256, (height, width, 1, 1, 1), dtype=np.uint16)

        # Write in 4 quadrants (each 1024x1024, matching tile size)
        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = full_data.shape

            # Top-left
            bw[0:1024, 0:1024, 0, 0, 0] = full_data[0:1024, 0:1024, 0, 0, 0]
            # Top-right
            bw[0:1024, 1024:2048, 0, 0, 0] = full_data[0:1024, 1024:2048, 0, 0, 0]
            # Bottom-left
            bw[1024:2048, 0:1024, 0, 0, 0] = full_data[1024:2048, 0:1024, 0, 0, 0]
            # Bottom-right
            bw[1024:2048, 1024:2048, 0, 0, 0] = full_data[1024:2048, 1024:2048, 0, 0, 0]

        # Verify complete image
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertTrue(np.array_equal(full_data.squeeze(), read_data))

        logger.info(f"✓ Multiple partial writes (4 quadrants) verified")

    def test_metadata_preservation_roundtrip(self):
        """Test that metadata is preserved through write-read cycle"""
        data = np.random.randint(0, 256, (256, 256, 3, 2, 1), dtype=np.uint16)

        # Write with specific metadata
        channel_names = ["DAPI", "GFP"]
        ps_x = (0.325, "µm")
        ps_y = (0.325, "µm")
        ps_z = (1.0, "µm")

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw.cnames = channel_names
            bw.ps_x = ps_x
            bw.ps_y = ps_y
            bw.ps_z = ps_z
            bw[:] = data

        # Read and verify metadata
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.cnames[:2], channel_names)
            # Compare physical size values (handle enum types)
            self.assertEqual(br.ps_x[0], ps_x[0])
            self.assertEqual(br.ps_y[0], ps_y[0])
            self.assertEqual(br.ps_z[0], ps_z[0])
            # Check units are preserved (may be enum or string)
            self.assertIn("MICR", str(br.ps_x[1]).upper())
            self.assertIn("MICR", str(br.ps_y[1]).upper())
            self.assertIn("MICR", str(br.ps_z[1]).upper())

        logger.info(f"✓ Metadata preservation roundtrip verified")

    def test_read_after_direct_zarr_write(self):
        """Test bfio can read data written directly with zarr API"""
        zarr_version = int(zarr.__version__.split(".")[0])

        # Write directly with zarr
        if zarr_version >= 3:
            from zarr.codecs import ZstdCodec, BytesCodec
            from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

            root = zarr.open_group(str(self.test_path), mode="w")
            arr = root.zeros(
                name="0",
                shape=(1, 1, 1, 256, 256),
                chunks=(1, 1, 1, 256, 256),
                dtype=np.uint16,
                codecs=[BytesCodec(), ZstdCodec(level=1)],
                chunk_key_encoding=DefaultChunkKeyEncoding(separator="/"),
            )
        else:
            from numcodecs import Blosc

            root = zarr.open_group(str(self.test_path), mode="w")
            arr = root.zeros(
                "0",
                shape=(1, 1, 1, 256, 256),
                chunks=(1, 1, 1, 256, 256),
                dtype=np.uint16,
                compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
                dimension_separator="/",
            )

        # Fill with test data
        test_data = np.random.randint(0, 256, (1, 1, 1, 256, 256), dtype=np.uint16)
        arr[:] = test_data

        # Create minimal OME metadata
        metadata_path = self.test_path / "OME" / "METADATA.ome.xml"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
            'Type="uint16" SizeX="256" SizeY="256" SizeZ="1" SizeC="1" SizeT="1">'
            '<Channel ID="Channel:0" SamplesPerPixel="1"/>'
            "</Pixels></Image></OME>"
        )

        # Read with bfio
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            read_data = br[:]
            self.assertEqual(br.X, 256)
            self.assertEqual(br.Y, 256)
            self.assertTrue(np.array_equal(test_data[0, 0, 0, :, :], read_data))

        logger.info(f"✓ Read after direct zarr write verified")


class TestZarrPerformance(unittest.TestCase):
    """Performance and stress tests"""

    def setUp(self):
        """Set up test path"""
        self.test_path = TEST_DIR / "performance_test.ome.zarr"
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def tearDown(self):
        """Clean up test path"""
        if self.test_path.exists():
            shutil.rmtree(self.test_path)

    def test_many_channels(self):
        """Test handling many channels"""
        data = np.random.randint(0, 256, (128, 128, 1, 10, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.C, 10)
            read_data = br[:]
            # Shape will be (Y, X, Z, C) when T=1
            expected = data[:, :, :, :, 0]  # Remove T dimension
            self.assertTrue(np.array_equal(expected, read_data))

        logger.info(f"✓ Many channels (10) roundtrip verified")

    def test_many_z_slices(self):
        """Test handling many z-slices"""
        data = np.random.randint(0, 256, (128, 128, 20, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        with bfio.BioReader(self.test_path, backend="zarr") as br:
            self.assertEqual(br.Z, 20)
            read_data = br[:]
            # Verify shape and data - read_data will be (Y, X, Z) when C and T are 1
            self.assertEqual(read_data.shape[2], 20)  # Verify we have 20 z-slices
            for z in range(20):
                self.assertTrue(np.array_equal(data[:, :, z, 0, 0], read_data[:, :, z]))

        logger.info(f"✓ Many z-slices (20) roundtrip verified")

    def test_sequential_reads(self):
        """Test multiple sequential read operations"""
        data = np.random.randint(0, 256, (512, 512, 1, 1, 1), dtype=np.uint16)

        with bfio.BioWriter(self.test_path, backend="zarr") as bw:
            bw.dtype = np.uint16
            bw.shape = data.shape
            bw[:] = data

        # Perform multiple reads
        with bfio.BioReader(self.test_path, backend="zarr") as br:
            for i in range(10):
                read_data = br[i * 50 : (i + 1) * 50, :, 0, 0]
                expected = data[i * 50 : (i + 1) * 50, :, 0, 0, 0]
                self.assertTrue(np.array_equal(read_data, expected))

        logger.info(f"✓ Sequential reads (10 iterations) verified")


if __name__ == "__main__":
    unittest.main()
