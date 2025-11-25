# -*- coding: utf-8 -*-
# import core packages
import json
import logging
from pathlib import Path
from typing import Dict
import shutil


# Third party packages
import ome_types
from xml.etree import ElementTree as ET


from bfiocpp import TSReader, TSWriter, Seq, FileType, get_ome_xml
import bfio.base_classes
from bfio.utils import clean_ome_xml_for_known_issues
import zarr


class TensorstoreReader(bfio.base_classes.TSAbstractReader):
    logger = logging.getLogger("bfio.backends.TensorstoreReader")

    _rdr = None
    _offsets_bytes = None
    _STATE_DICT = [
        "_metadata",
        "frontend",
        "X",
        "Y",
        "Z",
        "C",
        "T",
        "data_type",
        "_file_path",
        "_file_type",
        "_axes_list",
    ]

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug("__init__(): Initializing _rdr (TSReader)...")
        extension = "".join(self.frontend._file_path.suffixes)
        if extension.endswith(".ome.tif") or extension.endswith(".ome.tiff"):
            # # check if it satisfies all the condition for python backend
            self._file_type = FileType.OmeTiff
            self._file_path = str(self.frontend._file_path.resolve())
            self._axes_list = ""
        elif extension.endswith(".zarr"):
            # if path exists, make sure it is a directory
            if not Path.is_dir(self.frontend._file_path):
                raise ValueError(
                    "this filetype is not supported by tensorstore backend"
                )
            else:
                self._file_path, self._axes_list = self.get_zarr_array_info()
                self._file_type = FileType.OmeZarr

        self._rdr = TSReader(self._file_path, self._file_type, self._axes_list)
        self.X = self._rdr._X
        self.Y = self._rdr._Y
        self.Z = self._rdr._Z
        self.C = self._rdr._C
        self.T = self._rdr._T
        self.data_type = self._rdr._datatype

    def get_zarr_array_info(self):
        self.logger.debug(f"Level is {self.frontend.level}")

        # Detect zarr version
        zarr_version = int(zarr.__version__.split('.')[0])
        self.logger.debug(f"Zarr version: {zarr.__version__}")

        root = None
        root_path = self.frontend._file_path
        try:
            if zarr_version >= 3:
                root = zarr.open_group(str(root_path.resolve()), mode="r")
            else:
                root = zarr.open(str(root_path.resolve()), mode="r")
        except Exception:
            # a workaround for pre-compute slide output directory structure
            # Handle both zarr v2 and v3 exceptions
            root_path = self.frontend._file_path / "data.zarr"
            if zarr_version >= 3:
                root = zarr.open_group(str(root_path.resolve()), mode="r")
            else:
                root = zarr.open(str(root_path.resolve()), mode="r")

        # Check type compatibility for both zarr v2 and v3
        if zarr_version >= 3:
            is_array = isinstance(root, zarr.Array)
            is_group = isinstance(root, zarr.Group)
        else:
            is_array = isinstance(root, zarr.core.Array)
            is_group = isinstance(root, zarr.hierarchy.Group)

        axes_list = ""
        if self.frontend.level is None:
            if is_array:
                return str(root_path.resolve()), axes_list
            elif is_group:
                #  the top level is a group, check if this has any arrays
                num_arrays = len(sorted(root.array_keys()))
                if num_arrays > 0:
                    array_key = next(root.array_keys())
                    root_path = root_path / str(array_key)
                    try:
                        axes_metadata = root.attrs["multiscales"][0]["axes"]
                        axes_list = "".join(
                            axes["name"].upper() for axes in axes_metadata
                        )
                    except KeyError:
                        self.logger.warning(
                            "Unable to find multiscales metadata. Z, C and T "
                            + "dimensions might be incorrect."
                        )

                    return str(root_path.resolve()), axes_list
                else:
                    # need to go one more level
                    group_key = next(root.group_keys())
                    root = root[group_key]
                    try:
                        axes_metadata = root.attrs["multiscales"][0]["axes"]
                        axes_list = "".join(
                            axes["name"].upper() for axes in axes_metadata
                        )
                    except KeyError:
                        self.logger.warning(
                            "Unable to find multiscales metadata. Z, C and T "
                            + "dimensions might be incorrect."
                        )

                    array_key = next(root.array_keys())
                    root_path = root_path / str(group_key) / str(array_key)
                    return str(root_path.resolve()), axes_list
            else:
                return str(root_path.resolve()), axes_list
        else:
            if is_array:
                self.close()
                raise ValueError(
                    "Level is specified but the zarr file does not contain "
                    + "multiple resoulutions."
                )
            elif is_group:
                if len(sorted(root.array_keys())) > self.frontend.level:
                    root_path = root_path / str(self.frontend.level)
                    try:
                        axes_metadata = root.attrs["multiscales"][0]["axes"]
                        axes_list = "".join(
                            axes["name"].upper() for axes in axes_metadata
                        )
                    except KeyError:
                        self.logger.warning(
                            "Unable to find multiscales metadata. Z, C and T "
                            + "dimensions might be incorrect."
                        )
                    return str(root_path.resolve()), axes_list
                else:
                    raise ValueError(
                        "The zarr file does not contain resolution "
                        + "level {}.".format(self.frontend.level)
                    )
            else:
                raise ValueError(
                    "The zarr file does not contain resolution level {}.".format(
                        self.frontend.level
                    )
                )

    def __getstate__(self) -> Dict:
        state_dict = {n: getattr(self, n) for n in self._STATE_DICT}

        return state_dict

    def __setstate__(self, state) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self._rdr = TSReader(self._file_path, self._file_type, self._axes_list)

    def read_metadata(self):

        self.logger.debug("read_metadata(): Reading metadata...")
        if self._file_type == FileType.OmeTiff:
            return self.read_tiff_metadata()
        if self._file_type == FileType.OmeZarr:
            return self.read_zarr_metadata()

    def read_image(self, X, Y, Z, C, T):

        cols = Seq(X[0], X[-1] - 1, 1)
        rows = Seq(Y[0], Y[-1] - 1, 1)
        layers = Seq(Z[0], Z[-1] - 1, 1)
        if len(C) == 1:
            channels = Seq(C[0], C[0], 1)
        else:
            channels = Seq(C[0], C[-1], 1)
        if len(T) == 1:
            tsteps = Seq(T[0], T[0], 1)
        else:
            tsteps = Seq(T[0], T[-1], 1)

        return self._rdr.data(rows, cols, layers, channels, tsteps)

    def close(self):
        pass

    def __del__(self):
        self.close()

    def read_tiff_metadata(self):
        self.logger.debug("read_tiff_metadata(): Reading metadata...")
        if self._metadata is None:
            try:
                self._metadata = ome_types.from_xml(
                    get_ome_xml(str(self.frontend._file_path)), validate=False
                )
            except (ET.ParseError, ValueError):
                if self.frontend.clean_metadata:
                    cleaned = clean_ome_xml_for_known_issues(
                        get_ome_xml(str(self.frontend._file_path))
                    )
                    self._metadata = ome_types.from_xml(cleaned, validate=False)
                    self.logger.warning(
                        "read_metadata(): OME XML required reformatting."
                    )
                else:
                    raise

        return self._metadata

    def read_zarr_metadata(self):
        self.logger.debug("read_zarr_metadata(): Reading metadata...")
        if self._metadata is None:

            metadata_path = self.frontend._file_path.joinpath("METADATA.ome.xml")

            if not metadata_path.exists():
                # try to look for OME directory
                metadata_path = self.frontend._file_path.joinpath("OME").joinpath(
                    "METADATA.ome.xml"
                )
            if metadata_path.exists():
                if self._metadata is None:
                    with open(metadata_path) as fr:
                        metadata = fr.read()

                    try:
                        self._metadata = ome_types.from_xml(metadata, validate=False)
                    except ET.ParseError:
                        if self.frontend.clean_metadata:
                            cleaned = clean_ome_xml_for_known_issues(metadata)
                            self._metadata = ome_types.from_xml(cleaned, validate=False)
                            self.logger.warning(
                                "read_metadata(): OME XML required reformatting."
                            )
                        else:
                            raise

                if self.frontend.level is not None:
                    self._metadata.images[0].pixels.size_x = self._rdr._X
                    self._metadata.images[0].pixels.size_y = self._rdr._Y

                return self._metadata
            else:
                # Couldn't find OMEXML metadata, scrape metadata from file
                omexml = ome_types.model.OME.model_construct()
                ome_dtype = self._rdr._datatype
                if ome_dtype == "float64":
                    ome_dtype = "double"
                elif ome_dtype == "float32":
                    ome_dtype = "float"
                else:
                    pass
                # this is speculation, since each array in a group, in theory,
                # can have distinct properties
                ome_dim_order = ome_types.model.Pixels_DimensionOrder.XYZCT
                size_x = self._rdr._X
                size_y = self._rdr._Y
                size_z = self._rdr._Z
                size_c = self._rdr._C
                size_t = self._rdr._T

                ome_pixel = ome_types.model.Pixels(
                    dimension_order=ome_dim_order,
                    big_endian=False,
                    size_x=size_x,
                    size_y=size_y,
                    size_z=size_z,
                    size_c=size_c,
                    size_t=size_t,
                    channels=[],
                    type=ome_dtype,
                )

                for i in range(ome_pixel.size_c):
                    ome_pixel.channels.append(ome_types.model.Channel())

                omexml.images.append(
                    ome_types.model.Image(
                        name=Path(self.frontend._file_path).name, pixels=ome_pixel
                    )
                )

                self._metadata = omexml
                return self._metadata


class TensorstoreWriter(bfio.base_classes.TSAbstractWriter):
    logger = logging.getLogger("bfio.backends.TensorstoreWriter")

    def __init__(self, frontend):

        super().__init__(frontend)

        self._init_writer()

    def _init_writer(self):
        """_init_writer Initializes file writing.

        This method is called exactly once per object. Once it is called,
        all other methods of setting metadata will throw an error.

        NOTE: For Zarr, it is not explicitly necessary to make the file
              read-only once writing has begun. Thus functionality is mainly
              incorporated to remain consistent with the OME TIFF formats.
              In the future, it may be reasonable to not enforce read-only

        """

        if self.frontend._file_path.exists():
            shutil.rmtree(self.frontend._file_path)

        # Tensorstore writer currently only supports zarr
        if not self.frontend._file_path.name.endswith(".zarr"):
            raise ValueError("File type must be zarr to use tensorstore writer.")

        shape = (
            self.frontend.T,
            self.frontend.C,
            self.frontend.Z,
            self.frontend.Y,
            self.frontend.X,
        )

        self._writer = TSWriter(
            str(self.frontend._file_path.joinpath("0").resolve()),
            shape,
            (1, 1, 1, self.frontend._TILE_SIZE, self.frontend._TILE_SIZE),
            self.frontend.dtype,
            "TCZYX",
        )

        self.write_metadata()

    def write_metadata(self):

        # Create the metadata
        metadata_path = (
            Path(self.frontend._file_path).joinpath("OME").joinpath("METADATA.ome.xml")
        )

        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as fw:
            fw.write(str(self.frontend._metadata.to_xml()))

        with open(self.frontend._file_path.joinpath(".zgroup"), "w") as f:
            f.write('{\n\t"zarr_format": 2\n}')

        zarr_attrs = {
            "multiscales": [
                {
                    "version": "0.1",
                    "name": self.frontend._file_path.name,
                    "datasets": [{"path": "0"}],
                    "metadata": {"method": "mean"},
                }
            ]
        }
        with open(self.frontend._file_path.joinpath(".zattrs"), "w") as f:
            json.dump(zarr_attrs, f, indent=4)

        # This is recommended to do for cloud storage to increase read/write
        # speed, but it also increases write speed locally when threading.
        zarr.consolidate_metadata(str(self.frontend._file_path.resolve()))

    def write_image(self, X, Y, Z, C, T, image):

        cols = Seq(X[0], X[-1] - 1, 1)
        rows = Seq(Y[0], Y[-1] - 1, 1)
        layers = Seq(Z[0], Z[-1] - 1, 1)

        if len(C) == 1:
            channels = Seq(C[0], C[0], 1)
        else:
            channels = Seq(C[0], C[-1], 1)

        if len(T) == 1:
            tsteps = Seq(T[0], T[0], 1)
        else:
            tsteps = Seq(T[0], T[-1], 1)

        self._writer.write_image_data(
            image.flatten(), rows, cols, layers, channels, tsteps
        )

    def close(self):
        pass
