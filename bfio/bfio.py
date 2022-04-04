import json
import logging
import struct
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy
import ome_types

from bfio import JARS, LOGBACK, backends
from bfio.base_classes import BioBase

try:
    import jpype
    import jpype.imports

    def start() -> str:
        """Start the jvm.

        This function starts the jvm and imports all the necessary Java classes
        to read images using the Bioformats toolbox.

        Return:
            The Bio-Formats JAR version.
        """
        if jpype.isJVMStarted():
            from loci.formats import FormatTools

            JAR_VERSION = FormatTools.VERSION
            return JAR_VERSION

        logging.getLogger("bfio.start").info("Starting the jvm.")
        jpype.startJVM(f"-Dlogback.configurationFile={LOGBACK}", classpath=JARS)

        from loci.formats import FormatTools

        JAR_VERSION = FormatTools.VERSION

        logging.getLogger("bfio.start").info(
            "bioformats_package.jar version = {}".format(JAR_VERSION)
        )

        return JAR_VERSION

except ModuleNotFoundError:

    def start():  # NOQA: D103
        raise ModuleNotFoundError("Error importing jpype or a loci_tools.jar class.")


class BioReader(BioBase):
    """Read supported image formats using Bioformats.

    This class handles file reading of multiple formats. It can read files from
    any Bioformats supported file format, but is specially optimized for
    handling the OME tiled tiff format.

    There are three backends: ``java``, ``python``, and ``zarr``. The ``java``
    backend directly uses Bio-Formats for file reading, and can read any format
    that is supported by Bio-Formats. The ``python`` backend will only read
    images in OME Tiff format with tile tags set to 1024x1024, and is
    significantly faster than the "java" backend for reading these types of tiff
    files. The ``zarr`` backend will only read OME Zarr files.

    File reading and writing are multi-threaded by default, except for the
    ``java`` backend which does not currently support threading. Half of the
    available CPUs detected by multiprocessing.cpu_count() are used to read
    an image.

    For for information, visit the Bioformats page:
    https://www.openmicroscopy.org/bio-formats/

    Note:
        In order to use the ``java`` backend, jpype must be installed.
    """

    logger = logging.getLogger("bfio.bfio.BioReader")
    _STATE_DICT = [
        "_metadata",
        "_DIMS",
        "_file_path",
        "max_workers",
        "_backend_name",
        "clean_metadata",
        "_read_only",
        "_backend",
    ]

    def __init__(
        self,
        file_path: typing.Union[str, Path],
        max_workers: typing.Union[int, None] = None,
        backend: typing.Optional[str] = None,
        clean_metadata: bool = True,
    ) -> None:
        """Initialize the BioReader.

        Args:
            file_path: Path to file to read
            max_workers: Number of threads used to read and image. *Default is
                half the number of detected cores.*
            backend: Can be ``python``, ``java``, or ``zarr``. If None, then
                BioReader will try to autodetect the proper backend.
                *Default is python.*
            clean_metadata: Will try to reformat poorly formed OME XML metadata if True.
                If False, will throw an error if the metadata is poorly formed.
                *Default is True.*
        """
        # Initialize BioBase
        super(BioReader, self).__init__(
            file_path, max_workers=max_workers, backend=backend
        )

        self.clean_metadata = clean_metadata

        # Ensure backend is supported
        self.logger.debug("Starting the backend...")
        if self._backend_name == "python":
            self._backend = backends.PythonReader(self)
        elif self._backend_name == "java":
            try:
                self._backend = backends.JavaReader(self)
            except Exception as err:
                if repr(err).split("(")[0] in [
                    "UnknownFormatException",
                    "MissingLibraryException",
                ]:
                    self.logger.error(
                        "UnknownFormatException: Did not recognize file format: "
                        + f"{file_path}"
                    )
                    raise TypeError(
                        "UnknownFormatException: Did not recognize file format: "
                        + f"{file_path}"
                    )
                else:
                    raise
        elif self._backend_name == "zarr":
            self._backend = backends.ZarrReader(self)

            if max_workers == 2:
                self.max_workers = 1
                self.logger.debug(
                    "Setting max_workers to 1, since max_workers==2 runs slower."
                    + "To change back, set the object property."
                )
        else:
            raise ValueError('backend must be "python", "java", or "zarr"')
        self.logger.debug("Finished initializing the backend.")

        # Preload the metadata
        self._metadata = self._backend.read_metadata()

        # Get dims to speed up validation checks
        self._DIMS = {"X": self.X, "Y": self.Y, "Z": self.Z, "C": self.C, "T": self.T}

    def __getstate__(self) -> typing.Dict:

        state_dict = {n: getattr(self, n) for n in self._STATE_DICT}

        return state_dict

    def __setstate__(self, state) -> None:

        assert all(n in self._STATE_DICT for n in state.keys())
        assert all(n in state.keys() for n in self._STATE_DICT)

        for k, v in state.items():
            setattr(self, k, v)

        self._backend.frontend = self

    def __getitem__(self, keys: typing.Union[tuple, slice]) -> numpy.ndarray:
        """Image loading using numpy-like indexing.

        This is an abbreviated method of accessing the :attr:`~.read` method,
        where a portion of the image will be loaded using numpy-like slicing
        syntax. Up to 5 dimensions can be designated depending on the number of
        available dimensions in the image array (Y, X, Z, C, T).

        Note:
            Not all methods of indexing can be used, and some indexing will lead
            to unexpected results. For example, logical indexing cannot be used,
            and step sizes in slice objects is ignored for the first three
            indices. This means and index such as ``[0:100:2,0:100:2,0,0,0]``
            will return a 100x100x1x1x1 numpy array.

        Args:
            keys: numpy-like slicing used to load a
                section of an image.

        Returns:
            A numpy.ndarray where trailing empty dimensions are removed.

        Example:

            .. code-block:: python

                import bfio

                # Initialize the bioreader
                br = bfio.BioReader('Path/To/File.ome.tif')

                # Load and copy a 100x100 array of pixels
                a = br[:100,:100,:1,0,0]

                # Slice steps sizes are ignored for the first 3 indices, so this
                # returns the same as above
                a = br[0:100:2,0:100:2,0:1,0,0]

                # The last two dimensions can receive a tuple or list as input
                # Load the first and third channel
                a = br[:100,100,0:1,(0,2),0]

                # If the file is 3d, load the first 10 z-slices
                b = br[...,:10,0,0]
        """
        ind = self._parse_slice(keys)

        return self.read(**ind)

    def read(
        self,
        X: typing.Union[list, tuple, None] = None,
        Y: typing.Union[list, tuple, None] = None,
        Z: typing.Union[list, tuple, int, None] = None,
        C: typing.Union[list, tuple, int, None] = None,
        T: typing.Union[list, tuple, int, None] = None,
    ) -> numpy.ndarray:
        """Read the image.

        Read the all or part of the image. A n-dimmensional numpy.ndarray is
        returned such that all trailing empty dimensions will be removed.

        For example, if an image is read and it represents an xz plane, then the
        shape will be [1,m,n].

        Args:
            X: The (min,max) range of pixels to load along the x-axis (columns).
                If None, loads the full range. *Defaults to None.*
            Y: The (min,max) range of pixels to load along the y-axis (rows). If
                None, loads the full range. *Defaults to None.*
            Z: The (min,max) range of pixels to load along the z-axis (depth).
                Alternatively, an integer can be passed to select a single
                z-plane. If None, loads the full range. *Defaults to None.*
            C: Values indicating channel indices to load. If None, loads the
                full range. *Defaults to None.*
            T: Values indicating timepoints to load. If None, loads the full
                range. *Defaults to None.*

        Returns:
            A 5-dimensional numpy array.
        """
        # Validate inputs
        X = self._val_xyz(X, "X")
        Y = self._val_xyz(Y, "Y")
        Z = self._val_xyz(Z, "Z")
        C = self._val_ct(C, "C")
        T = self._val_ct(T, "T")

        # Define tile bounds
        X_tile_start = (X[0] // self._TILE_SIZE) * self._TILE_SIZE
        Y_tile_start = (Y[0] // self._TILE_SIZE) * self._TILE_SIZE
        X_tile_end = numpy.ceil(X[1] / self._TILE_SIZE).astype(int) * self._TILE_SIZE
        Y_tile_end = numpy.ceil(Y[1] / self._TILE_SIZE).astype(int) * self._TILE_SIZE
        X_tile_shape = X_tile_end - X_tile_start
        Y_tile_shape = Y_tile_end - Y_tile_start
        Z_tile_shape = Z[1] - Z[0]

        # Determine if enough planes will be loaded to benefit from tiling
        # There is a tradeoff between fast storing tiles and reshaping later versus
        # slow storing tiles without reshaping.
        # fast storing tiles is aligned memory, where slow storing is unaligned
        load_tiles = 10 * (
            X_tile_shape * Y_tile_shape / 1024**2 + 1
        ) < Z_tile_shape * len(C) * len(T)

        # Initialize the output for zarr and java
        if self._backend_name != "python":
            output = numpy.zeros(
                [Y_tile_shape, X_tile_shape, Z_tile_shape, len(C), len(T)],
                dtype=self.dtype,
            )

        # Initialize the output for python
        # We use a different matrix shape for loading images to reduce memory copy time
        # TODO: Should use this same scheme for the other readers to reduce copy times
        else:
            if load_tiles:
                y_tile = min(Y[1] - Y[0], 1024)
                x_tile = min(X[1] - X[0], 1024)
                output = numpy.zeros(
                    [
                        Z_tile_shape,
                        len(C),
                        len(T),
                        Y_tile_shape // 1024,
                        X_tile_shape // 1024,
                        y_tile,
                        x_tile,
                    ],
                    dtype=self.dtype,
                    order="C",
                )
            else:
                output = numpy.zeros(
                    [Z_tile_shape, len(C), len(T), Y[1] - Y[0], X[1] - X[0]],
                    dtype=self.dtype,
                    order="C",
                )

        # Read the image
        self._backend.load_tiles = load_tiles
        self._backend.read_image(
            [X_tile_start, X_tile_end], [Y_tile_start, Y_tile_end], Z, C, T, output
        )

        # Reshape the arrays into expected format
        if self._backend_name == "python":
            if load_tiles:
                output = output.transpose(3, 5, 4, 6, 0, 1, 2)
                X_tile_shape = X_tile_shape if x_tile == 1024 else X[1] - X[0]
                Y_tile_shape = Y_tile_shape if y_tile == 1024 else Y[1] - Y[0]
                output = output.reshape(
                    Y_tile_shape, X_tile_shape, Z_tile_shape, len(C), len(T)
                )
            else:
                output = output.transpose(3, 4, 0, 1, 2)

        output = output[
            Y[0] - Y_tile_start : Y[1] - Y_tile_start,
            X[0] - X_tile_start : X[1] - X_tile_start,
            ...,
        ]
        while output.shape[-1] == 1 and output.ndim > 2:
            output = output[..., 0]

        return output

    def _fetch(self) -> numpy.ndarray:
        """Method for fetching image supertiles.

        This method is intended to be run within a thread, and grabs a chunk of
        the image according to the coordinates in a queue.

        Currently, this function will only grab the first Z, C, and T positions
        regardless of what Z, C, and T coordinate are provided to the function.
        This function will need to be changed in then future to account for
        this.

        If the first value in X or Y is negative, then the image is pre-padded
        with the number of pixels equal to the absolute value of the negative
        number.

        If the last value in X or Y is larger than the size of the image, then
        the image is post-padded with the difference between the number and the
        size of the image. Input coordinate are read from the _supertile_index
        Queue object. Output data is stored in the _raw_buffer Queue object.

        As soon as the method is executed, a boolean value is put into the
        _data_in_buffer Queue to indicate that data is either in the buffer or
        will be put into the buffer.

        Returns:
            An image supertile
        """
        self._data_in_buffer.put(True)
        X, Y, Z, C, T = self._supertile_index.get()

        # Attach the jvm to the thread if present
        self._backend.attach()

        # Determine padding if needed
        reflect_x = False
        x_min = X[0]
        x_max = X[1]
        y_min = Y[0]
        y_max = Y[1]
        prepad_x = 0
        postpad_x = 0
        prepad_y = 0
        postpad_y = 0
        if x_min < 0:
            prepad_x = abs(x_min)
            x_min = 0
        if y_min < 0:
            prepad_y = abs(y_min)
            y_min = 0
        if x_max > self.num_x():
            if x_min >= self.num_x():
                x_min = self._TILE_SIZE * ((self.num_x() - 1) // self._TILE_SIZE)
                reflect_x = True
            x_max = self.num_x()
            postpad_x = x_max - self.num_x()
        if y_max > self.num_y():
            y_max = self.num_y()
            postpad_y = y_max - self.num_y()

        # Read the image
        image = self.read([x_min, x_max], [y_min, y_max], [0, 1], [0], [0]).squeeze()
        if reflect_x:
            image = numpy.fliplr(image)

        # Pad the image if needed
        if sum(1 for p in [prepad_x, prepad_y, postpad_x, postpad_y] if p != 0) > 0:
            image = numpy.pad(
                image, ((prepad_y, postpad_y), (prepad_x, postpad_x)), mode="symmetric"
            )

        # Store the data in the bufferI
        self._raw_buffer.put(image)

        # Detach the jvm
        self._backend.detach()

        return image

    def _buffer_supertile(self, column_start: int, column_end: int):
        """_buffer_supertile Process the pixel buffer.

        Give the column indices of the data to process, and determine if the
        buffer needs to be processed. This method performs two operations on the
        buffer. First, it checks to see if data in the buffer can be shifted out
        of the buffer if it's already been processed, where data before
        column_start is assumed to have been processed. Second, this function
        loads data into the buffer if the image reader has made some available
        and there is room in _pixel_buffer for it.

        Args:
            column_start: First column index of data to be loaded
            column_end: Last column index of data to be loaded
        """
        # If the column indices are outside what is available in the buffer,
        # shift the buffer so more data can be loaded.
        if column_end - self._tile_x_offset >= self._TILE_SIZE:
            x_min = column_start - self._tile_x_offset
            x_max = self._pixel_buffer.shape[1] - x_min
            self._pixel_buffer[:, 0:x_max] = self._pixel_buffer[:, x_min:]
            self._pixel_buffer[:, x_max:] = 0
            self._tile_x_offset = column_start
            self._tile_last_column = numpy.argwhere(
                (self._pixel_buffer == 0).all(axis=0)
            )[0, 0]

        # Is there data in the buffer?
        if self._supertile_index.qsize() > 0 or self._data_in_buffer.qsize() > 0:

            # If there is data in the _raw_buffer, return if there isn't room to load
            # it into the _pixel_buffer
            if self._pixel_buffer.shape[1] - self._tile_last_column < self._TILE_SIZE:
                return

            image = self._raw_buffer.get()
            if self._tile_last_column == 0:
                self._pixel_buffer[: image.shape[0], : image.shape[1]] = image
                self._tile_last_column = image.shape[1]
                self._tile_x_offset = column_start
            else:
                self._pixel_buffer[
                    : image.shape[0],
                    self._tile_last_column : self._tile_last_column + image.shape[1],
                ] = image
                self._tile_last_column += image.shape[1]

            self._data_in_buffer.get()

    def _get_tiles(
        self,
        X: typing.List[typing.List[int]],
        Y: typing.List[typing.List[int]],
        Z: typing.List[typing.List[int]],
        C: typing.List[typing.List[int]],
        T: typing.List[typing.List[int]],
    ) -> numpy.ndarray:
        """_get_tiles Handle data buffering and tiling.

        This function returns tiles of data according to the input coordinates.
        The X, Y, Z, C, and T are lists of lists, where each internal list
        indicates a set of coordinates specifying the range of pixel values to
        grab from an image.

        Args:
            X: List of 2-tuples indicating the (min,max) range of pixels to load
                within a tile.
            Y: List of 2-tuples indicating the (min,max) range of pixels to load
                within a tile.
            Z: Placeholder, to be implemented.
            C: Placeholder, to be implemented.
            T: Placeholder, to be implemented.

        Returns:
            2-dimensional ndarray.
        """
        self._buffer_supertile(X[0][0], X[0][1])

        if X[-1][0] - self._tile_x_offset > 1024:
            split_ind = 0
            while X[split_ind][0] - self._tile_x_offset < 1024:
                split_ind += 1
        else:
            split_ind = len(X)

        # Tile the data
        num_rows = Y[0][1] - Y[0][0]
        num_cols = X[0][1] - X[0][0]
        num_tiles = len(X)
        images = numpy.zeros(
            (num_tiles, num_rows, num_cols, 1), dtype=self.pixel_type()
        )

        for ind in range(split_ind):
            images[ind, :, :, 0] = self._pixel_buffer[
                Y[ind][0] - self._tile_y_offset : Y[ind][1] - self._tile_y_offset,
                X[ind][0] - self._tile_x_offset : X[ind][1] - self._tile_x_offset,
            ]

        if split_ind != num_tiles:
            self._buffer_supertile(X[-1][0], X[-1][1])
            for ind in range(split_ind, num_tiles):
                images[ind, :, :, 0] = self._pixel_buffer[
                    Y[ind][0] - self._tile_y_offset : Y[ind][1] - self._tile_y_offset,
                    X[ind][0] - self._tile_x_offset : X[ind][1] - self._tile_x_offset,
                ]

        return images

    def __call__(
        self,
        tile_size: typing.Union[list, tuple],
        tile_stride: typing.Union[list, tuple, None] = None,
        batch_size: typing.Union[int, None] = None,
        channels: typing.List[int] = [0],
    ) -> typing.Iterable[typing.Tuple[numpy.ndarray, tuple]]:
        """Iterate through tiles of an image.

        The BioReader object can be called, and will act as an iterator to load
        tiles of an image. The iterator buffers the loading of pixels
        asynchronously to quickly deliver images of the appropriate size.

        Args:
            tile_size: A list/tuple of length 2, indicating the height and width
                of the tiles to return.
            tile_stride: A list/tuple of length 2, indicating the row and column
                stride size. If None, then tile_stride = tile_size. *Defaults to
                None.*
            batch_size: Number of tiles to return on each iteration. *Defaults
                to None, which is the smaller of 32 or the*
                :attr:`~.maximum_batch_size`
            channels: A placeholder. Only the first channel is ever loaded.
                *Defaults to [0].*

        Returns:
            A tuple containing a 4-d numpy array and a tuple containing a list
            of X,Y,Z,C,T indices. The numpy array has dimensions
            ``[tile_num,tile_size[0],tile_size[1],channels]``

        Example:
            .. code:: python

                from bfio import BioReader
                import matplotlib.pyplot as plt

                br = BioReader('/path/to/file')

                for tiles,ind in br(tile_size=[256,256],tile_stride=[200,200]):
                    for i in tiles.shape[0]:
                        print(
                            'Displaying tile with X,Y coords: {},{}'.format(
                                ind[i][0],ind[i][1]
                            )
                        )
                        plt.figure()
                        plt.imshow(tiles[ind,:,:,0].squeeze())
                        plt.show()

        """
        self._iter_tile_size = tile_size
        self._iter_tile_stride = tile_stride
        self._iter_batch_size = batch_size
        self._iter_channels = channels

        return self

    def __iter__(self):  # NOQA:C901

        tile_size = self._iter_tile_size
        tile_stride = self._iter_tile_stride
        batch_size = self._iter_batch_size
        channels = self._iter_channels

        if tile_size is None:
            raise SyntaxError(
                "Cannot directly iterate over a BioReader object."
                + "Call it (i.e. for i in bioreader(256,256))"
            )

        self._iter_tile_size = None
        self._iter_tile_stride = None
        self._iter_batch_size = None
        self._iter_channels = None

        # Ensure that the number of tiles does not exceed the supertile width
        if batch_size is None:
            batch_size = min([32, self.maximum_batch_size(tile_size, tile_stride)])
        else:
            assert batch_size <= self.maximum_batch_size(
                tile_size, tile_stride
            ), "batch_size must be less than or equal to {}.".format(
                self.maximum_batch_size(tile_size, tile_stride)
            )

        # input error checking
        assert len(tile_size) == 2, "tile_size must be a list with 2 elements"
        if tile_stride is not None:
            assert len(tile_stride) == 2, "stride must be a list with 2 elements"
        else:
            tile_stride = tile_size

        # calculate padding if needed
        if not (set(tile_size) & set(tile_stride)):
            xyoffset = [
                (tile_size[0] - tile_stride[0]) / 2,
                (tile_size[1] - tile_stride[1]) / 2,
            ]
            xypad = [
                (tile_size[0] - tile_stride[0]) / 2,
                (tile_size[1] - tile_stride[1]) / 2,
            ]
            xypad[0] = (
                xyoffset[0] + (tile_stride[0] - numpy.mod(self.Y, tile_stride[0])) / 2
            )
            xypad[1] = (
                xyoffset[1] + (tile_stride[1] - numpy.mod(self.X, tile_stride[1])) / 2
            )
            xypad = (
                (int(xyoffset[0]), int(2 * xypad[0] - xyoffset[0])),
                (int(xyoffset[1]), int(2 * xypad[1] - xyoffset[1])),
            )
        else:
            xyoffset = [0, 0]
            xypad = (
                (0, max([tile_size[0] - tile_stride[0], 0])),
                (0, max([tile_size[1] - tile_stride[1], 0])),
            )

        # determine supertile sizes
        y_tile_dim = int(numpy.ceil((self.Y - 1) / 1024))
        x_tile_dim = 1

        # Initialize the pixel buffer
        self._pixel_buffer = numpy.zeros(
            (y_tile_dim * 1024 + tile_size[0], 2 * x_tile_dim * 1024 + tile_size[1]),
            dtype=self.dtype,
        )
        self._tile_x_offset = -xypad[1][0]
        self._tile_y_offset = -xypad[0][0]

        # Generate the supertile loading order
        tiles = []
        y_tile_list = list(range(0, self.Y + xypad[0][1], 1024 * y_tile_dim))
        if y_tile_list[-1] != 1024 * y_tile_dim:
            y_tile_list.append(1024 * y_tile_dim)
        if y_tile_list[0] != xypad[0][0]:
            y_tile_list[0] = -xypad[0][0]
        x_tile_list = list(range(0, self.X + xypad[1][1], 1024 * x_tile_dim))
        if x_tile_list[-1] < self.X + xypad[1][1]:
            x_tile_list.append(x_tile_list[-1] + 1024)
        if x_tile_list[0] != xypad[1][0]:
            x_tile_list[0] = -xypad[1][0]
        for yi in range(len(y_tile_list) - 1):
            for xi in range(len(x_tile_list) - 1):
                y_range = [y_tile_list[yi], y_tile_list[yi + 1]]
                x_range = [x_tile_list[xi], x_tile_list[xi + 1]]
                tiles.append([x_range, y_range])
                self._supertile_index.put((x_range, y_range, [0, 1], [0], [0]))

        # Start the thread pool and start loading the first supertile
        thread_pool = ThreadPoolExecutor(self._max_workers)
        self._fetch_thread = thread_pool.submit(self._fetch)

        # generate the indices for each tile
        # TODO: modify this to grab more than just the first z-index
        X = []
        Y = []
        Z = []
        C = []
        T = []
        x_list = numpy.array(numpy.arange(-xypad[1][0], self.X, tile_stride[1]))
        y_list = numpy.array(numpy.arange(-xypad[0][0], self.Y, tile_stride[0]))
        for x in x_list:
            for y in y_list:
                X.append([x, x + tile_size[1]])
                Y.append([y, y + tile_size[0]])
                Z.append([0, 1])
                C.append(channels)
                T.append([0])

        # Set up batches
        batches = list(range(0, len(X), batch_size))

        # get the first batch
        b = min([batch_size, len(X)])
        index = (X[0:b], Y[0:b], Z[0:b], C[0:b], T[0:b])
        images = self._get_tiles(*index)

        # start looping through batches
        for bn in batches[1:]:
            # start the thread to get the next batch
            b = min([bn + batch_size, len(X)])
            self._tile_thread = thread_pool.submit(
                self._get_tiles, X[bn:b], Y[bn:b], Z[bn:b], C[bn:b], T[bn:b]
            )

            # Load another supertile if possible
            if self._supertile_index.qsize() > 0 and not self._fetch_thread.running():
                self._fetch_thread = thread_pool.submit(self._fetch)

            # return the curent set of images
            yield images, index

            # get the images from the thread
            index = (X[bn:b], Y[bn:b], Z[bn:b], C[bn:b], T[bn:b])
            images = self._tile_thread.result()

        thread_pool.shutdown()

        # return the last set of images
        yield images, index

    @classmethod
    def image_size(cls, filepath: Path):  # NOQA: C901
        """image_size Read image width and height from header.

        This class method only reads the header information of tiff files or the
        zarr array json to identify the image width and height. There are
        instances when the image dimensions may want to be known without
        actually loading the image, and reading only the header is considerably
        faster than loading bioformats just to read simple metadata information.

        If the file is not a TIFF or OME Zarr, returns width = height = -1.

        This code was adapted to only operate on tiff images and includes
        additional to read the header of little endian encoded BigTIFF files.
        The original code can be found at:
        https://github.com/shibukawa/imagesize_py

        Args:
            filepath: Path to tiff file
        Returns:
            Tuple of ints indicating width and height.

        """
        # Support strings input
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Handle a zarr file
        if filepath.name.endswith("ome.zarr"):
            with open(filepath.joinpath("0").joinpath(".zarray"), "r") as fr:
                zarray = json.load(fr)
                height = zarray["shape"][3]
                width = zarray["shape"][4]
            return width, height

        height = -1
        width = -1

        with open(str(filepath), "rb") as fhandle:
            head = fhandle.read(24)
            size = len(head)

            # handle big endian TIFF
            if size >= 8 and head.startswith(b"\x4d\x4d\x00\x2a"):
                offset = struct.unpack(">L", head[4:8])[0]
                fhandle.seek(offset)
                ifdsize = struct.unpack(">H", fhandle.read(2))[0]
                for i in range(ifdsize):
                    tag, datatype, count, data = struct.unpack(
                        ">HHLL", fhandle.read(12)
                    )
                    if tag == 256:
                        if datatype == 3:
                            width = int(data / 65536)
                        elif datatype == 4:
                            width = data
                        else:
                            raise ValueError(
                                "Invalid TIFF file:"
                                + "width column data type should be SHORT/LONG."
                            )
                    elif tag == 257:
                        if datatype == 3:
                            height = int(data / 65536)
                        elif datatype == 4:
                            height = data
                        else:
                            raise ValueError(
                                "Invalid TIFF file:"
                                + "height column data type should be SHORT/LONG."
                            )
                    if width != -1 and height != -1:
                        break
                if width == -1 or height == -1:
                    raise ValueError(
                        "Invalid TIFF file:"
                        + "width and/or height IDS entries are missing."
                    )
            # handle little endian Tiff
            elif size >= 8 and head.startswith(b"\x49\x49\x2a\x00"):
                offset = struct.unpack("<L", head[4:8])[0]
                fhandle.seek(offset)
                ifdsize = struct.unpack("<H", fhandle.read(2))[0]
                for i in range(ifdsize):
                    tag, datatype, count, data = struct.unpack(
                        "<HHLL", fhandle.read(12)
                    )
                    if tag == 256:
                        width = data
                    elif tag == 257:
                        height = data
                    if width != -1 and height != -1:
                        break
                if width == -1 or height == -1:
                    raise ValueError(
                        "Invalid TIFF file:"
                        + "width and/or height IDS entries are missing."
                    )
            # handle little endian BigTiff
            elif size >= 8 and head.startswith(b"\x49\x49\x2b\x00"):
                bytesize_offset = struct.unpack("<L", head[4:8])[0]
                if bytesize_offset != 8:
                    raise ValueError(
                        "Invalid BigTIFF file:"
                        + "Expected offset to be 8, found {} instead.".format(
                            bytesize_offset
                        )
                    )
                offset = struct.unpack("<Q", head[8:16])[0]
                fhandle.seek(offset)
                ifdsize = struct.unpack("<Q", fhandle.read(8))[0]
                for i in range(ifdsize):
                    tag, datatype, count, data = struct.unpack(
                        "<HHQQ", fhandle.read(20)
                    )
                    if tag == 256:
                        width = data
                    elif tag == 257:
                        height = data
                    if width != -1 and height != -1:
                        break
                if width == -1 or height == -1:
                    raise ValueError(
                        "Invalid BigTIFF file:"
                        + "width and/or height IDS entries are missing."
                    )

        return width, height


class BioWriter(BioBase):
    """BioWriter Write OME tiled tiff images.

    This class handles the writing OME tiled tif images. There is a Java backend
    version of this tool that directly interacts with the Bioformats java
    library directly, and is primarily used for testing. It is currently not
    possible to change the tile size (which is set to 1024x1024).

    Unlike the BioReader class, the properties of this class are settable until
    the first time the ``write`` method is called.

    For for information, visit the Bioformats page:
    https://www.openmicroscopy.org/bio-formats/

    Note:
        In order to use the ``java`` backend, jpype must be installed.
    """

    logger = logging.getLogger("bfio.bfio.BioWriter")

    def __init__(  # NOQA: C901
        self,
        file_path: typing.Union[str, Path],
        max_workers: typing.Union[int, None] = None,
        backend: typing.Optional[str] = None,
        metadata: typing.Union[ome_types.model.OME, None] = None,
        image: typing.Union[numpy.ndarray, None] = None,
        **kwargs,
    ) -> None:
        """Initialize a BioWriter.

        Args:
            file_path: Path to file to read
            max_workers: Number of threads used to read and image. *Default is
                half the number of detected cores.*
            backend: Must be ``python`` or ``java``. *Default is python.*
            metadata: This directly sets the ome tiff metadata using the OMEXML
                class if specified. *Defaults to None.*
            image: The metadata will be set based on the dimensions and data
                type of the numpy array specified by this keyword argument.
                Ignored if metadata is specified. *Defaults to None.*
            kwargs: Most BioWriter object properties can be passed as keyword
                arguments to initialize the image metadata. If the metadata
                argument is used, then keyword arguments are ignored.
        """
        super(BioWriter, self).__init__(
            file_path=file_path,
            max_workers=max_workers,
            backend=backend,
            read_only=False,
        )

        if metadata:
            assert metadata.__class__.__name__ == "OME"
            self._metadata = metadata.copy(deep=True)

            self._metadata.images[0].name = self._file_path.name
            self._metadata.images[
                0
            ].pixels.dimension_order = ome_types.model.pixels.DimensionOrder.XYZCT
        else:
            self._metadata = self._minimal_xml()

            if isinstance(image, numpy.ndarray):
                assert (
                    len(image.shape) <= 5
                ), "Image can be at most 5-dimensional (x,y,z,c,t)."
                self.spp = 1
                self.dtype = image.dtype
                for k, v in zip("YXZCT", image.shape):
                    setattr(self, k, v)

            elif kwargs:
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Ensure backend is supported
        if self._backend_name == "python":
            self._backend = backends.PythonWriter(self)
        elif self._backend_name == "java":
            self._backend = backends.JavaWriter(self)
        elif self._backend_name == "zarr":
            self._backend = backends.ZarrWriter(self)
        else:
            raise ValueError('backend must be "python", "java", or "zarr"')

        if not self._file_path.name.endswith(
            ".ome.tif"
        ) and not self._file_path.name.endswith(".ome.tif"):
            ValueError("The file extension must be .ome.tif or .ome.zarr")

        if len(self.metadata.images) > 1:
            self.logger.warning(
                "The BioWriter only writes single image "
                + "files, but the metadata has {} images. ".format(
                    len(self.metadata.images)
                )
                + "Setting the number of images to 1."
            )
            self.metadata.images = self.metadata.images[:1]

        # Get dims to speed up validation checks
        self._DIMS = {"X": self.X, "Y": self.Y, "Z": self.Z, "C": self.C, "T": self.T}

    def __setitem__(
        self, keys: typing.Union[tuple, slice], value: numpy.ndarray
    ) -> None:
        """Image saving using numpy-like indexing.

        This is an abbreviated method of accessing the :attr:`~.write` method,
        where a portion of the image will be saved using numpy-like slicing
        syntax. Up to 5 dimensions can be designated depending on the number of
        available dimensions in the image array (Y, X, Z, C, T).

        Note:
            Not all methods of indexing can be used, and some indexing will lead
            to unexpected results. For example, logical indexing cannot be used,
            and step sizes in slice objects is ignored for the first three
            indices. This means and index such as ``[0:100:2,0:100:2,0,0,0]``
            will save a 100x100x1x1x1 numpy array.

        Args:
            keys: numpy-like slicing used to save a section of an image.
            value: Image chunk to save.

        Example:
            .. code-block:: python

                import bfio

                # Initialize the biowriter
                bw = bfio.BioWriter('Path/To/File.ome.tif',
                                    X=100,
                                    Y=100,
                                    dtype=numpy.uint8)

                # Load and copy a 100x100 array of pixels
                bw[:100,:100,0,0,0] = np.zeros((100,100),dtype=numpy.uint8)

                # Slice steps sizes are ignored for the first 3 indices, so this
                # does the same as above
                bw[0:100:2,0:100:2] = np.zeros((100,100),dtype=numpy.uint8)

                # The last two dimensions can receive a tuple or list as input
                # Save two channels
                bw[:100,100,0,:2,0] = np.ones((100,100,1,2),dtype=numpy.uint8)

                # If the file is 3d, save the first 10 z-slices
                br[...,:10,0,0] = np.ones((100,100,1,2),dtype=numpy.uint8)
        """
        ind = self._parse_slice(keys)

        while len(value.shape) < 5:
            value = value[..., numpy.newaxis]

        for i, d in enumerate("YXZCT"):
            if ind[d] is None:
                if value.shape[i] != getattr(self, d):
                    raise IndexError(
                        "Shape of image {} does not match the ".format(value.shape)
                        + "save dimensions {}.".format(ind)
                    )
            elif d in "YXZ" and ind[d][1] - ind[d][0] != value.shape[i]:
                raise IndexError(
                    "Shape of image {} does not match the ".format(value.shape)
                    + "save dimensions {}.".format(ind)
                )
            elif d in "CT" and len(ind[d]) != value.shape[i]:
                raise IndexError(
                    "Shape of image {} does not match the ".format(value.shape)
                    + "save dimensions {}.".format(ind)
                )
            elif d in "YXZ":
                ind[d] = ind[d][0]

        self.write(value, **ind)

    def _minimal_xml(self) -> ome_types.model.OME:
        """Generates minimal xml for ome tif initialization.

        Returns:
            ome_types.model.OME
        """
        assert (
            not self._read_only
        ), "The image has started to be written. To modify the xml again, reinitialize."
        omexml = ome_types.model.OME()
        omexml.images.append(
            ome_types.model.Image(
                id="Image:0",
                pixels=ome_types.model.Pixels(
                    id="Pixels:0",
                    dimension_order="XYZCT",
                    big_endian=False,
                    size_c=1,
                    size_z=1,
                    size_t=1,
                    size_x=1,
                    size_y=1,
                    channels=[
                        ome_types.model.Channel(
                            id="Channel:0",
                            samples_per_pixel=1,
                        )
                    ],
                    type=ome_types.model.simple_types.PixelType.UINT8,
                    tiff_data_blocks=[ome_types.model.TiffData()],
                ),
            )
        )

        return omexml

    def write(
        self,
        image: numpy.ndarray,
        X: typing.Union[int, None] = None,
        Y: typing.Union[int, None] = None,
        Z: typing.Union[int, None] = None,
        C: typing.Union[tuple, list, int, None] = None,
        T: typing.Union[tuple, list, int, None] = None,
    ) -> None:
        """write_image Write the image.

        Write all or part of the image. A 5-dimmensional numpy.ndarray is
        required as the image input.

        Args:
            image: a 5-d numpy array
            X: The starting index of where to save data along the x-axis
                (columns). If None, loads the full range. *Defaults to None.*
            Y: The starting index of where to save data along the y-axis (rows).
                If None, loads the full range. *Defaults to None.*
            Z: The starting index of where to save data along the z-axis
                (depth). If None, loads the full range. *Defaults to None.*
            C: Values indicating channel indices to load. If None, loads the
                full range. *Defaults to None.*
            T: Values indicating timepoints to load. If None, loads the full
                range. *Defaults to None.*
        """
        # Set pixel bounds
        if X is None:
            X = 0
        if Y is None:
            Y = 0
        if Z is None:
            Z = 0

        if isinstance(X, int):
            X = [X]
        if isinstance(Y, int):
            Y = [Y]
        if isinstance(Z, int):
            Z = [Z]

        X.append(image.shape[1] + X[0])
        Y.append(image.shape[0] + Y[0])
        Z.append(image.shape[2] + Z[0])

        # Validate inputs
        X = self._val_xyz(X, "X")
        Y = self._val_xyz(Y, "Y")
        Z = self._val_xyz(Z, "Z")
        C = self._val_ct(C, "C")
        T = self._val_ct(T, "T")

        assert len(image.shape) == 5, "Image must be 5-dimensional (x,y,z,c,t)."

        saving_shape = (Y[1] - Y[0], X[1] - X[0], Z[1] - Z[0], len(C), len(T))
        for d, v in zip(image.shape, saving_shape):
            if d != v:
                raise ValueError(
                    "Image shape {} does not match saving shape {}.".format(
                        image.shape, saving_shape
                    )
                )

        # Define tile bounds
        X_tile_start = (X[0] // self._TILE_SIZE) * self._TILE_SIZE
        Y_tile_start = (Y[0] // self._TILE_SIZE) * self._TILE_SIZE
        X_tile_end = numpy.ceil(X[1] / self._TILE_SIZE).astype(int) * self._TILE_SIZE
        Y_tile_end = numpy.ceil(Y[1] / self._TILE_SIZE).astype(int) * self._TILE_SIZE

        # Read the image
        self._backend.write_image(
            [X_tile_start, X_tile_end], [Y_tile_start, Y_tile_end], Z, C, T, image
        )

    def close(self) -> None:
        """Close the image.

        This function should be called when an image will no longer be written
        to. This allows for proper closing and organization of metadata.
        """
        if self._backend is not None:
            self._backend.close()

    def _put(self):
        """_put Method for saving image supertiles.

        This method is intended to be run within a thread, and writes a
        chunk of the image according to the coordinates in a queue.

        Currently, this function will only write the first Z, C, and T
        positions regardless of what Z, C, and T coordinate are provided
        to the function. This function will need to be changed in then
        future to account for this.

        If the last value in X or Y is larger than the size of the
        image, then the image is cropped to the appropriate size.

        Input coordinates are read from the _supertile_index Queue object.

        Input data is stored in the _raw_buffer Queue object.

        A boolean value is returned to indicate the processed has finished.
        """
        image = self._raw_buffer.get()
        X, Y, Z, C, T = self._supertile_index.get()

        # Attach the jvm to the thread
        self._backend.attach()

        # Write the image
        self.write(
            image[: self.num_y(), :, numpy.newaxis, numpy.newaxis, numpy.newaxis],
            X=[X[0]],
            Y=[Y[0]],
        )

        return True

    def _buffer_supertile(self, column_start: int, column_end: int):
        """_buffer_supertile Process the pixel buffer.

        Give the column indices of the data to process, and determine if
        the buffer needs to be processed. This method checks to see if
        data in the buffer can be shifted into the _raw_buffer for writing.

        Args:
            column_start: First column index of data to be loaded
            column_end: Last column index of data to be loaded

        """
        # If the start column index is outside of the width of the supertile,
        # write the data and shift the pixels
        if column_start - self._tile_x_offset >= 1024:
            self._raw_buffer.put(numpy.copy(self._pixel_buffer[:, 0:1024]))
            self._pixel_buffer[:, 0:1024] = self._pixel_buffer[:, 1024:2048]
            self._pixel_buffer[:, 1024:] = 0
            self._tile_x_offset += 1024
            self._tile_last_column = numpy.argwhere(
                (self._pixel_buffer == 0).all(axis=0)
            )[0, 0]

    def _assemble_tiles(self, images, X, Y, Z, C, T):
        """_assemble_tiles Handle data untiling.

        This function puts tiles into the _pixel_buffer, effectively
        untiling them.

        Args:
            X (list): List of 2-tuples indicating the (min,max)
                range of pixels to load within a tile.
            Y (list): List of 2-tuples indicating the (min,max)
                range of pixels to load within a tile.
            Z (None): Placeholder, to be implemented.
            C (None): Placeholder, to be implemented.
            T (None): Placeholder, to be implemented.

        Returns:
            numpy.ndarray: 2-dimensional ndarray.
        """
        self._buffer_supertile(X[0][0], X[0][1])

        if X[-1][0] - self._tile_x_offset > 1024:
            split_ind = 0
            while X[split_ind][0] - self._tile_x_offset < 1024:
                split_ind += 1
        else:
            split_ind = len(X)

        # Untile the data
        num_tiles = len(X)

        for ind in range(split_ind):
            r_min = Y[ind][0] - self._tile_y_offset
            r_max = Y[ind][1] - self._tile_y_offset
            c_min = X[ind][0] - self._tile_x_offset
            c_max = X[ind][1] - self._tile_x_offset
            self._pixel_buffer[r_min:r_max, c_min:c_max] = images[ind, :, :, 0]

        if split_ind != num_tiles:
            self._buffer_supertile(X[-1][0], X[-1][1])
            for ind in range(split_ind, num_tiles):
                r_min = Y[ind][0] - self._tile_y_offset
                r_max = Y[ind][1] - self._tile_y_offset
                c_min = X[ind][0] - self._tile_x_offset
                c_max = X[ind][1] - self._tile_x_offset
                self._pixel_buffer[r_min:r_max, c_min:c_max] = images[ind, :, :, 0]

        self._tile_last_column = c_max

        return True

    def _writerate(
        self,
        tile_size: typing.Union[typing.List, typing.Tuple],
        tile_stride: typing.Union[typing.List, typing.Tuple, None] = None,
        batch_size=None,
        channels=[0],
    ):
        """Writerate Image saving iterator.

        This method is an iterator to save tiles of an image. This method
        buffers the saving of pixels asynchronously to quickly save
        images to disk. It is designed to work in complement to the
        BioReader.iterate method, and expects images to be fed into it in
        the exact same order as they would come out of that method.

        Data is sent to this iterator using the send() method once the
        iterator has been created. See the example for more information.

        Args:
            tile_size: A list/tuple of length 2, indicating the height and width
                of the tiles to return.
            tile_stride: A list/tuple of length 2, indicating the row and column
                stride size. If None, then tile_stride = tile_size. Defaults to None.
            batch_size: Number of tiles to return on each iteration. Defaults to 32.
            channels: A placeholder. Only the first channel is ever loaded.
                Defaults to [0].

        Yields:
            Nothing

        Example:
            from bfio import BioReader, BioWriter
            import numpy as np

            # Create the BioReader
            br = bfio.BioReader('/path/to/file')

            # Create the BioWriter
            out_path = '/path/to/output'
            bw = bfio.BioWriter(out_path,metadata=br.read_metadata())

            # Get the batch size
            batch_size = br.maximum_batch_size(
                tile_size=[256,256],tile_stride=[256,256]
            )
            readerator = br.iterate(
                tile_size=[256,256],tile_stride=[256,256],batch_size=batch_size
            )
            writerator = bw.writerate(
                tile_size=[256,256],tile_stride=[256,256],batch_size=batch_size
            )

            # Initialize the writerator
            next(writerator)

            # Load tiles of the imgae and save them
            for images,indices in readerator:
                writerator.send(images)
            bw.close_image()

            # Verify images are the same
            original_image = br.read_image()
            bw = bfio.BioReader(out_path)
            saved_image = bw.read_image()

            print(
                'Original and saved images are the same: {}'.format(
                    numpy.array_equal(original_image,saved_image)
                )
            )

        """
        # Enure that the number of tiles does not exceed the width of a supertile
        if batch_size is None:
            batch_size = min([32, self.maximum_batch_size(tile_size, tile_stride)])
        else:
            assert batch_size <= self.maximum_batch_size(
                tile_size, tile_stride
            ), "batch_size must be less than or equal to {}.".format(
                self.maximum_batch_size(tile_size, tile_stride)
            )

        # input error checking
        assert len(tile_size) == 2, "tile_size must be a list with 2 elements"
        if tile_stride is not None:
            assert len(tile_stride) == 2, "stride must be a list with 2 elements"
        else:
            tile_stride = tile_size

        # calculate unpadding
        if not (set(tile_size) & set(tile_stride)):
            xyoffset = [
                int((tile_size[0] - tile_stride[0]) / 2),
                int((tile_size[1] - tile_stride[1]) / 2),
            ]
            xypad = [
                (tile_size[0] - tile_stride[0]) / 2,
                (tile_size[1] - tile_stride[1]) / 2,
            ]
            xypad[0] = (
                xyoffset[0]
                + (tile_stride[0] - numpy.mod(self.num_y(), tile_stride[0])) / 2
            )
            xypad[1] = (
                xyoffset[1]
                + (tile_stride[1] - numpy.mod(self.num_x(), tile_stride[1])) / 2
            )
            xypad = (
                (int(xyoffset[0]), int(2 * xypad[0] - xyoffset[0])),
                (int(xyoffset[1]), int(2 * xypad[1] - xyoffset[1])),
            )
        else:
            xyoffset = [0, 0]
            xypad = (
                (0, max([tile_size[0] - tile_stride[0], 0])),
                (0, max([tile_size[1] - tile_stride[1], 0])),
            )

        # determine supertile sizes
        y_tile_dim = int(numpy.ceil((self.num_y() - 1) / 1024))
        x_tile_dim = 1

        # Initialize the pixel buffer
        self._pixel_buffer = numpy.zeros(
            (y_tile_dim * 1024 + tile_size[0], 2 * x_tile_dim * 1024 + tile_size[1]),
            dtype=self.pixel_type(),
        )
        self._tile_x_offset = 0
        self._tile_y_offset = 0

        # Generate the supertile saving order
        tiles = []
        y_tile_list = list(range(0, self.num_y(), 1024 * y_tile_dim))
        if y_tile_list[-1] != 1024 * y_tile_dim:
            y_tile_list.append(1024 * y_tile_dim)
        x_tile_list = list(range(0, self.num_x(), 1024 * x_tile_dim))
        if x_tile_list[-1] < self.num_x() + xypad[1][1]:
            x_tile_list.append(x_tile_list[-1] + 1024)

        for yi in range(len(y_tile_list) - 1):
            for xi in range(len(x_tile_list) - 1):
                y_range = [y_tile_list[yi], y_tile_list[yi + 1]]
                x_range = [x_tile_list[xi], x_tile_list[xi + 1]]
                tiles.append([x_range, y_range])
                self._supertile_index.put((x_range, y_range, [0, 1], [0], [0]))

        # Start the thread pool and start loading the first supertile
        thread_pool = ThreadPoolExecutor(self._max_workers)

        # generate the indices for each tile
        # TODO: modify this to grab more than just the first z-index
        X = []
        Y = []
        Z = []
        C = []
        T = []
        x_list = numpy.array(numpy.arange(0, self.num_x(), tile_stride[1]))
        y_list = numpy.array(numpy.arange(0, self.num_y(), tile_stride[0]))
        for x in x_list:
            for y in y_list:
                X.append([x, x + tile_stride[1]])
                Y.append([y, y + tile_stride[0]])
                Z.append([0, 1])
                C.append(channels)
                T.append([0])

        # start looping through batches
        bn = 0
        while bn < len(X):
            # Wait for tiles to be sent
            images = yield

            # Wait for the last untiling thread to finish
            if self._tile_thread is not None:
                self._tile_thread.result()

            # start a thread to untile the data
            b = bn + images.shape[0]
            self._tile_thread = thread_pool.submit(
                self._assemble_tiles,
                images,
                X[bn:b],
                Y[bn:b],
                Z[bn:b],
                C[bn:b],
                T[bn:b],
            )
            bn = b

            # Save a supertile if a thread is available
            if self._raw_buffer.qsize() > 0:
                if self._put_thread is not None:
                    self._put_thread.result()
                self._put_thread = thread_pool.submit(self._put)

        # Wait for the final untiling thread to finish
        self._tile_thread.result()

        # Put the remaining pixels in the buffer into the _raw_buffer
        self._raw_buffer.put(
            self._pixel_buffer[:, 0 : self.num_x() - self._tile_x_offset]
        )

        # Save the last supertile
        if self._put_thread is not None:
            self._put_thread.result()  # wait for the previous thread to finish
        self._put()  # no need to use a thread for final save

        thread_pool.shutdown()

        yield


try:
    from napari_plugin_engine import napari_hook_implementation

    class NapariReader:
        """Special class to read data into Napari."""

        def __init__(self, file: str):

            # Let BioReader try to guess the backend based on file extension
            try:
                self.br = BioReader(file)

            # If the backend is wrong, fall back to BioFormats
            except ValueError:
                self.br = BioReader(file, backend="java")

            # Raise an error if the data type is unrecognized by BioFormats/bfio
            numpy.iinfo(self.br.dtype).min

        def __call__(self, file):

            metadata = {
                "contrast_limits": (
                    numpy.iinfo(self.br.dtype).min,
                    numpy.iinfo(self.br.dtype).max,
                )
            }

            return [
                (self, metadata, "image"),
            ]

        def __getitem__(self, keys):

            return self.br[tuple(reversed(keys))]

        @property
        def dtype(self):
            return self.br.dtype

        @property
        def shape(self):
            return tuple(reversed(self.br.shape))

        @property
        def ndim(self):
            return 5

    @napari_hook_implementation(specname="napari_get_reader")
    def get_reader(path: str):

        try:
            reader = NapariReader(path)
            BioReader.logger.info("Reading with the BioReader.")
        except Exception as e:
            BioReader.logger.info(e)
            reader = None

        return reader

    @napari_hook_implementation(specname="napari_write_image")
    def get_writer(path: str, data: numpy.ndarray, meta: dict):

        if isinstance(data, NapariReader):
            bw = BioWriter(path, metadata=data.br.metadata)

            bw[:] = data.br[:]
        else:
            if meta["rgb"]:
                BioWriter.logger.info("The BioWriter cannot write color images.")
                return None

            bw = BioWriter(path)
            bw.shape = data.shape
            bw.dtype = data.dtype

            data = numpy.transpose(data, tuple(reversed(range(data.ndim))))

            while data.ndim < 5:
                data = data[..., numpy.newaxis]

            bw[:] = data

        return path

except ModuleNotFoundError:
    pass
