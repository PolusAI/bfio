# -*- coding: utf-8 -*-
# import core packages
import io
import logging
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import threading
from jpype import JBoolean

# Third party packages
import imagecodecs
import numpy
import ome_types
from tifffile import tifffile
from xml.etree import ElementTree as ET


# bfio internals
from bfio import __version__ as version
import bfio.base_classes
from bfio.utils import start, clean_ome_xml_for_known_issues, pixels_per_cm

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.backends")


class PythonReader(bfio.base_classes.AbstractReader):
    logger = logging.getLogger("bfio.backends.PythonReader")

    _rdr: tifffile.TiffFile = None
    _offsets_bytes = None
    _STATE_DICT = ["_metadata", "frontend"]

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug("__init__(): Initializing _rdr (tifffile.TiffFile)...")
        self._rdr = tifffile.TiffFile(self.frontend._file_path)
        if self._rdr.ome_metadata is None:
            raise TypeError(
                "No OME metadata detected, use the java backend to read this file."
            )
        else:
            self.read_metadata()
        width = self._metadata.images[0].pixels.size_x
        height = self._metadata.images[0].pixels.size_y

        self._rdr_pages = self._rdr.pages
        # if level is given try accessing the sub-res image
        if self.frontend.level is not None:
            if len(self._rdr.series) != 0:
                series = self._rdr.series[0]
                if len(series.levels) <= self.frontend.level:
                    self.close()
                    raise ValueError(
                        "{} does not have a resolution level {}.".format(
                            self.frontend._file_path.name, self.frontend.level
                        )
                    )
                else:
                    self.logger.debug(
                        f"Reading sub-resolution level {self.frontend.level}"
                    )
                    self._rdr_pages = series.levels[self.frontend.level]
                    height = self._rdr_pages[0].shape[0]
                    width = self._rdr_pages[0].shape[1]
                    self._metadata.images[0].pixels.size_x = width
                    self._metadata.images[0].pixels.size_y = height

        for tag in self._rdr_pages[0].tags:
            logger.debug(tag)

        if not self._rdr_pages[0].is_tiled or self._rdr_pages[0].rowsperstrip != 0:
            if (
                self._rdr_pages[0].tilewidth < self.frontend._TILE_SIZE
                or self._rdr_pages[0].tilelength < self.frontend._TILE_SIZE
            ):
                self.close()
                raise TypeError(
                    frontend._file_path.name
                    + " is not a tiled tiff."
                    + " The python backend of the BioReader only "
                    + "supports OME tiled tiffs. Use the java backend "
                    + "to load this image."
                )

        elif (
            self._rdr_pages[0].tilewidth != self.frontend._TILE_SIZE
            or self._rdr_pages[0].tilelength != self.frontend._TILE_SIZE
        ):
            if width > frontend._TILE_SIZE or height > frontend._TILE_SIZE:
                self.close()
                raise ValueError(
                    "Tile width and height should be {} when ".format(
                        self.frontend._TILE_SIZE
                    )
                    + "using the python backend, but found "
                    + "tilewidth={} and tilelength={}. Use the java ".format(
                        self._rdr_pages[0].tilewidth, self._rdr_pages[0].tilelength
                    )
                    + "backend to read this image."
                )
        elif self._metadata.images[0].pixels.dimension_order.value != "XYZCT":
            raise TypeError(
                "The dimension order of the data is not XYZCT. "
                + "Use the java backend to read this image."
            )
        elif self._metadata.images[0].pixels.interleaved:
            raise TypeError(
                "The data is RGB interleaved and cannot be read by the PythonReader. "
                + "Use the java backend to read this image."
            )

        # Close the reader until we need it
        self._rdr.filehandle.close()

    def __getstate__(self) -> Dict:
        state_dict = {n: getattr(self, n) for n in self._STATE_DICT}
        state_dict.update({"file_path": self.frontend._file_path})
        state_dict.update({"level": self.frontend.level})

        return state_dict

    def __setstate__(self, state) -> None:
        for k, v in state.items():
            if k == "file_path" or k == "level":
                pass
            else:
                setattr(self, k, v)

        self._lock = threading.Lock()
        self._rdr = tifffile.TiffFile(state["file_path"])
        self._rdr_pages = self._rdr.pages
        if state["level"] is not None:
            if len(self._rdr.series) != 0:
                series = self._rdr.series[0]
                self._rdr_pages = series.levels[state["level"]]
        self._rdr.filehandle.close()

    def read_metadata(self):
        self.logger.debug("read_metadata(): Reading metadata...")

        if self._metadata is None:
            try:
                self._metadata = ome_types.from_xml(
                    self._rdr.ome_metadata, validate=False
                )
            except (ET.ParseError, ValueError):
                if self.frontend.clean_metadata:
                    cleaned = clean_ome_xml_for_known_issues(self._rdr.ome_metadata)
                    self._metadata = ome_types.from_xml(cleaned, validate=False)
                    self.logger.warning(
                        "read_metadata(): OME XML required reformatting."
                    )
                else:
                    raise

        return self._metadata

    class _TiffBytesOffsets:
        def __init__(self, parent, index):
            self._tiff_frame = tifffile.TiffFrame(parent, index)

        def get_offset(self):
            return self._tiff_frame._nextifd()

        def get_index(self):
            return self._tiff_frame.index

        def get_dataoffsets(self):
            return self._tiff_frame.dataoffsets

        def get_databytecounts(self):
            return self._tiff_frame.databytecounts

    def _page_offsets_bytes(self, index: int):
        if index == 0:
            return self._rdr_pages[0].dataoffsets, self._rdr_pages[0].databytecounts
        parent = self._rdr

        if self._offsets_bytes is None or self._offsets_bytes.get_index() + 1 != index:
            self._rdr.pages._seek(int(index))
        else:
            self._rdr.filehandle.seek(self._offsets_bytes.get_offset())

        obc = self._TiffBytesOffsets(parent, index)
        self._offsets_bytes = obc
        return obc.get_dataoffsets(), obc.get_databytecounts()

    def _chunk_indices(self, X, Y, Z, C=[0], T=[0]):
        self.logger.debug(f"_chunk_indices(): (X,Y,Z,C,T) -> ({X},{Y},{Z},{C},{T})")
        assert all(len(D) == 2 for D in [X, Y, Z])
        assert all(isinstance(D, list) for D in [C, T])

        offsets = []
        bytecounts = []

        ts = self.frontend._TILE_SIZE

        x_tiles = numpy.arange(X[0] // ts, numpy.ceil(X[1] / ts), dtype=int)
        y_tile_stride = numpy.ceil(self.frontend.x / ts).astype(int)

        for t in T:
            t_index = self.frontend.Z * self.frontend.C * t
            for c in C:
                c_index = t_index + self.frontend.Z * c
                for z in range(Z[0], Z[1]):
                    index = z + c_index

                    dataoffsets, databytecounts = self._page_offsets_bytes(index)
                    for y in range(Y[0] // ts, int(numpy.ceil(Y[1] / ts))):
                        y_offset = int(y * y_tile_stride)
                        ind = (x_tiles + y_offset).tolist()

                        o = [dataoffsets[i] for i in ind]
                        b = [databytecounts[i] for i in ind]

                        offsets.extend(o)
                        bytecounts.extend(b)

        return offsets, bytecounts

    def _process_chunk(self, args):
        keyframe = self._keyframe
        out = self._image

        w, l, d, c, t = self._tile_indices[args[1]]

        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)

        if segment is None:
            segment = keyframe.nodata

        self.logger.debug("_process_chunk(): shape = {}".format(shape))
        self.logger.debug(
            "_process_chunk(): (w,l,d) = {},{},{}".format(w[0], l[0], d[0])
        )

        if self.load_tiles:
            width = out.shape[6]
            height = out.shape[5]
            out[
                d[0],
                c[0],
                t[0],
                l[0] // self._TILE_SIZE[0],
                w[0] // self._TILE_SIZE[1],
                :height,
                :width,
            ] = segment[0, :height, :width, 0]
        else:
            width = min(out.shape[4] - w[0], self._TILE_SIZE[1])
            height = min(out.shape[3] - l[0], self._TILE_SIZE[0])
            out[d[0], c[0], t[0], l[0] : l[0] + height, w[0] : w[0] + width] = segment[
                0, :height, :width, 0
            ]

    def _read_image(self, X, Y, Z, C, T, output):
        # Get keyframe
        self._keyframe = self._rdr_pages[0].keyframe
        # Open the file
        fh = self._rdr_pages[0].parent.filehandle
        # fh = self._rdr.series[0].levels[0].parent.filehandle
        fh.open()

        # Set tile size if request size is < _TILE_SIZE for efficiency
        self._TILE_SIZE = (
            min(self._image.shape[-2], self.frontend._TILE_SIZE),
            min(self._image.shape[-1], self.frontend._TILE_SIZE),
        )

        # Get binary data info
        offsets, bytecounts = self._chunk_indices(X, Y, Z, C, T)

        self.logger.debug("read_image(): _tile_indices = {}".format(self._tile_indices))

        if self.frontend._max_workers > 1:
            with ThreadPoolExecutor(self.frontend._max_workers) as executor:
                # cast to list so that any read errors are raised
                list(
                    executor.map(
                        self._process_chunk, fh.read_segments(offsets, bytecounts)
                    )
                )
        else:
            for args in fh.read_segments(offsets, bytecounts):
                self._process_chunk(args)

        # Close the file
        fh.close()

    def close(self):
        if self._rdr is not None:
            self._rdr.close()

    def __del__(self):
        self.close()


class TiffIFDHeader:
    databyteoffsets: List[int]
    databytecounts: List[int]
    dataoffsetsoffset: Tuple
    databytecountsoffset: Tuple
    ifd: io.BytesIO
    ifdoffset: int
    ifdsize: int = 0
    descriptionoffset: int
    descriptionlenoffset: int
    _ifdstart: int = 8
    _ifdpos = 0
    _value = None

    def __init__(
        self, tile_count: int, bytecountformat: str, offsetformat: str, byteorder: str
    ):
        self.databyteoffsets = [0 for _ in range(tile_count)]
        self.databytecounts = [0 for _ in range(tile_count)]
        self.bytecountformat = bytecountformat
        self.ifd = io.BytesIO()
        self.offsetformat = offsetformat
        self.byteorder = byteorder

    def _pack(self, fmt, *val):
        if fmt[0] not in "<>":
            fmt = self.byteorder + fmt
        return struct.pack(fmt, *val)

    def getvalue(self):
        if self._value is None:
            offsetformat = self.offsetformat
            bytecountformat = self.bytecountformat

            # update strip/tile offsets
            offset, pos = self.dataoffsetsoffset
            self.ifd.seek(offset)
            if pos is not None:
                self.ifd.write(self._pack(offsetformat, self._ifdstart + pos))
                self.ifd.seek(pos)
                for size in self.databyteoffsets:
                    self.ifd.write(self._pack(offsetformat, size))
            else:
                self.ifd.write(self._pack(offsetformat, self.databyteoffsets[0]))

            # update strip/tile bytecounts
            offset, pos = self.databytecountsoffset
            self.ifd.seek(offset)
            if pos:
                self.ifd.write(self._pack(offsetformat, self._ifdstart + pos))
                self.ifd.seek(pos)
            self.ifd.write(self._pack(bytecountformat, *self.databytecounts))

            self._value = self.ifd.getvalue()

        return self._value

    def set_next_ifd(self, pos):
        self.ifd.seek(self.ifdoffset)
        self.ifd.write(self._pack(self.offsetformat, pos))
        self.ifd.seek(self.ifdsize)


class TiffIFDHeaders(object):
    tags: List[Tuple] = []
    headers: List[TiffIFDHeader] = []
    _ifdpos: int = 8

    def __init__(self, writer: "PythonWriter"):
        self.tiff = writer._writer.tiff
        self.tags = writer._tags
        self.page_count = writer.frontend.Z * writer.frontend.C * writer.frontend.T
        self.headers = [
            TiffIFDHeader(
                writer._numtiles,
                writer._bytecountformat,
                self.tiff.offsetformat,
                self.tiff.byteorder,
            )
            for _ in range(self.page_count)
        ]

        # Generate the first header
        self._generate_page(0)

        # remove tags that should be written only once
        self._tags = self.tags  # backup
        writer._tags = [tag for tag in writer._tags if not tag[-1]]

        # For multipage tiffs, change the description
        for ind, tag in enumerate(writer._tags):
            if tag[0] == 270:
                del writer._tags[ind]
                break
        description = (
            "ImageJ=\nhyperstack=true\nimages=1\n"
            + f"channels={writer.frontend.C}\n"
            + f"slices={writer.frontend.Z}\n"
            + f"frames={writer.frontend.T}"
        )
        writer._addtag(270, "s", 0, description)  # Description
        self.tags = sorted(writer._tags, key=lambda x: x[0])

        for i in range(1, self.page_count):
            self._generate_page(i)

        # Set next ifd to the start of the file, indicating the last page
        self.headers[-1].set_next_ifd(0)

    def _pack(self, fmt, *val):
        if fmt[0] not in "<>":
            fmt = self.tiff.byteorder + fmt
        return struct.pack(fmt, *val)

    def __getitem__(self, index):
        return self.headers[index]

    def _generate_page(self, index):
        tagnoformat = self.tiff.tagnoformat
        offsetformat = self.tiff.offsetformat
        offsetsize = self.tiff.offsetsize
        tagsize = self.tiff.tagsize
        tagbytecounts = 325
        tagoffsets = 324
        tags = self.tags

        # create IFD in memory, do not write to disk
        header = self.headers[index]
        header._ifdstart = self._ifdpos
        header.ifd.write(self._pack(tagnoformat, len(tags)))
        tagoffset = header.ifd.tell()
        header.ifd.write(b"".join(t[1] for t in tags))
        header.ifdoffset = header.ifd.tell()
        header.ifd.write(self._pack(offsetformat, 0))  # offset to next IFD

        # write tag values and patch offsets in ifdentries
        for tagindex, tag in enumerate(tags):
            offset = tagoffset + tagindex * tagsize + offsetsize + 4
            code = tag[0]
            value = tag[2]

            if value:
                pos = header.ifd.tell()

                if pos % 2:
                    # tag value is expected to begin on word boundary
                    header.ifd.write(b"\0")
                    pos += 1
                header.ifd.seek(offset)
                header.ifd.write(self._pack(offsetformat, self._ifdpos + pos))
                header.ifd.seek(pos)
                header.ifd.write(value)

                if code == tagoffsets:
                    header.dataoffsetsoffset = offset, pos

                elif code == tagbytecounts:
                    header.databytecountsoffset = offset, pos

                elif code == 270 and value.endswith(b"\0\0\0\0"):
                    # image description buffer
                    header.descriptionoffset = self._ifdpos + pos
                    header.descriptionlenoffset = (
                        self._ifdpos + tagoffset + tagindex * tagsize + 4
                    )

            elif code == tagoffsets:
                header.dataoffsetsoffset = offset, None

            elif code == tagbytecounts:
                header.databytecountsoffset = offset, None

        header.ifdsize = header.ifd.tell()
        if header.ifdsize % 2:
            header.ifd.write(b"\0")
            header.ifdsize += 1

        self._ifdpos += header.ifdsize
        header._ifdpos = self._ifdpos

        # Update the next ifd reference
        header.set_next_ifd(self._ifdpos)

    def __len__(self):
        return sum(self[i].ifdsize for i in range(len(self.headers)))


class PythonWriter(bfio.base_classes.AbstractWriter):
    _page_open = False
    _current_page = None

    logger = logging.getLogger("bfio.backends.PythonWriter")

    def __init__(self, frontend):
        super().__init__(frontend)

    def _pack(self, fmt, *val):
        if fmt[0] not in "<>":
            fmt = self._writer.tiff.byteorder + fmt
        return struct.pack(fmt, *val)

    def _addtag(self, code, dtype, count, value, writeonce=False):
        tags = self._tags

        # compute ifdentry & ifdvalue bytes from code, dtype, count, value
        # append (code, ifdentry, ifdvalue, writeonce) to tags list
        if not isinstance(code, int):
            code = tifffile.TIFF.TAGS[code]
        try:
            datatype = dtype
            dataformat = tifffile.TIFF.DATA_FORMATS[datatype][-1]
        except KeyError as exc:
            try:
                dataformat = dtype
                if dataformat[0] in "<>":
                    dataformat = dataformat[1:]
                datatype = tifffile.TIFF.DATA_DTYPES[dataformat]
            except (KeyError, TypeError):
                raise ValueError(f"unknown dtype {dtype}") from exc
        del dtype

        rawcount = count
        if datatype == 2:
            # string
            if isinstance(value, str):
                if code != 270:
                    # enforce 7-bit ASCII on Unicode strings
                    try:
                        value = value.encode("ascii")
                    except UnicodeEncodeError as exc:
                        raise ValueError("TIFF strings must be 7-bit ASCII") from exc
                else:
                    # OME description must be utf-8, which breaks tiff spec
                    value = value.encode()
            elif not isinstance(value, bytes):
                raise ValueError("TIFF strings must be 7-bit ASCII")
            if len(value) == 0 or value[-1] != b"\x00":
                value += b"\x00"
            count = len(value)
            if code == 270:
                self._descriptiontag = tifffile.TiffTag(self, 0, 270, 2, count, None, 0)
                rawcount = value.find(b"\x00\x00")
                if rawcount < 0:
                    rawcount = count
                else:
                    # length of string without buffer
                    rawcount = max(self._writer.tiff.offsetsize + 1, rawcount + 1)
                    rawcount = min(count, rawcount)
            else:
                rawcount = count
            value = (value,)

        elif isinstance(value, bytes):
            # packed binary data
            itemsize = struct.calcsize(dataformat)
            if len(value) % itemsize:
                raise ValueError("invalid packed binary data")
            count = len(value) // itemsize
            rawcount = count

        if datatype in (5, 10):  # rational
            count *= 2
            dataformat = dataformat[-1]

        ifdentry = [
            self._pack("HH", code, datatype),
            self._pack(self._writer.tiff.offsetformat, rawcount),
        ]

        ifdvalue = None
        if struct.calcsize(dataformat) * count <= self._writer.tiff.offsetsize:
            # value(s) can be written directly
            if isinstance(value, bytes):
                ifdentry.append(self._pack(f"{self.tiff.offsetsize}s", value))
            elif count == 1:
                if isinstance(value, (tuple, list, numpy.ndarray)):
                    value = value[0]
                ifdentry.append(
                    self._pack(
                        f"{self._writer.tiff.offsetsize}s",
                        self._pack(dataformat, value),
                    )
                )
            else:
                ifdentry.append(
                    self._pack(
                        f"{self._writer.tiff.offsetsize}s",
                        self._pack(f"{count}{dataformat}", *value),
                    )
                )
        else:
            # use offset to value(s)
            ifdentry.append(self._pack(self._writer.tiff.offsetformat, 0))
            if isinstance(value, bytes):
                ifdvalue = value
            elif isinstance(value, numpy.ndarray):
                if value.size != count:
                    raise RuntimeError("value.size != count")
                if value.dtype.char != dataformat:
                    raise RuntimeError("value.dtype.char != dtype")
                ifdvalue = value.tobytes()
            elif isinstance(value, (tuple, list)):
                ifdvalue = self._pack(f"{count}{dataformat}", *value)
            else:
                ifdvalue = self._pack(dataformat, value)
        tags.append((code, b"".join(ifdentry), ifdvalue, writeonce))

    def _init_writer(self):
        """_init_writer Initializes file writing.

        This method is called exactly once per object. Once it is
        called, all other methods of setting metadata will throw an
        error.

        """

        self._tags = []

        self.frontend._metadata.images[0].id = (
            f"Image:{Path(self.frontend._file_path).name}"
        )

        if (
            self.frontend.X
            * self.frontend.Y
            * self.frontend.Z
            * self.frontend.C
            * self.frontend.T
            * self.frontend.bpp
            > 2**31
        ):
            big_tiff = True
        else:
            big_tiff = False

        if self.frontend._metadata.images[0].pixels.big_endian:
            byte_order = ">"
        else:
            byte_order = "<"

        self._writer = tifffile.TiffWriter(
            self.frontend._file_path,
            bigtiff=big_tiff,
            byteorder=byte_order,
            append=False,
        )

        self._byteorder = self._writer.tiff.byteorder

        self._datashape = (1, 1, 1) + (self.frontend.Y, self.frontend.X) + (1,)

        self._datadtype = numpy.dtype(self.frontend.dtype).newbyteorder(self._byteorder)

        offsetsize = self._writer.tiff.offsetsize

        self._compresstag = tifffile.COMPRESSION.ADOBE_DEFLATE

        # normalize data shape to 5D or 6D, depending on volume:
        #   (pages, planar_samples, height, width, contig_samples)
        self._samplesperpixel = 1
        self._bitspersample = self._datadtype.itemsize * 8

        self._tagbytecounts = 325  # TileByteCounts
        self._tagoffsets = 324  # TileOffsets

        def rational(arg, max_denominator=1000000):
            # return nominator and denominator from float or two integers
            from fractions import Fraction  # delayed import

            try:
                f = Fraction.from_float(arg)
            except TypeError:
                f = Fraction(arg[0], arg[1])
            f = f.limit_denominator(max_denominator)
            return f.numerator, f.denominator

        description = ome_types.to_xml(self.frontend._metadata)

        self._addtag(270, "s", 0, description, writeonce=True)  # Description
        self._addtag(305, "s", 0, f"bfio v{version}")  # Software
        # addtag(306, 's', 0, datetime, writeonce=True)
        self._addtag(259, "H", 1, self._compresstag)  # Compression
        self._addtag(256, "I", 1, self._datashape[-2])  # ImageWidth
        self._addtag(257, "I", 1, self._datashape[-3])  # ImageLength
        self._addtag(322, "I", 1, self.frontend._TILE_SIZE)  # TileWidth
        self._addtag(323, "I", 1, self.frontend._TILE_SIZE)  # TileLength

        sampleformat = {"u": 1, "i": 2, "f": 3, "c": 6}[self._datadtype.kind]
        self._addtag(
            339, "H", self._samplesperpixel, (sampleformat,) * self._samplesperpixel
        )

        self._addtag(277, "H", 1, self._samplesperpixel)
        self._addtag(258, "H", 1, self._bitspersample)

        # PhotometricInterpretation
        self._addtag(262, "H", 1, tifffile.PHOTOMETRIC.MINISBLACK.value)

        if self.frontend.physical_size_x[0] is not None:
            self._addtag(
                282,
                "2I",
                1,
                rational(
                    pixels_per_cm(
                        self.frontend.X,
                        self.frontend.physical_size_x[0],
                        self.frontend.physical_size_x[1],
                    )
                ),
            )  # XResolution in pixels/cm
            self._addtag(
                283,
                "2I",
                1,
                rational(
                    pixels_per_cm(
                        self.frontend.Y,
                        self.frontend.physical_size_y[0],
                        self.frontend.physical_size_y[1],
                    )
                ),
            )  # YResolution in pixels/cm
            self._addtag(296, "H", 1, 3)  # ResolutionUnit = cm
        else:
            self._addtag(282, "2I", 1, (1, 1))  # XResolution
            self._addtag(283, "2I", 1, (1, 1))  # YResolution
            self._addtag(296, "H", 1, 1)  # ResolutionUnit

        def bytecount_format(bytecounts, size=offsetsize):
            # return small bytecount format
            if len(bytecounts) == 1:
                return {4: "I", 8: "Q"}[size]
            bytecount = bytecounts[0] * 10
            if bytecount < 2**16:
                return "H"
            if bytecount < 2**32:
                return "I"
            if size == 4:
                return "I"
            return "Q"

        # one chunk per tile per plane
        self._tiles = (
            (self._datashape[3] + self.frontend._TILE_SIZE - 1)
            // self.frontend._TILE_SIZE,
            (self._datashape[4] + self.frontend._TILE_SIZE - 1)
            // self.frontend._TILE_SIZE,
        )

        self._numtiles = tifffile.product(self._tiles)
        self._databytecounts = [
            self.frontend._TILE_SIZE**2 * self._datadtype.itemsize
        ] * self._numtiles
        self._bytecountformat = bytecount_format(self._databytecounts)
        self._addtag(
            self._tagbytecounts,
            self._bytecountformat,
            self._numtiles,
            self._databytecounts,
        )
        self._addtag(
            self._tagoffsets,
            self._writer.tiff.offsetformat,
            self._numtiles,
            [0] * self._numtiles,
        )
        self._bytecountformat = self._bytecountformat * self._numtiles

        # the entries in an IFD must be sorted in ascending order by tag code
        self._tags = sorted(self._tags, key=lambda x: x[0])

        fh = self._writer.filehandle
        self._ifdpos = fh.tell()

        offsetformat = self._writer.tiff.offsetformat
        offsetsize = self._writer.tiff.offsetsize

        if self._ifdpos % 2:
            # location of IFD must begin on a word boundary_ifdoffset
            fh.write(b"\0")
            self._ifdpos += 1

        # update pointer at ifdoffset
        fh.seek(self._writer._ifdoffset)
        fh.write(self._pack(offsetformat, self._ifdpos))
        fh.seek(self._ifdpos)

        # Create the ifd headers
        TiffIFDHeaders._ifdpos = self._ifdpos
        self.headers = TiffIFDHeaders(self)

        # create a gap between the headers and the beginning of the tile data
        headers_size = len(self.headers)
        fh.seek(headers_size + self._ifdpos)
        skip = (16 - (headers_size % 16)) % 16
        fh.seek(skip, 1)
        self._dataoffset = headers_size + skip

    def iter_tiles(self, data, tile, tiles):
        """Return iterator over tiles in data array of normalized shape."""
        shape = data.shape
        if not 1 < len(tile) < 4:
            raise ValueError("invalid tile shape")
        for page in data:
            for plane in page:
                for ty in range(tiles[0]):
                    for tx in range(tiles[1]):
                        chunk = numpy.empty(tile + (shape[-1],), dtype=data.dtype)
                        c1 = min(tile[0], shape[3] - ty * tile[0])
                        c2 = min(tile[1], shape[4] - tx * tile[1])
                        chunk[c1:, c2:] = 0
                        chunk[:c1, :c2] = plane[
                            0,
                            ty * tile[0] : ty * tile[0] + c1,
                            tx * tile[1] : tx * tile[1] + c2,
                        ]
                        yield chunk

    def _write_tiles(self, data, X, Y, Z, C, T):
        assert len(X) == 2 and len(Y) == 2
        if self.frontend._TILE_SIZE != 2**10:
            logger.warning("TILE_SIZE is not set to 1024, tile may save incorrectly.")
        if X[0] % self.frontend._TILE_SIZE != 0 or Y[0] % self.frontend._TILE_SIZE != 0:
            logger.warning(
                "X or Y positions are not on tile boundary, tile may save incorrectly"
            )

        fh = self._writer.filehandle

        x_tiles = list(
            range(
                X[0] // self.frontend._TILE_SIZE,
                1 + (X[1] - 1) // self.frontend._TILE_SIZE,
            )
        )
        tiles = []
        for y in range(
            Y[0] // self.frontend._TILE_SIZE, 1 + (Y[1] - 1) // self.frontend._TILE_SIZE
        ):
            tiles.extend([y * self._tiles[1] + x for x in x_tiles])

        tile_shape = (
            (Y[1] - Y[0] - 1 + self.frontend._TILE_SIZE) // self.frontend._TILE_SIZE,
            (X[1] - X[0] - 1 + self.frontend._TILE_SIZE) // self.frontend._TILE_SIZE,
        )

        tileiters = []
        for ti, t in zip(range(len(T)), T):
            t_index = t * self.frontend.Z * self.frontend.C
            for ci, c in zip(range(len(C)), C):
                c_index = t_index + c * self.frontend.Z
                for zi, z in zip(range(0, Z[1] - Z[0]), range(Z[0], Z[1])):
                    index = c_index + z
                    tileiters.append(
                        (
                            index,
                            self.iter_tiles(
                                data[:, :, zi, ci, ti].reshape(
                                    1, 1, 1, data.shape[0], data.shape[1], 1
                                ),
                                (self.frontend._TILE_SIZE, self.frontend._TILE_SIZE),
                                tile_shape,
                            ),
                        )
                    )

        def compress(page_index, tile_index, data, level=1):
            return (page_index, tile_index, imagecodecs.deflate_encode(data, level))

        if self.frontend._max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.frontend._max_workers) as executor:
                compressed_tiles = []
                for page_index, tileiter in tileiters:
                    for tileindex, tile in zip(tiles, tileiter):
                        compressed_tiles.append(
                            executor.submit(compress, page_index, tileindex, tile)
                        )

                for thread in as_completed(compressed_tiles):
                    page_index, tileindex, tile = thread.result()
                    self.headers[page_index].databyteoffsets[tileindex] = fh.tell()
                    fh.write(tile)
                    self.headers[page_index].databytecounts[tileindex] = len(tile)

        else:
            for page_index, tileiter in tileiters:
                for tileindex, tile in zip(tiles, tileiter):
                    _, _, t = compress(page_index, tileindex, tile)
                    self.headers[page_index].databyteoffsets[tileindex] = fh.tell()
                    fh.write(t)
                    self.headers[page_index].databytecounts[tileindex] = len(t)

    def close(self):
        """close_image Close the image.

        This function should be called when an image will no longer be written
        to. This allows for proper closing and organization of metadata.
        """
        if self._writer is not None:
            for header in self.headers:
                self._writer.filehandle.seek(header._ifdstart)
                self._writer.filehandle.write(header.getvalue())
            self._writer.filehandle.close()
            self._writer = None

    def _write_image(self, X, Y, Z, C, T, image):
        # Do the work
        self._write_tiles(image, X, Y, Z, C, T)

    def __del__(self):
        self.close()


try:
    import jpype
    import jpype.imports
    from jpype.types import JString

    class JavaReader(bfio.base_classes.AbstractReader):
        logger = logging.getLogger("bfio.backends.JavaReader")
        _chunk_size = 4096
        _rdr = None
        _classes_loaded = False

        def _load_java_classes(self):
            if not jpype.isJVMStarted():
                start()

            global ImageReader
            from loci.formats import ImageReader

            global ServiceFactory
            from loci.common.services import ServiceFactory

            global OMEXMLService
            from loci.formats.services import OMEXMLService

            global IMetadata
            from loci.formats.meta import IMetadata

            JavaReader._classes_loaded = True

        def __init__(self, frontend):
            if not JavaReader._classes_loaded:
                self._load_java_classes()

            super().__init__(frontend)

            factory = ServiceFactory()
            service = factory.getInstance(OMEXMLService)
            # service = None
            self.omexml = service.createOMEXMLMetadata()

            self._rdr = ImageReader()
            self._rdr.setOriginalMetadataPopulated(True)
            self._rdr.setMetadataStore(self.omexml)
            self._rdr.setId(JString(str(self.frontend._file_path.absolute())))
            if self.frontend.level is not None:
                if self._rdr.getSeriesCount() > self.frontend.level:
                    self._rdr.setSeries(self.frontend.level)
                else:
                    self.close()
                    raise ValueError(
                        "{} does not have a resolution level {}.".format(
                            self.frontend._file_path.name, self.frontend.level
                        )
                    )

        def read_metadata(self):
            self.logger.debug("read_metadata(): Reading metadata...")

            if self._metadata is None:
                self._metadata = ome_types.from_xml(
                    clean_ome_xml_for_known_issues(str(self.omexml.dumpXML()))
                )
            if (
                self.frontend.level is not None
                and len(self._metadata.images) > self.frontend.level
            ):
                self._metadata.images = [self._metadata.images[self.frontend.level]]
            return self._metadata

        def _read_image(self, X, Y, Z, C, T, output):
            out = self._image
            interleaved = self.frontend.metadata.images[0].pixels.interleaved

            self._prev_read_cached_loc = None
            self._cached_read_data = None
            for ti, t in enumerate(T):
                for zi, z in enumerate(range(Z[0], Z[1])):
                    for ci, c in enumerate(C):
                        index = self._rdr.getIndex(z, c // self.frontend.spp, t)

                        x_max = min([X[1], self.frontend.X])
                        for x in range(X[0], x_max, self._chunk_size):
                            x_range = min([self._chunk_size, x_max - x])

                            y_max = min([Y[1], self.frontend.Y])
                            for y in range(Y[0], y_max, self._chunk_size):
                                y_range = min([self._chunk_size, y_max - y])
                                current_read_loc = (index, x, y, x_range, y_range)
                                if current_read_loc != self._prev_read_cached_loc:
                                    tmp_read = self._rdr.openBytes(
                                        index, x, y, x_range, y_range
                                    )
                                    self._cached_read_data = numpy.frombuffer(
                                        bytes(tmp_read),
                                        self.frontend.dtype,
                                    )
                                    self._prev_read_cached_loc = current_read_loc

                                image = self._cached_read_data
                                # TODO: This should be changed in the future
                                # This reloads all channels for a tile on each
                                # loop. Ideally, there would be some better
                                # logic here to only load the necessary channel
                                # information once.

                                # For now, we are adding some basic caching
                                if self._rdr.getFormat() not in ["Zeiss CZI"]:
                                    if interleaved:
                                        image = image[c :: self.frontend.spp]
                                        image = image.reshape(y_range, x_range)
                                    else:
                                        image = image.reshape(
                                            self.frontend.spp, y_range, x_range
                                        )[c % self.frontend.spp, ...]
                                else:
                                    image = image.reshape(
                                        y_range, x_range, self.frontend.spp
                                    )[..., c % self.frontend.spp]

                                out[
                                    y - Y[0] : y + y_range - Y[0],
                                    x - X[0] : x + x_range - X[0],
                                    zi,
                                    ci,
                                    ti,
                                ] = image

        def close(self):
            if jpype.isJVMStarted() and self._rdr is not None:
                self._rdr.close(JBoolean(False))

        def __del__(self):
            self.close()

    class JavaWriter(bfio.base_classes.AbstractWriter):
        logger = logging.getLogger("bfio.backends.JavaWriter")

        # For Bio-Formats, the first tile has to be written before any other tile
        first_tile = False

        _classes_loaded = False

        def _load_java_classes(self):
            if not jpype.isJVMStarted():
                start()

            global OMETiffWriter
            from loci.formats.out import OMETiffWriter

            global ServiceFactory
            from loci.common.services import ServiceFactory

            global OMEXMLService
            from loci.formats.services import OMEXMLService

            global IMetadata
            from loci.formats.meta import IMetadata

            global CompressionType
            from loci.formats.codec import CompressionType

            JavaWriter._classes_loaded = True

        def __init__(self, frontend):
            if not JavaWriter._classes_loaded:
                self._load_java_classes()

            super().__init__(frontend)

            # Force data to use XYZCT ordering
            self.frontend.metadata.images[0].pixels.dimension_order = "XYZCT"

            # Expand interleaved RGB to separate image planes for each channel
            image = self.frontend.metadata.images[0]

            pseudo_interleaved = any(
                channel.samples_per_pixel > 1 for channel in image.pixels.channels
            )

            if image.pixels.interleaved or pseudo_interleaved:
                image.pixels.interleaved = False

                for _ in range(len(image.pixels.channels)):
                    channel = image.pixels.channels.pop(0)
                    expand_channels = channel.samples_per_pixel
                    channel.samples_per_pixel = 1

                    for c in range(expand_channels):
                        new_channel = ome_types.model.Channel(**channel.model_dump())
                        new_channel.id = channel.id + f":{c}"
                        image.pixels.channels.append(new_channel)

        def _init_writer(self):
            """_init_writer Initializes file writing.

            This method is called exactly once per object. Once it is
            called, all other methods of setting metadata will throw an
            error.

            """
            if self.frontend._file_path.exists():
                self.frontend._file_path.unlink()

            writer = OMETiffWriter()

            # Set big tiff flag if file will be larger than 2GB
            if self.frontend.X * self.frontend.Y * self.frontend.bpp > 2**31:
                writer.setBigTiff(True)
            else:
                writer.setBigTiff(False)

            # Set the metadata
            service = ServiceFactory().getInstance(OMEXMLService)
            xml = ome_types.to_xml(self.frontend.metadata)
            metadata = service.createOMEXMLMetadata(xml)
            writer.setMetadataRetrieve(metadata)

            # Set the file path
            writer.setId(str(self.frontend._file_path))

            # Set compression to
            writer.setCompression(CompressionType.ZLIB.getCompression())

            # Set image tiles
            writer.setTileSizeX(1024)
            writer.setTileSizeY(1024)

            self._writer = writer

        def _process_chunk(self, dims):
            out = self._image

            X, Y, Z, C, T = dims

            index = (
                Z[0] + self.frontend.z * C[0] + self.frontend.z * self.frontend.c * T[0]
            )

            x_range = min([self.frontend.X, X[1] + 1024]) - X[1]
            y_range = min([self.frontend.Y, Y[1] + 1024]) - Y[1]

            self.logger.debug("_process_chunk(): dims = {}".format(dims))
            pixel_buffer = out[
                Y[0] : Y[0] + y_range, X[0] : X[0] + x_range, Z[0], C[0], T[0]
            ].tobytes()

            self._writer.saveBytes(index, pixel_buffer, X[1], Y[1], x_range, y_range)

        def _write_image(self, X, Y, Z, C, T, image):
            if not self.first_tile:
                args = self._tile_indices.pop(0)
                if (
                    args[0][1] > self.frontend._TILE_SIZE
                    or args[0][1] > self.frontend._TILE_SIZE
                ):
                    raise ValueError(
                        "The first write using the java backend "
                        + "must include the first tile."
                    )
                self._process_chunk(args)
                self.first_tile = True

            for args in self._tile_indices:
                self._process_chunk(args)

        def close(self):
            if jpype.isJVMStarted() and self._writer is not None:
                self._writer.close()

        def __del__(self):
            self.close()

except ModuleNotFoundError:
    logger.warning(
        "Java backend is not available. This could be due to a "
        + "missing dependency (jpype)."
    )

    class JavaReader(bfio.base_classes.AbstractReader):
        def __init__(self, frontend):
            raise ImportError("JavaReader class unavailable. Could not import jpype.")

    class JavaWriter(bfio.base_classes.AbstractWriter):
        def __init__(self, frontend):
            raise ImportError("JavaWriter class unavailable. Could not import jpype.")


try:
    import zarr
    from numcodecs import Blosc

    class ZarrReader(bfio.base_classes.AbstractReader):
        logger = logging.getLogger("bfio.backends.ZarrReader")

        def __init__(self, frontend):
            super().__init__(frontend)

            self.logger.debug("__init__(): Initializing _rdr (zarr)...")
            self.logger.debug(f"Level is {self.frontend.level}")

            try:
                self._root = zarr.open(
                    str(self.frontend._file_path.resolve()), mode="r"
                )
            except zarr.errors.PathNotFoundError:
                # a workaround for pre-compute slide output directory structure
                data_zarr_path = str(self.frontend._file_path.resolve()) + "/data.zarr"
                self._root = zarr.open(data_zarr_path, mode="r")

            if self.frontend.level is None:
                if isinstance(self._root, zarr.core.Array):
                    self._rdr = self._root
                elif isinstance(self._root, zarr.hierarchy.Group):
                    # the top level is a group, check if this has any arrays
                    num_arrays = len(sorted(self._root.array_keys()))
                    if num_arrays > 0:
                        self._rdr = self._root[next(self._root.array_keys())]
                    else:
                        # need to go one more level
                        self._root = self._root[next(self._root.group_keys())]
                        self._rdr = self._root[next(self._root.array_keys())]
                else:
                    pass
            else:
                if isinstance(self._root, zarr.core.Array):
                    self.close()
                    raise ValueError(
                        "Level is specified but the zarr file does not contain "
                        + "multiple resoulutions."
                    )
                elif isinstance(self._root, zarr.hierarchy.Group):
                    if len(sorted(self._root.array_keys())) > self.frontend.level:
                        self._rdr = self._root[self.frontend.level]
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

            self._axes_list = []

        def _get_axis_info(self):
            shape_len = len(self._rdr.shape)
            if shape_len == 5:
                self._axes_list = ["t", "c", "z", "y", "x"]
            else:
                data_key = 0
                if self.frontend.level is not None:
                    data_key = self.frontend.level
                try:
                    axes_metadata = self._root.attrs["multiscales"][data_key]["axes"]
                    for axes in axes_metadata:
                        self._axes_list.append(axes["name"])
                except AttributeError or KeyError:
                    self.logger.warning(
                        "Unable to find multiscales metadata. Z, C and T "
                        + "dimensions might be incorrect."
                    )
                    if shape_len == 4:
                        self._axes_list = ["c", "z", "y", "x"]
                    elif shape_len == 3:
                        self._axes_list = ["z", "y", "x"]
                    elif shape_len == 2:
                        self._axes_list = ["y", "x"]

        def read_metadata(self):
            self.logger.debug("read_metadata(): Reading metadata...")
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
                    self._metadata.images[0].pixels.size_x = self._rdr.shape[-1]
                    self._metadata.images[0].pixels.size_y = self._rdr.shape[-2]

                return self._metadata
            else:
                # Couldn't find OMEXML metadata, scrape metadata from file
                omexml = ome_types.model.OME.model_construct()
                ome_dtype = self._rdr.dtype.name
                if ome_dtype == "float64":
                    ome_dtype = "double"  # to match ome_types pydantic model
                elif ome_dtype == "float32":
                    ome_dtype = "float"  # to match ome_types pydantic model
                else:
                    pass
                # this is speculation, since each array in a group, in theory,
                # can have distinct properties
                ome_dim_order = ome_types.model.Pixels_DimensionOrder.XYZCT
                size_x = 1
                size_y = 1
                size_z = 1
                size_c = 1
                size_t = 1

                assert len(self._rdr.shape) >= 2
                self._get_axis_info()

                if len(self._rdr.shape) == 5:
                    # 5D data, we know what to do
                    size_x = self._rdr.shape[4]
                    size_y = self._rdr.shape[3]
                    size_z = self._rdr.shape[2]
                    size_c = self._rdr.shape[1]
                    size_t = self._rdr.shape[0]
                else:
                    # last two dims are X and Y
                    size_x = self._rdr.shape[-1]
                    size_y = self._rdr.shape[-2]
                    # update z, c and t if any info available
                    if "z" in self._axes_list:
                        size_z = self._rdr.shape[self._axes_list.index("z")]
                    if "c" in self._axes_list:
                        size_c = self._rdr.shape[self._axes_list.index("c")]
                    if "t" in self._axes_list:
                        size_t = self._rdr.shape[self._axes_list.index("t")]

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

                return omexml

        def _process_chunk(self, dims):
            X, Y, Z, C, T = dims

            ts = self.frontend._TILE_SIZE

            if self._axes_list == []:
                self._get_axis_info()

            # actual zarr array can be of 2-5D, but bfio interface
            # is 5D
            requested_slices = []
            if "t" in self._axes_list:
                requested_slices.append(slice(T[1], T[1] + 1))
            if "c" in self._axes_list:
                requested_slices.append(slice(C[1], C[1] + 1))
            if "z" in self._axes_list:
                requested_slices.append(slice(Z[1], Z[1] + 1))

            requested_slices.append(slice(Y[1], Y[1] + ts))
            requested_slices.append(slice(X[1], X[1] + ts))
            data = self._rdr[tuple(requested_slices)].squeeze()
            self._image[
                Y[0] : Y[0] + data.shape[-2],
                X[0] : X[0] + data.shape[-1],
                Z[0],
                C[0],
                T[0],
            ] = data

        def _read_image(self, X, Y, Z, C, T, output):
            if self.frontend._max_workers > 1:
                with ThreadPoolExecutor(self.frontend._max_workers) as executor:
                    executor.map(self._process_chunk, self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)

        def close(self):
            pass

    class ZarrWriter(bfio.base_classes.AbstractWriter):
        logger = logging.getLogger("bfio.backends.ZarrWriter")

        def __init__(self, frontend):
            super().__init__(frontend)

        def _init_writer(self):
            """_init_writer Initializes file writing.

            This method is called exactly once per object. Once it is called,
            all other methods of setting metadata will throw an error.

            NOTE: For Zarr, it is not explicitly necessary to make the file
                  read-only once writing has begun. Thus functionality is mainly
                  incorporated to remain consistent with the OME TIFF formats.
                  In the future, it may be reasonable to not enforce read-only

            """
            if self.frontend.append is False:
                if self.frontend._file_path.exists():
                    shutil.rmtree(self.frontend._file_path)

            shape = (
                self.frontend.T,
                self.frontend.C,
                self.frontend.Z,
                self.frontend.Y,
                self.frontend.X,
            )

            compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
            mode = "w"
            if self.frontend.append is True:
                mode = "a"
            self._root = zarr.open_group(
                store=str(self.frontend._file_path.resolve()), mode=mode
            )

            # Create the metadata
            metadata_path = (
                Path(self.frontend._file_path)
                .joinpath("OME")
                .joinpath("METADATA.ome.xml")
            )

            if self.frontend.append is False or (
                self.frontend.append is True and metadata_path.exists() is False
            ):
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as fw:
                    fw.write(str(self.frontend._metadata.to_xml()))

                self._root.attrs["multiscales"] = [
                    {
                        "version": "0.1",
                        "name": self.frontend._file_path.name,
                        "datasets": [{"path": "0"}],
                        "metadata": {"method": "mean"},
                    }
                ]
            if (
                self.frontend.append is True
                and len(sorted(self._root.array_keys())) > 0
            ):
                writer = self._root["0"]
            else:
                writer = self._root.zeros(
                    "0",
                    shape=shape,
                    chunks=(
                        1,
                        1,
                        1,
                        self.frontend._TILE_SIZE,
                        self.frontend._TILE_SIZE,
                    ),
                    dtype=self.frontend.dtype,
                    compressor=compressor,
                    dimension_separator="/",
                )

            # This is recommended to do for cloud storage to increase read/write
            # speed, but it also increases write speed locally when threading.
            consolidated_metadata_file = Path(self.frontend._file_path).joinpath(
                ".zmetadata"
            )
            if self.frontend.append is False or (
                self.frontend.append is True
                and consolidated_metadata_file.exists() is False
            ):
                zarr.consolidate_metadata(str(self.frontend._file_path.resolve()))

            self._writer = writer

        def _process_chunk(self, dims):
            out = self._image

            X, Y, Z, C, T = dims

            y1e = min([Y[1] + self.frontend._TILE_SIZE, self.frontend._DIMS["Y"]])
            y0e = min([Y[0] + self.frontend._TILE_SIZE, out.shape[0]])
            x1e = min([X[1] + self.frontend._TILE_SIZE, self.frontend._DIMS["X"]])
            x0e = min([X[0] + self.frontend._TILE_SIZE, out.shape[1]])
            self._writer[
                T[1] : T[1] + 1,
                C[1] : C[1] + 1,
                Z[1] : Z[1] + 1,
                Y[1] : y1e,
                X[1] : x1e,
            ] = out[
                Y[0] : y0e,
                X[0] : x0e,
                Z[0] : Z[0] + 1,
                C[0] : C[0] + 1,
                T[0] : T[0] + 1,
            ].transpose(
                4, 3, 2, 0, 1
            )

        def _write_image(self, X, Y, Z, C, T, image):
            if self.frontend._max_workers > 1:
                with ThreadPoolExecutor(self.frontend._max_workers) as executor:
                    executor.map(self._process_chunk, self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)

        def close(self):
            pass

except ModuleNotFoundError:
    logger.info(
        "Zarr backend is not available. This could be due to a "
        + "missing dependency (i.e. zarr)"
    )

    class ZarrReader(bfio.base_classes.AbstractReader):
        def __init__(self, frontend):
            raise ImportError(
                "ZarrReader class unavailable. Could not import" + " zarr."
            )

    class ZarrWriter(bfio.base_classes.AbstractWriter):
        def __init__(self, frontend):
            raise ImportError(
                "ZarrWriter class unavailable. Could not import" + " zarr."
            )
