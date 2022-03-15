# import core packages
import copy
import io
import logging
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Third party packages
import imagecodecs
import numpy
import ome_types
import re
from tifffile import tifffile
from xmlschema.validators.exceptions import XMLSchemaValidationError
from lxml.etree import XMLSchemaValidateError
from xml.etree import ElementTree as ET

# bfio internals
import bfio
import bfio.base_classes

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.backends")

KNOWN_INVALID_OME_XSD_REFERENCES = [
    "www.openmicroscopy.org/Schemas/ome/2013-06",
    "www.openmicroscopy.org/Schemas/OME/2012-03",
]
REPLACEMENT_OME_XSD_REFERENCE = "www.openmicroscopy.org/Schemas/OME/2016-06"

ome_formats = {
    "detector_id": lambda x: f"Detector:{x}",
    "instrument_id": lambda x: f"Instrument:{x}",
    "image_id": lambda x: f"Image:{x}",
}


def clean_ome_xml_for_known_issues(xml: str) -> str:
    """Clean an OME XML string.

    This was modified from from AICSImageIO:
    https://github.com/AllenCellModeling/aicsimageio/blob/240c1c76a7e884aa37e11a1fbe0fcbb89fea6515/aicsimageio/metadata/utils.py#L187
    """
    # Store list of changes to print out with warning
    metadata_changes = []

    # Fix xsd reference
    # This is from OMEXML object just having invalid reference
    for known_invalid_ref in KNOWN_INVALID_OME_XSD_REFERENCES:
        if known_invalid_ref in xml:
            xml = xml.replace(
                known_invalid_ref,
                REPLACEMENT_OME_XSD_REFERENCE,
            )
            metadata_changes.append(
                f"Replaced '{known_invalid_ref}' with "
                f"'{REPLACEMENT_OME_XSD_REFERENCE}'."
            )

    # Read in XML
    root = ET.fromstring(xml)

    # Get the namespace
    # In XML etree this looks like
    # "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
    # and must prepend any etree finds
    namespace_matches = re.match(r"\{.*\}", root.tag)
    if namespace_matches is not None:
        namespace = namespace_matches.group(0)
    else:
        raise ValueError("XML does not contain a namespace")

    # Fix MicroManager Instrument and Detector
    instrument = root.find(f"{namespace}Instrument")
    if instrument is not None:
        instrument_id = instrument.get("ID")
        if instrument_id == "Microscope":
            ome_instrument_id = ome_formats["instrument_id"](0)
            instrument.set("ID", ome_instrument_id)
            metadata_changes.append(
                f"Updated attribute 'ID' from '{instrument_id}' to "
                f"'{ome_instrument_id}' on Instrument element."
            )

            for detector_index, detector in enumerate(
                instrument.findall(f"{namespace}Detector")
            ):
                detector_id = detector.get("ID")
                if detector_id is not None:
                    # Create ome detector id if needed
                    ome_detector_id = None
                    if detector_id == "Camera":
                        ome_detector_id = ome_formats["detector_id"](detector_index)
                    elif not detector_id.startswith("Detector:"):
                        ome_detector_id = ome_formats["detector_id"](detector_id)

                    # Apply ome detector id if replaced
                    if ome_detector_id is not None:
                        detector.set("ID", ome_detector_id)
                        metadata_changes.append(
                            f"Updated attribute 'ID' from '{detector_id}' to "
                            f"'{ome_detector_id}' on Detector element at "
                            f"position {detector_index}."
                        )

    # Find all Image elements and fix IDs and refs to fixed instruments
    # This is for certain for test files of o.urs and ACTK files
    for image_index, image in enumerate(root.findall(f"{namespace}Image")):
        image_id = image.get("ID")
        if image_id is not None:
            found_image_id = image_id

            if not found_image_id.startswith("Image"):
                ome_image_id = ome_formats["image_id"](found_image_id)
                image.set("ID", ome_image_id)
                metadata_changes.append(
                    f"Updated attribute 'ID' from '{image_id}' to '{ome_image_id}' "
                    f"on Image element at position {image_index}."
                )

        # Fix MicroManager bad instrument refs
        instrument_ref = image.find(f"{namespace}InstrumentRef")
        if instrument_ref is not None:
            instrument_ref_id = instrument_ref.get("ID")
            if instrument_ref_id == "Microscope":
                instrument_ref.set("ID", ome_instrument_id)

        # Find all Pixels elements and fix IDs
        for pixels_index, pixels in enumerate(image.findall(f"{namespace}Pixels")):
            pixels_id = pixels.get("ID")
            if pixels_id is not None:
                found_pixels_id = pixels_id

                if not found_pixels_id.startswith("Pixels"):
                    pixels.set("ID", f"Pixels:{found_pixels_id}")
                    metadata_changes.append(
                        f"Updated attribute 'ID' from '{found_pixels_id}' to "
                        f"Pixels:{found_pixels_id}' on Pixels element at "
                        f"position {pixels_index}."
                    )

            # Determine if there is an out-of-order channel / plane elem
            # This is due to OMEXML "add channel" function
            # That added Channels and appropriate Planes to the XML
            # But, placed them in:
            # Channel
            # Plane
            # Plane
            # ...
            # Channel
            # Plane
            # Plane
            #
            # Instead of grouped together:
            # Channel
            # Channel
            # ...
            # Plane
            # Plane
            # ...
            #
            # This effects all CFE files (new and old) but for different reasons
            pixels_children_out_of_order = False
            encountered_something_besides_channel = False
            encountered_plane = False
            for child in pixels:
                if child.tag != f"{namespace}Channel":
                    encountered_something_besides_channel = True
                if child.tag == f"{namespace}Plane":
                    encountered_plane = True
                if (
                    encountered_something_besides_channel
                    and child.tag == f"{namespace}Channel"
                ):
                    pixels_children_out_of_order = True
                    break
                if encountered_plane and child.tag in [
                    f"{namespace}{t}" for t in ["BinData", "TiffData", "MetadataOnly"]
                ]:
                    pixels_children_out_of_order = True
                    break

            # Ensure order of:
            # channels -> bindata | tiffdata | metadataonly -> planes
            if pixels_children_out_of_order:
                # Get all relevant elems
                channels = [
                    copy.deepcopy(c) for c in pixels.findall(f"{namespace}Channel")
                ]
                bin_data = [
                    copy.deepcopy(b) for b in pixels.findall(f"{namespace}BinData")
                ]
                tiff_data = [
                    copy.deepcopy(t) for t in pixels.findall(f"{namespace}TiffData")
                ]
                # There should only be one metadata only element but to standardize
                # list comprehensions later we findall
                metadata_only = [
                    copy.deepcopy(m) for m in pixels.findall(f"{namespace}MetadataOnly")
                ]
                planes = [copy.deepcopy(p) for p in pixels.findall(f"{namespace}Plane")]

                # Old (2018 ish) cell feature explorer files sometimes contain both
                # an empty metadata only element and filled tiffdata elements
                # Since the metadata only elements are empty we can check this and
                # choose the tiff data elements instead
                #
                # First check if there are any metadata only elements
                if len(metadata_only) == 1:
                    # Now check if _one of_ of the other two choices are filled
                    # ^ in Python is XOR
                    if (len(bin_data) > 0) ^ (len(tiff_data) > 0):
                        metadata_children = list(metadata_only[0])
                        # Now check if the metadata only elem has no children
                        if len(metadata_children) == 0:
                            # If so, just "purge" by creating empty list
                            metadata_only = []

                        # If there are children elements
                        # Return XML and let XMLSchema Validation show error
                        else:
                            return xml

                # After cleaning metadata only, validate the normal behaviors of
                # OME schema
                #
                # Validate that there is only one of bindata, tiffdata, or metadata
                if len(bin_data) > 0:
                    if len(tiff_data) == 0 and len(metadata_only) == 0:
                        selected_choice = bin_data
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                elif len(tiff_data) > 0:
                    if len(bin_data) == 0 and len(metadata_only) == 0:
                        selected_choice = tiff_data
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                elif len(metadata_only) == 1:
                    if len(bin_data) == 0 and len(tiff_data) == 0:
                        selected_choice = metadata_only
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                else:
                    # Return XML and let XMLSchema Validation show error
                    return xml

                # Remove all children from element to be replaced
                # with ordered elements
                for elem in list(pixels):
                    pixels.remove(elem)

                # Re-attach elements
                for channel in channels:
                    pixels.append(channel)
                for elem in selected_choice:
                    pixels.append(elem)
                for plane in planes:
                    pixels.append(plane)

                metadata_changes.append(
                    f"Reordered children of Pixels element at "
                    f"position {pixels_index}."
                )

    # This is a result of dumping basically all experiement metadata
    # into "StructuredAnnotation" blocks
    #
    # This affects new (2020) Cell Feature Explorer files
    #
    # Because these are structured annotations we don't want to mess with anyones
    # besides the AICS generated bad structured annotations
    aics_anno_removed_count = 0
    sa = root.find(f"{namespace}StructuredAnnotations")
    if sa is not None:
        for xml_anno in sa.findall(f"{namespace}XMLAnnotation"):
            # At least these are namespaced
            if xml_anno.get("Namespace") == "alleninstitute.org/CZIMetadata":
                # Get ID because some elements have annotation refs
                # in both the base Image element and all plane elements
                aics_anno_id = xml_anno.get("ID")
                for image in root.findall(f"{namespace}Image"):
                    for anno_ref in image.findall(f"{namespace}AnnotationRef"):
                        if anno_ref.get("ID") == aics_anno_id:
                            image.remove(anno_ref)

                    # Clean planes
                    if image is not None:
                        found_image = image

                        pixels_planes: Optional[ET.Element] = found_image.find(
                            f"{namespace}Pixels"
                        )
                        if pixels_planes is not None:
                            for plane in pixels_planes.findall(f"{namespace}Plane"):
                                for anno_ref in plane.findall(
                                    f"{namespace}AnnotationRef"
                                ):
                                    if anno_ref.get("ID") == aics_anno_id:
                                        plane.remove(anno_ref)

                # Remove the whole etree
                sa.remove(xml_anno)
                aics_anno_removed_count += 1

    # Log changes
    if aics_anno_removed_count > 0:
        metadata_changes.append(
            f"Removed {aics_anno_removed_count} AICS generated XMLAnnotations."
        )

    # If there are no annotations in StructuredAnnotations, remove it
    if sa is not None:
        if len(list(sa)) == 0:
            root.remove(sa)

    # If any piece of metadata was changed alert and rewrite
    if len(metadata_changes) > 0:

        # Register namespace
        ET.register_namespace("", f"http://{REPLACEMENT_OME_XSD_REFERENCE}")

        # Write out cleaned XML to string
        xml = ET.tostring(
            root,
            encoding="unicode",
            method="xml",
        )

    return xml


class PythonReader(bfio.base_classes.AbstractReader):

    logger = logging.getLogger("bfio.backends.PythonReader")

    _rdr: tifffile.TiffFile = None

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug("__init__(): Initializing _rdr (tifffile.TiffFile)...")
        self._rdr = tifffile.TiffFile(self.frontend._file_path)
        self.read_metadata()
        width = self._metadata.images[0].pixels.size_x
        height = self._metadata.images[0].pixels.size_y

        for tag in self._rdr.pages[0].tags:
            logger.debug(tag)

        if not self._rdr.pages[0].is_tiled or not self._rdr.pages[0].rowsperstrip:

            if (
                self._rdr.pages[0].tilewidth < self.frontend._TILE_SIZE
                or self._rdr.pages[0].tilelength < self.frontend._TILE_SIZE
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
            self._rdr.pages[0].tilewidth != self.frontend._TILE_SIZE
            or self._rdr.pages[0].tilelength != self.frontend._TILE_SIZE
        ):

            if width > frontend._TILE_SIZE or height > frontend._TILE_SIZE:
                self.close()
                raise ValueError(
                    "Tile width and height should be {} when ".format(
                        self.frontend._TILE_SIZE
                    )
                    + "using the python backend, but found "
                    + "tilewidth={} and tilelength={}. Use the java ".format(
                        self._rdr.pages[0].tilewidth, self._rdr.pages[0].tilelength
                    )
                    + "backend to read this image."
                )

    def read_metadata(self):
        self.logger.debug("read_metadata(): Reading metadata...")

        if self._metadata is None:

            try:
                # self._metadata = ome_types.from_xml(self._rdr.ome_metadata)
                self._metadata = ome_types.from_xml(
                    self._rdr.ome_metadata,
                    # Uncomment this when ome_types releases a new version
                    # https://github.com/tlambert03/ome-types/pull/127
                    # validate=True,
                )
            except (XMLSchemaValidationError, XMLSchemaValidateError):
                if self.frontend.clean_metadata:
                    cleaned = clean_ome_xml_for_known_issues(self._rdr.ome_metadata)
                    self._metadata = ome_types.from_xml(
                        cleaned,
                        # Uncomment this when ome_types releases a new version
                        # https://github.com/tlambert03/ome-types/pull/127
                        # validate=True,
                    )
                    self.logger.warning(
                        "read_metadata(): OME XML required reformatting."
                    )
                else:
                    raise

        return self._metadata

    def _chunk_indices(self, X, Y, Z):

        self.logger.debug("_chunk_indices(): (X,Y,Z) -> ({},{},{})".format(X, Y, Z))
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2

        offsets = []
        bytecounts = []

        ts = self.frontend._TILE_SIZE

        x_tiles = numpy.arange(X[0] // ts, numpy.ceil(X[1] / ts), dtype=int)
        y_tile_stride = numpy.ceil(self.frontend.x / ts).astype(int)

        self.logger.debug("_chunk_indices(): x_tiles = {}".format(x_tiles))
        self.logger.debug("_chunk_indices(): y_tile_stride = {}".format(y_tile_stride))

        for z in range(Z[0], Z[1]):
            for y in range(Y[0] // ts, int(numpy.ceil(Y[1] / ts))):
                y_offset = int(y * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()

                o = [self._rdr.pages[z].dataoffsets[i] for i in ind]
                b = [self._rdr.pages[z].databytecounts[i] for i in ind]

                self.logger.debug("_chunk_indices(): offsets = {}".format(o))
                self.logger.debug("_chunk_indices(): bytecounts = {}".format(b))

                offsets.extend(o)
                bytecounts.extend(b)

        return offsets, bytecounts

    def _process_chunk(self, args):

        keyframe = self._keyframe
        out = self._image

        w, l, d, _, _ = self._tile_indices[args[1]]

        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)

        if segment is None:
            segment = keyframe.nodata

        self.logger.debug("_process_chunk(): shape = {}".format(shape))
        self.logger.debug(
            "_process_chunk(): (w,l,d) = {},{},{}".format(w[0], l[0], d[0])
        )

        out[
            l[0] : l[0] + shape[1], w[0] : w[0] + shape[2], d[0], 0, 0
        ] = segment.squeeze()

    def _read_image(self, X, Y, Z, C, T, output):
        if (len(C) > 1 and C[0] != 0) or (len(T) > 0 and T[0] != 0):
            self.logger.warning(
                "More than channel 0 was specified for channel or timepoint data."
                + "The Python backend will load only the first channel/timepoint."
            )

        # Get keyframe
        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle

        # Get binary data info
        offsets, bytecounts = self._chunk_indices(X, Y, Z)

        self.logger.debug("read_image(): _tile_indices = {}".format(self._tile_indices))

        if self.frontend.max_workers > 1:
            with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                executor.map(self._process_chunk, fh.read_segments(offsets, bytecounts))
        else:
            for args in fh.read_segments(offsets, bytecounts):
                self._process_chunk(args)

    def close(self):
        self._rdr.close()

    def __del__(self):
        self.close()


class PythonWriter(bfio.base_classes.AbstractWriter):
    _page_open = False
    _current_page = None

    logger = logging.getLogger("bfio.backends.PythonWriter")

    def __init__(self, frontend):
        super().__init__(frontend)

        if self.frontend.C > 1:
            self.logger.warning(
                "The BioWriter only writes single channel "
                + "images, but the metadata has {} channels. ".format(self.frontend.C)
                + "Setting the number of channels to 1."
            )
            self.frontend.C = 1
        if self.frontend.T > 1:
            self.logger.warning(
                "The BioWriter only writes single timepoint "
                + "images, but the metadata has {} timepoints. ".format(self.frontend.T)
                + "Setting the number of timepoints to 1."
            )
            self.frontend.T = 1

    def _pack(self, fmt, *val):
        return struct.pack(self._byteorder + fmt, *val)

    def _addtag(self, code, dtype, count, value, writeonce=False):
        tags = self._tags

        # compute ifdentry & ifdvalue bytes from code, dtype, count, value
        # append (code, ifdentry, ifdvalue, writeonce) to tags list
        if not isinstance(code, int):
            code = tifffile.TIFF.TAGS[code]
        try:
            tifftype = tifffile.TIFF.DATA_DTYPES[dtype]
        except KeyError as exc:
            raise ValueError(f"unknown dtype {dtype}") from exc
        rawcount = count

        if dtype == "s":
            # strings; enforce 7-bit ASCII on unicode strings
            if code == 270:
                value = tifffile.bytestr(value, "utf-8") + b"\0"
            else:
                value = tifffile.bytestr(value, "ascii") + b"\0"
            count = rawcount = len(value)
            rawcount = value.find(b"\0\0")
            if rawcount < 0:
                rawcount = count
            else:
                rawcount += 1  # length of string without buffer
            value = (value,)
        elif isinstance(value, bytes):
            # packed binary data
            dtsize = struct.calcsize(dtype)
            if len(value) % dtsize:
                raise ValueError("invalid packed binary data")
            count = len(value) // dtsize
        if len(dtype) > 1:
            count *= int(dtype[:-1])
            dtype = dtype[-1]
        ifdentry = [
            self._pack("HH", code, tifftype),
            self._pack(self._writer._offsetformat, rawcount),
        ]
        ifdvalue = None
        if struct.calcsize(dtype) * count <= self._writer._offsetsize:
            # value(s) can be written directly
            if isinstance(value, bytes):
                ifdentry.append(self._pack(self._writer._valueformat, value))
            elif count == 1:
                if isinstance(value, (tuple, list, numpy.ndarray)):
                    value = value[0]
                ifdentry.append(
                    self._pack(self._writer._valueformat, self._pack(dtype, value))
                )
            else:
                ifdentry.append(
                    self._pack(
                        self._writer._valueformat,
                        self._pack(str(count) + dtype, *value),
                    )
                )
        else:
            # use offset to value(s)
            ifdentry.append(self._pack(self._writer._offsetformat, 0))
            if isinstance(value, bytes):
                ifdvalue = value
            elif isinstance(value, numpy.ndarray):
                if value.size != count:
                    raise RuntimeError("value.size != count")
                if value.dtype.char != dtype:
                    raise RuntimeError("value.dtype.char != dtype")
                ifdvalue = value.tobytes()
            elif isinstance(value, (tuple, list)):
                ifdvalue = self._pack(str(count) + dtype, *value)
            else:
                ifdvalue = self._pack(dtype, value)
        tags.append((code, b"".join(ifdentry), ifdvalue, writeonce))

    def _init_writer(self):
        """_init_writer Initializes file writing.

        This method is called exactly once per object. Once it is
        called, all other methods of setting metadata will throw an
        error.

        """

        self._tags = []

        self.frontend._metadata.images[
            0
        ].id = f"Image:{Path(self.frontend._file_path).name}"

        if self.frontend.X * self.frontend.Y * self.frontend.bpp > 2**31:
            big_tiff = True
        else:
            big_tiff = False

        self._writer = tifffile.TiffWriter(
            self.frontend._file_path, bigtiff=big_tiff, append=False
        )

        self._byteorder = self._writer._byteorder

        self._datashape = (1, 1, 1) + (self.frontend.Y, self.frontend.X) + (1,)

        self._datadtype = numpy.dtype(self.frontend.dtype).newbyteorder(self._byteorder)

        offsetsize = self._writer._offsetsize

        # self._compresstag = tifffile.TIFF.COMPRESSION.NONE
        self._compresstag = tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE

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
        self._addtag(305, "s", 0, f"bfio v{bfio.__version__}")  # Software
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
        self._addtag(262, "H", 1, tifffile.TIFF.PHOTOMETRIC.MINISBLACK.value)

        if self.frontend.physical_size_x[0] is not None:
            self._addtag(
                282, "2I", 1, rational(10000 / self.frontend.physical_size_x[0])
            )  # XResolution in pixels/cm
            self._addtag(
                283, "2I", 1, rational(10000 / self.frontend.physical_size_y[0])
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
            self._writer._offsetformat,
            self._numtiles,
            [0] * self._numtiles,
        )
        self._bytecountformat = self._bytecountformat * self._numtiles

        # the entries in an IFD must be sorted in ascending order by tag code
        self._tags = sorted(self._tags, key=lambda x: x[0])

    def _open_next_page(self):
        if self._current_page is None:
            self._current_page = 0
        else:
            self._current_page += 1

        if self._current_page == 1:
            for ind, tag in enumerate(self._tags):
                if tag[0] == 270:
                    del self._tags[ind]
                    break
            description = (
                "ImageJ=\nhyperstack=true\nimages=1\n"
                + f"channels={self.frontend.C}\n"
                + f"slices={self.frontend.Z}\n"
                + "frames={self.frontend.T}"
            )
            self._addtag(270, "s", 0, description)  # Description
            self._tags = sorted(self._tags, key=lambda x: x[0])

        fh = self._writer._fh

        self._ifdpos = fh.tell()

        tagnoformat = self._writer._tagnoformat
        offsetformat = self._writer._offsetformat
        offsetsize = self._writer._offsetsize
        tagsize = self._writer._tagsize
        tagbytecounts = self._tagbytecounts
        tagoffsets = self._tagoffsets
        tags = self._tags

        if self._ifdpos % 2:
            # location of IFD must begin on a word boundary
            fh.write(b"\0")
            self._ifdpos += 1

        # update pointer at ifdoffset
        fh.seek(self._writer._ifdoffset)
        fh.write(self._pack(offsetformat, self._ifdpos))
        fh.seek(self._ifdpos)

        # create IFD in memory, do not write to disk
        if self._current_page < 2:
            self._ifd = io.BytesIO()
            self._ifd.write(self._pack(tagnoformat, len(tags)))
            tagoffset = self._ifd.tell()
            self._ifd.write(b"".join(t[1] for t in tags))
            self._ifdoffset = self._ifd.tell()
            self._ifd.write(self._pack(offsetformat, 0))  # offset to next IFD
            # write tag values and patch offsets in ifdentries
            for tagindex, tag in enumerate(tags):
                offset = tagoffset + tagindex * tagsize + offsetsize + 4
                code = tag[0]
                value = tag[2]
                if value:
                    pos = self._ifd.tell()
                    if pos % 2:
                        # tag value is expected to begin on word boundary
                        self._ifd.write(b"\0")
                        pos += 1
                    self._ifd.seek(offset)
                    self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
                    self._ifd.seek(pos)
                    self._ifd.write(value)
                    if code == tagoffsets:
                        self._dataoffsetsoffset = offset, pos
                    elif code == tagbytecounts:
                        self._databytecountsoffset = offset, pos
                    elif code == 270 and value.endswith(b"\0\0\0\0"):
                        # image description buffer
                        self._descriptionoffset = self._ifdpos + pos
                        self._descriptionlenoffset = (
                            self._ifdpos + tagoffset + tagindex * tagsize + 4
                        )
                elif code == tagoffsets:
                    self._dataoffsetsoffset = offset, None
                elif code == tagbytecounts:
                    self._databytecountsoffset = offset, None
            self._ifdsize = self._ifd.tell()
            if self._ifdsize % 2:
                self._ifd.write(b"\0")
                self._ifdsize += 1

        self._databytecounts = [0 for _ in self._databytecounts]
        self._databyteoffsets = [0 for _ in self._databytecounts]

        # move to file position where data writing will begin
        # will write the tags later when the tile offsets are known
        fh.seek(self._ifdsize, 1)

        # write image data
        self._dataoffset = fh.tell()
        skip = (16 - (self._dataoffset % 16)) % 16
        fh.seek(skip, 1)
        self._dataoffset += skip

        self._page_open = True

    def _close_page(self):

        offsetformat = self._writer._offsetformat
        bytecountformat = self._bytecountformat

        fh = self._writer._fh

        # update strip/tile offsets
        offset, pos = self._dataoffsetsoffset
        self._ifd.seek(offset)
        if pos is not None:
            self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
            self._ifd.seek(pos)
            for size in self._databyteoffsets:
                self._ifd.write(self._pack(offsetformat, size))
        else:
            self._ifd.write(self._pack(offsetformat, self._dataoffset))

        # update strip/tile bytecounts
        offset, pos = self._databytecountsoffset
        self._ifd.seek(offset)
        if pos:
            self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
            self._ifd.seek(pos)
        self._ifd.write(self._pack(bytecountformat, *self._databytecounts))

        self._fhpos = fh.tell()
        fh.seek(self._ifdpos)
        fh.write(self._ifd.getvalue())
        fh.flush()
        fh.seek(self._fhpos)

        self._writer._ifdoffset = self._ifdpos + self._ifdoffset

        # remove tags that should be written only once
        if self._current_page == 0:
            self._tags = [tag for tag in self._tags if not tag[-1]]

        self._page_open = False

    def _write_tiles(self, data, X, Y):

        assert len(X) == 2 and len(Y) == 2

        if X[0] % self.frontend._TILE_SIZE != 0 or Y[0] % self.frontend._TILE_SIZE != 0:
            logger.warning(
                "X or Y positions are not on tile boundary, tile may save incorrectly"
            )

        fh = self._writer._fh

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

        data = data.reshape(1, 1, 1, data.shape[0], data.shape[1], 1)
        tileiter = tifffile.iter_tiles(
            data, (self.frontend._TILE_SIZE, self.frontend._TILE_SIZE), tile_shape
        )

        def compress(data, level=1):

            return imagecodecs.deflate_encode(data, level)

        tileiter = [copy.deepcopy(tile) for tile in tileiter]
        if self.frontend.max_workers > 1:

            with ThreadPoolExecutor(max_workers=self.frontend.max_workers) as executor:

                compressed_tiles = iter(executor.map(compress, tileiter))

            for tileindex in tiles:
                t = next(compressed_tiles)
                self._databyteoffsets[tileindex] = fh.tell()
                fh.write(t)
                self._databytecounts[tileindex] = len(t)
        else:
            for tileindex, tile in zip(tiles, tileiter):

                t = compress(tile)
                self._databyteoffsets[tileindex] = fh.tell()
                fh.write(t)
                self._databytecounts[tileindex] = len(t)

        return None

    def close(self):
        """close_image Close the image.

        This function should be called when an image will no longer be written
        to. This allows for proper closing and organization of metadata.
        """
        if self._writer is not None:
            if self._page_open:
                self._close_page()
            self._ifd.close()
            self._writer._fh.close()

    def _write_image(self, X, Y, Z, C, T, image):

        if self._current_page is not None and Z[0] < self._current_page:
            raise ValueError(
                "Cannot write z layers below the current open page. "
                + "(current page={},Z[0]={})".format(self._current_page, Z[0])
            )

        # Do the work
        for zi, z in zip(range(0, Z[1] - Z[0]), range(Z[0], Z[1])):
            while z != self._current_page:
                if self._page_open:
                    self._close_page()
                self._open_next_page()
            self._write_tiles(image[..., zi, 0, 0], X, Y)

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
                bfio.start()

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

        def read_metadata(self):

            self.logger.debug("read_metadata(): Reading metadata...")

            if self._metadata is None:

                try:
                    self._metadata = ome_types.from_xml(str(self.omexml.dumpXML()))
                except XMLSchemaValidationError:
                    if self.frontend.clean_metadata:
                        self._metadata = ome_types.from_xml(
                            clean_ome_xml_for_known_issues(str(self.omexml.dumpXML()))
                        )
                        self.logger.warning(
                            "read_metadata(): OME XML required reformatting."
                        )
                    else:
                        raise

            return self._metadata

        def _read_image(self, X, Y, Z, C, T, output):

            out = self._image

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

                                image = self._rdr.openBytes(
                                    index, x, y, x_range, y_range
                                )
                                image = numpy.frombuffer(
                                    bytes(image),
                                    self.frontend.dtype,
                                )

                                # TODO: This should be changed in the future
                                # This reloads all channels for a tile on each
                                # loop. Ideally, there would be some better
                                # logic here to only load the necessary channel
                                # information once.
                                if self._rdr.getFormat() not in ["Zeiss CZI"]:
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
                self._rdr.close()

        def __del__(self):
            self.close()

    class JavaWriter(bfio.base_classes.AbstractWriter):

        logger = logging.getLogger("bfio.backends.JavaWriter")

        # For Bioformats, the first tile has to be written before any other tile
        first_tile = False

        _classes_loaded = False

        def _load_java_classes(self):
            if not jpype.isJVMStarted():
                bfio.start()

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

                    for _ in range(expand_channels):
                        image.pixels.channels.append(channel)

            # Test to see if the loci_tools.jar is present
            if bfio.JARS is None:
                raise FileNotFoundError("The bioformats.jar could not be found.")

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

            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                    executor.map(self._process_chunk, self._tile_indices)
            else:

                for args in self._tile_indices:
                    self._process_chunk(args)

        def close(self):
            if jpype.isJVMStarted() and self._writer is not None:
                self._writer.close()

        def __del__(self):

            self.close()

except ModuleNotFoundError:

    logger.info(
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

            self._root = zarr.open(str(self.frontend._file_path.resolve()), mode="r")

            self._rdr = self._root["0"]

        def read_metadata(self):
            self.logger.debug("read_metadata(): Reading metadata...")
            if "metadata" in self._root.attrs.keys():

                if self._metadata is None:

                    try:
                        self._metadata = ome_types.from_xml(
                            self._root.attrs["metadata"]
                            # Uncomment this when ome_types releases a new version
                            # https://github.com/tlambert03/ome-types/pull/127
                            # validate=True,
                        )
                    except XMLSchemaValidationError:
                        if self.frontend.clean_metadata:
                            self._metadata = ome_types.from_xml(
                                clean_ome_xml_for_known_issues(
                                    self._root.attrs["metadata"]
                                    # Uncomment when ome_types releases a new version
                                    # https://github.com/tlambert03/ome-types/pull/127
                                    # validate=True,
                                )
                            )
                            self.logger.warning(
                                "read_metadata(): OME XML required reformatting."
                            )
                        else:
                            raise

                return self._metadata
            else:
                # Couldn't find OMEXML metadata, scrape metadata from file
                omexml = ome_types.model.OME.construct()
                omexml.images[0].Name = Path(self.frontend._file_path).name
                p = omexml.images[0].Pixels

                for i, d in enumerate("XYZCT"):
                    setattr(p, "Size{}".format(d), self._rdr.shape[4 - i])

                p.channel_count = p.SizeC
                for i in range(0, p.SizeC):
                    p.Channel(i).Name = ""

                p.DimensionOrder = ome_types.model.pixels.DimensionOrder.XYZCT

                dtype = numpy.dtype(self._rdr.dtype.name).type
                for k, v in self.frontend._DTYPE.items():
                    if dtype == v:
                        p.PixelType = k
                        break

                return omexml

        def _process_chunk(self, dims):

            X, Y, Z, C, T = dims

            ts = self.frontend._TILE_SIZE

            data = self._rdr[
                T[1], C[1], Z[1] : Z[1] + 1, Y[1] : Y[1] + ts, X[1] : X[1] + ts
            ]

            self._image[
                Y[0] : Y[0] + data.shape[-2],
                X[0] : X[0] + data.shape[-1],
                Z[0] : Z[0] + 1,
                C[0],
                T[0],
            ] = data.transpose(1, 2, 0)

        def _read_image(self, X, Y, Z, C, T, output):

            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
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

            self._root = zarr.group(store=str(self.frontend._file_path.resolve()))

            self._root.attrs["metadata"] = str(self.frontend._metadata)

            self._root.attrs["multiscales"] = [
                {
                    "version": "0.1",
                    "name": self.frontend._file_path.name,
                    "datasets": [{"path": "0"}],
                    "metadata": {"method": "mean"},
                }
            ]

            writer = self._root.zeros(
                "0",
                shape=shape,
                chunks=(1, 1, 1, self.frontend._TILE_SIZE, self.frontend._TILE_SIZE),
                dtype=self.frontend.dtype,
                compressor=compressor,
            )

            # This is recommended to do for cloud storage to increase read/write
            # speed, but it also increases write speed locally when threading.
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

            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
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
