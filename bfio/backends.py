from tifffile import tifffile
from pathlib import Path
import numpy
from concurrent.futures import ThreadPoolExecutor
import bfio
from bfio.OmeXml import OMEXML
import bfio.base_classes
import struct, copy, zlib, io, typing, logging, threading

logging.basicConfig(format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger("bfio.backends")

class PythonReader(bfio.base_classes.AbstractReader):
    
    logger = logging.getLogger("bfio.backends.PythonReader")
    
    _rdr = None

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug("__init__(): Initializing _rdr (tifffile.TiffFile)...")
        self._rdr = tifffile.TiffFile(self.frontend._file_path)
        metadata = self.read_metadata()
        width = metadata.image().Pixels.get_SizeX()
        height = metadata.image().Pixels.get_SizeY()
        
        for tag in self._rdr.pages[0].tags:
            logger.debug(tag)
        
        tile_size = None
        
        if not self._rdr.pages[0].is_tiled:
            if width > self.frontend._TILE_SIZE or width > self.frontend._TILE_SIZE:
                raise TypeError(frontend._file_path.name + " is not a tiled tiff." +
                                " The python backend of the BioReader only " +
                                "supports OME tiled tiffs. Use the java backend " +
                                "to load this image.")
            
        elif self._rdr.pages[0].tilewidth != self.frontend._TILE_SIZE or \
            self._rdr.pages[0].tilelength != self.frontend._TILE_SIZE:
                
            if (width > frontend._TILE_SIZE or height > frontend._TILE_SIZE):
                raise ValueError("Tile width and height should be {} when ".format(self.frontend._TILE_SIZE) +
                                 "using the python backend, but found " +
                                 "tilewidth={} and tilelength={}. Use the java ".format(self._rdr.pages[0].tilewidth,
                                                                                        self._rdr.pages[0].tilelength) +
                                 "backend to read this image.")
                
        # Private member variables used for reading tiles
        self._keyframe = None       # tifffile object with decompression and chunking methods
        self._image = None          # output image buffer used for threaded tile reading
        self._tile_indices = None   # list that maps file chunks to XYZ coordinates
        
    def read_metadata(self):
        self.logger.debug("read_metadata(): Reading metadata...")
        return OMEXML(self._rdr.ome_metadata)
    
    def _chunk_indices(self,X,Y,Z):
        
        self.logger.debug("_chunk_indices(): (X,Y,Z) -> ({},{},{})".format(X,Y,Z))
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2
        
        offsets = []
        bytecounts = []
        
        ts = self.frontend._TILE_SIZE
        
        x_tiles = numpy.arange(X[0]//ts,numpy.ceil(X[1]/ts),dtype=int)
        y_tile_stride = numpy.ceil(self.frontend.x/ts).astype(int)
        
        self.logger.debug("_chunk_indices(): x_tiles = {}".format(x_tiles))
        self.logger.debug("_chunk_indices(): y_tile_stride = {}".format(y_tile_stride))
        
        for z in range(Z[0],Z[1]):
            for y in range(Y[0]//ts,int(numpy.ceil(Y[1]/ts))):
                y_offset = int(y * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()
                
                o = [self._rdr.pages[z].dataoffsets[i] for i in ind]
                b = [self._rdr.pages[z].databytecounts[i] for i in ind]
                
                self.logger.debug("_chunk_indices(): offsets = {}".format(o))
                self.logger.debug("_chunk_indices(): bytecounts = {}".format(b))
                
                offsets.extend(o)
                bytecounts.extend(b)
        
        return offsets,bytecounts
    
    def _process_chunk(self, args):
        
        keyframe = self._keyframe
        out = self._image
        
        w,l,d,_,_ = self._tile_indices[args[1]]
        
        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)
        
        if segment is None:
            segment = keyframe.nodata
            
        self.logger.debug("_process_chunk(): shape = {}".format(shape))
        self.logger.debug("_process_chunk(): (w,l,d) = {},{},{}".format(w[0],l[0],d[0]))
        
        out[l[0]: l[0] + shape[1],
            w[0]: w[0] + shape[2],
            d[0],0,0] = segment.squeeze()
        
    def _read_image(self,X,Y,Z,C,T,output):
        if (len(C)>1 and C[0]!=0) or (len(T)>0 and T[0]!=0):
            raise Warning("More than channel 0 was specified for either channel or timepoint data." + \
                          "For the Python backend, only the first channel/timepoint will be loaded.")
        
        # Get keyframe and filehandle objects
        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle

        # Get binary data info
        offsets,bytecounts = self._chunk_indices(X,Y,Z)
        
        self.logger.debug("read_image(): _tile_indices = {}".format(self._tile_indices))
        
        if self.frontend.max_workers > 1:
            with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                executor.map(self._process_chunk,fh.read_segments(offsets,bytecounts))
        else:
            for args in fh.read_segments(offsets,bytecounts):
                self._process_chunk(args)

    def close(self):
        self._rdr.close()

class PythonWriter(bfio.base_classes.AbstractWriter):
    _page_open = False
    _current_page = None
    
    logger = logging.getLogger("bfio.backends.PythonWriter")
    
    def __init__(self, frontend):
        super().__init__(frontend)
        
        if self.frontend.C > 1:
            self.logger.warning("The BioWriter only writes single channel " +
                                "images, but the metadata has {} channels. ".format(self.frontend.C) +
                                "Setting the number of channels to 1 " +
                                "and discarding extra channels.")
            self.frontend.C = 1
        if self.frontend.T > 1:
            self.logger.warning("The BioWriter only writes single timepoint " +
                                "images, but the metadata has {} timepoints. ".format(self.frontend.T) +
                                "Setting the number of timepoints to 1." +
                                "and discarding extra timepoints.")
            self.frontend.T = 1

    def _pack(self, fmt, *val):
        return struct.pack(self._byteorder + fmt, *val)
    
    def _addtag(self,code, dtype, count, value, writeonce=False):
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
        ifdentry = [self._pack("HH", code, tifftype),
                    self._pack(self._writer._offsetformat, rawcount)]
        ifdvalue = None
        if struct.calcsize(dtype) * count <= self._writer._offsetsize:
            # value(s) can be written directly
            if isinstance(value, bytes):
                ifdentry.append(self._pack(self._writer._valueformat, value))
            elif count == 1:
                if isinstance(value, (tuple, list, numpy.ndarray)):
                    value = value[0]
                ifdentry.append(self._pack(self._writer._valueformat, self._pack(dtype, value)))
            else:
                ifdentry.append(self._pack(self._writer._valueformat,
                                self._pack(str(count) + dtype, *value)))
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

        self.frontend._metadata.image().set_ID(Path(self.frontend._file_path).name)

        self._writer = tifffile.TiffWriter(self.frontend._file_path, bigtiff=True, append=False)

        self._byteorder = self._writer._byteorder

        self._datashape = (1, 1, 1) + (self.frontend.Y, self.frontend.X) + (1,)
        
        self._datadtype = numpy.dtype(self.frontend.dtype).newbyteorder(self._byteorder)

        tagnoformat = self._writer._tagnoformat
        valueformat = self._writer._valueformat
        offsetformat = self._writer._offsetformat
        offsetsize = self._writer._offsetsize
        tagsize = self._writer._tagsize

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

        description = "".join(["<?xml version='1.0' encoding='UTF-8'?>",
                                "<!-- Warning: this comment is an OME-XML metadata block, which contains crucial dimensional parameters and other important metadata. ",
                                "Please edit cautiously (if at all), and back up the original data before doing so. "
                                "For more information, see the OME-TIFF web site: https://docs.openmicroscopy.org/latest/ome-model/ome-tiff/. -->",
                                str(self.frontend._metadata).replace("ome:", "").replace(":ome", "")])
        
        self._addtag(270, "s", 0, description, writeonce=True)  # Description
        self._addtag(305, "s", 0, f"bfio v{bfio.__version__}")  # Software
        # addtag(306, "s", 0, datetime, writeonce=True)
        self._addtag(259, "H", 1, self._compresstag)  # Compression
        self._addtag(256, "I", 1, self._datashape[-2])  # ImageWidth
        self._addtag(257, "I", 1, self._datashape[-3])  # ImageLength
        self._addtag(322, "I", 1, self.frontend._TILE_SIZE)  # TileWidth
        self._addtag(323, "I", 1, self.frontend._TILE_SIZE)  # TileLength

        sampleformat = {"u": 1, "i": 2, "f": 3, "c": 6}[self._datadtype.kind]
        self._addtag(339, "H", self._samplesperpixel,
                        (sampleformat,) * self._samplesperpixel)

        self._addtag(277, "H", 1, self._samplesperpixel)
        self._addtag(258, "H", 1, self._bitspersample)

        subsampling = None
        maxsampling = 1
        # PhotometricInterpretation
        self._addtag(262, "H", 1, tifffile.TIFF.PHOTOMETRIC.MINISBLACK.value)

        if self.frontend.physical_size_x[0] is not None:
            self._addtag(282, "2I", 1,
                            rational(10000 / self.frontend.physical_size_x[0] ))  # XResolution in pixels/cm
            self._addtag(283, "2I", 1, rational(10000 / self.frontend.physical_size_y[0]))  # YResolution in pixels/cm
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
            if bytecount < 2 ** 16:
                return "H"
            if bytecount < 2 ** 32:
                return "I"
            if size == 4:
                return "I"
            return "Q"

        # can save data array contiguous
        contiguous = False

        # one chunk per tile per plane
        self._tiles = (
            (self._datashape[3] + self.frontend._TILE_SIZE - 1) // self.frontend._TILE_SIZE,
            (self._datashape[4] + self.frontend._TILE_SIZE - 1) // self.frontend._TILE_SIZE,
        )

        self._numtiles = tifffile.product(self._tiles)
        self._databytecounts = [
                                    self.frontend._TILE_SIZE ** 2 * self._datadtype.itemsize] * self._numtiles
        self._bytecountformat = bytecount_format(self._databytecounts)
        self._addtag(self._tagbytecounts, self._bytecountformat, self._numtiles, self._databytecounts)
        self._addtag(self._tagoffsets, self._writer._offsetformat, self._numtiles, [0] * self._numtiles)
        self._bytecountformat = self._bytecountformat * self._numtiles

        # the entries in an IFD must be sorted in ascending order by tag code
        self._tags = sorted(self._tags, key=lambda x: x[0])

    def _open_next_page(self):
        if self._current_page == None:
            self._current_page = 0
        else:
            self._current_page += 1

        if self._current_page == 1:
            for ind, tag in enumerate(self._tags):
                if tag[0] == 270:
                    del self._tags[ind]
                    break
            description = "ImageJ=\nhyperstack=true\nimages={}\nchannels={}\nslices={}\nframes={}".format(
                1, self.frontend.C, self.frontend.Z, self.frontend.T)
            self._addtag(270, "s", 0, description)  # Description
            self._tags = sorted(self._tags, key=lambda x: x[0])

        fh = self._writer._fh

        self._ifdpos = fh.tell()

        tagnoformat = self._writer._tagnoformat
        valueformat = self._writer._valueformat
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
                                self._ifdpos + tagoffset + tagindex * tagsize + 4)
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
        if pos != None:
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
            logger.warning("X or Y positions are not on tile boundary, tile may save incorrectly")

        fh = self._writer._fh

        x_tiles = list(range(X[0] // self.frontend._TILE_SIZE, 1 + (X[1] - 1) // self.frontend._TILE_SIZE))
        tiles = []
        for y in range(Y[0] // self.frontend._TILE_SIZE, 1 + (Y[1] - 1) // self.frontend._TILE_SIZE):
            tiles.extend([y * self._tiles[1] + x for x in x_tiles])

        tile_shape = ((Y[1] - Y[0] - 1 + self.frontend._TILE_SIZE) // self.frontend._TILE_SIZE,
                        (X[1] - X[0] - 1 + self.frontend._TILE_SIZE) // self.frontend._TILE_SIZE)

        data = data.reshape(1, 1, 1, data.shape[0], data.shape[1], 1)
        tileiter = tifffile.iter_tiles(data,
                                        (self.frontend._TILE_SIZE, self.frontend._TILE_SIZE), tile_shape)

        # define compress function
        compressor = tifffile.TIFF.COMPESSORS[self._compresstag]

        def compress(data, level=1):
            data = memoryview(data)
            cpr = zlib.compressobj(level,
                                    memLevel=9,
                                    wbits=15)
            output = b"".join([cpr.compress(data), cpr.flush()])
            return output

        offsetformat = self._writer._offsetformat
        tagnoformat = self._writer._tagnoformat


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
            for tileindex, tile in zip(tiles,tileiter):
                
                t = compress(tile)
                self._databyteoffsets[tileindex] = fh.tell()
                fh.write(t)
                self._databytecounts[tileindex] = len(t)
                
        return None

    def close(self):
        """close_image Close the image

        This function should be called when an image will no longer be written
        to. This allows for proper closing and organization of metadata.
        """
        if self._writer != None:
            if self._page_open:
                self._close_page()
            self._ifd.close()
            self._writer._fh.close()

    def _write_image(self,X,Y,Z,C,T,image):

        if self._current_page != None and Z[0] < self._current_page:
            raise ValueError("Cannot write z layers below the current open page. (current page={},Z[0]={})".format(
                self._current_page, Z[0]))

        # Do the work
        for zi, z in zip(range(0, Z[1] - Z[0]), range(Z[0], Z[1])):
            while z != self._current_page:
                if self._page_open:
                    self._close_page()
                self._open_next_page()
            self._write_tiles(image[..., zi, 0, 0], X, Y)

try:
    import bioformats
    import javabridge
    
    class JavaReader(bfio.base_classes.AbstractReader):
        
        logger = logging.getLogger("bfio.backends.JavaReader")
        _rdr = None
        
        def __init__(self, frontend):
            super().__init__(frontend)
            
            self._rdr = bioformats.ImageReader(str(self.frontend._file_path))
            
            # Test to see if the loci_tools.jar is present
            if bfio.JARS == None:
                raise FileNotFoundError("The loci_tools.jar could not be found.")
            
        def read_metadata(self, update=False):
            # Wrap the ImageReader to get access to additional class methods
            rdr = javabridge.JClassWrapper("loci.formats.ImageReader")()
            
            rdr.setOriginalMetadataPopulated(True)
            
            # Access the OMEXML Service
            clsOMEXMLService = javabridge.JClassWrapper(
                "loci.formats.services.OMEXMLService")
            serviceFactory = javabridge.JClassWrapper(
                "loci.common.services.ServiceFactory")()
            service = serviceFactory.getInstance(clsOMEXMLService.klass)
            omexml = service.createOMEXMLMetadata()
            
            # Read the metadata
            rdr.setMetadataStore(omexml)
            rdr.setId(str(self.frontend._file_path))
            
            # Close the rdr
            rdr.close()
            
            return OMEXML(omexml.dumpXML())
        
        def _process_chunk(self, dims):
            
            self.attach()
            
            out = self._image
            
            X,Y,Z,C,T = dims
            
            self.logger.debug("_process_chunk(): dims = {}".format(dims))
            x_range = min([self.frontend.X, X[1]+1024]) - X[1]
            y_range = min([self.frontend.Y, Y[1]+1024]) - Y[1]

            with bioformats.ImageReader(str(self.frontend._file_path)) as rdr:
                image = rdr.read(c=C[1], z=Z[1], t=T[1],
                                 rescale=False,
                                 XYWH=(X[1], Y[1], x_range, y_range))
            
            out[Y[0]: Y[0]+image.shape[0],
                X[0]: X[0]+image.shape[1],
                Z[0],
                C[0],
                T[0]] = image

            self.detach()
            
        def _read_image(self,X,Y,Z,C,T,output):
            
            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                    executor.map(self._process_chunk,self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)
        
        def attach(self):
            javabridge.attach()
            
        def detach(self):
            javabridge.detach()
            
        def close(self):
            if javabridge.get_env() != None and self._rdr != None:
                self._rdr.close()
        
    class JavaWriter(bfio.base_classes.AbstractWriter):
        
        logger = logging.getLogger("bfio.backends.JavaWriter")
        
        # For Bioformats, the first tile has to be written before any other tile
        first_tile = False
        
        def __init__(self, frontend):
            super().__init__(frontend)
            
            # Test to see if the loci_tools.jar is present
            if bfio.JARS == None:
                raise FileNotFoundError("The loci_tools.jar could not be found.")
            
        def _init_writer(self):
            """_init_writer Initializes file writing.

            This method is called exactly once per object. Once it is
            called, all other methods of setting metadata will throw an
            error.
            
            """
            if self.frontend._file_path.exists():
                self.frontend._file_path.unlink()
                
            class_name = "loci/formats/out/OMETiffWriter"
            IFormatWriter = bioformats.formatwriter.make_iformat_writer_class(class_name)

            class OMETiffWriter(IFormatWriter):

                new_fn = javabridge.make_new("loci/formats/out/OMETiffWriter", "()V")

                def __init__(self):
                    
                    self.new_fn()
                    
                setId = javabridge.make_method("setId", "(Ljava/lang/String;)V",
                                               "Sets the current file name.")
                saveBytesXYWH = javabridge.make_method("saveBytes", "(I[BIIII)V",
                                                       "Saves the given byte array to the current file")
                close = javabridge.make_method("close", "()V",
                                               "Closes currently open file(s) and frees allocated memory.")
                setTileSizeX = javabridge.make_method("setTileSizeX", "(I)I",
                                                      "Set tile size width in pixels.")
                setTileSizeY = javabridge.make_method("setTileSizeY", "(I)I",
                                                      "Set tile size height in pixels.")
                getTileSizeX = javabridge.make_method("getTileSizeX", "()I",
                                                      "Set tile size width in pixels.")
                getTileSizeY = javabridge.make_method("getTileSizeY", "()I",
                                                      "Set tile size height in pixels.")
                setBigTiff = javabridge.make_method("setBigTiff", "(Z)V",
                                                    "Set the BigTiff flag.")
            
            writer = OMETiffWriter()

            # Always set bigtiff flag.
            writer.setBigTiff(True)

            script = """
            importClass(Packages.loci.formats.services.OMEXMLService,
                        Packages.loci.common.services.ServiceFactory);
            var service = new ServiceFactory().getInstance(OMEXMLService);
            var metadata = service.createOMEXMLMetadata(xml);
            var writer = writer
            writer.setMetadataRetrieve(metadata);
            """
            javabridge.run_script(script,
                                  dict(path=str(self.frontend._file_path),
                                       xml=str(self.frontend.metadata).replace("<ome:", "<").replace("</ome:", "</"),
                                       writer=writer))
            writer.setId(str(self.frontend._file_path))
            writer.setInterleaved(False)
            writer.setCompression("LZW")
            x = writer.setTileSizeX(self.frontend._TILE_SIZE)
            y = writer.setTileSizeY(self.frontend._TILE_SIZE)

            self._writer = writer
        
        def _process_chunk(self, dims):
            
            self.attach()
            
            out = self._image
            
            X,Y,Z,C,T = dims
            
            x_range = min([self.frontend.X, X[1]+1024]) - X[1]
            y_range = min([self.frontend.Y, Y[1]+1024]) - Y[1]
            
            self.logger.debug("_process_chunk(): dims = {}".format(dims))
            index = Z[1] + self.frontend.Z * C[1] + \
                    self.frontend.Z * self.frontend.C * T[1]
            pixel_buffer = bioformats.formatwriter.convert_pixels_to_buffer(
                out[Y[0]:Y[0]+y_range, X[0]:X[0]+x_range, Z[0], C[0], T[0]], self.frontend.metadata.image(0).Pixels.PixelType)
            self._writer.saveBytesXYWH(index, pixel_buffer, X[1], Y[1], x_range, y_range)

            self.detach()
        
        def _write_image(self,X,Y,Z,C,T,image):
            
            if not self.first_tile:
                args = self._tile_indices.pop(0)
                if args[0][1] > self.frontend._TILE_SIZE or args[0][1] > self.frontend._TILE_SIZE:
                    raise ValueError("The first write using the java backend " +
                                     "must include the first tile.")
                self._process_chunk(args)
                self.first_tile = True
            
            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                    executor.map(self._process_chunk,self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)
            
        def attach(self):
            javabridge.attach()
            
        def detach(self):
            javabridge.detach()

        def close(self):
            if javabridge.get_env() != None and self._writer != None:
                self._writer.close()

except ModuleNotFoundError:
    
    logger.warning("Java backend is not available. This could be due to a " +
                   "missing dependency (javabridge or bioformats), or " + 
                   "javabridge could not find the java runtime.")
    
    class JavaReader(bfio.base_classes.AbstractReader):
        
        def __init__(self, frontend):
            
            raise ImportError("JavaReader class unavailable. Could not import" +
                              " javabridge and/or bioformats.")
            
    class JavaWriter(bfio.base_classes.AbstractWriter):
        
        def __init__(self, frontend):
            
            raise ImportError("JavaWriter class unavailable. Could not import" +
                              " javabridge and/or bioformats.")