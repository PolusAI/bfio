# import core packages
import struct, copy, zlib, io, typing, logging, threading, shutil
from concurrent.futures import ThreadPoolExecutor
from tifffile import tifffile
from pathlib import Path

# Third party packages
import numpy
import imagecodecs

# bfio internals
import bfio
from bfio.OmeXml import OMEXML
from bfio import OmeXml
import bfio.base_classes

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('bfio.backends')

class PythonReader(bfio.base_classes.AbstractReader):

    logger = logging.getLogger('bfio.backends.PythonReader')

    _rdr = None

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug('__init__(): Initializing _rdr (tifffile.TiffFile)...')
        self._rdr = tifffile.TiffFile(self.frontend._file_path)
        metadata = self.read_metadata()
        width = metadata.image().Pixels.get_SizeX()
        height = metadata.image().Pixels.get_SizeY()

        for tag in self._rdr.pages[0].tags:
            logger.debug(tag)

        tile_size = None

        if not self._rdr.pages[0].is_tiled or not self._rdr.pages[0].rowsperstrip:
            self.close()
            raise TypeError(frontend._file_path.name + ' is not a tiled tiff.' +
                            ' The python backend of the BioReader only ' +
                            'supports OME tiled tiffs. Use the java backend ' +
                            'to load this image.')

        elif self._rdr.pages[0].tilewidth != self.frontend._TILE_SIZE or \
            self._rdr.pages[0].tilelength != self.frontend._TILE_SIZE:

            if (width > frontend._TILE_SIZE or height > frontend._TILE_SIZE):
                self.close()
                raise ValueError('Tile width and height should be {} when '.format(self.frontend._TILE_SIZE) +
                                 'using the python backend, but found ' +
                                 'tilewidth={} and tilelength={}. Use the java '.format(self._rdr.pages[0].tilewidth,
                                                                                        self._rdr.pages[0].tilelength) +
                                 'backend to read this image.')

    def read_metadata(self):
        self.logger.debug('read_metadata(): Reading metadata...')
        return OMEXML(self._rdr.ome_metadata)

    def _chunk_indices(self,X,Y,Z):

        self.logger.debug('_chunk_indices(): (X,Y,Z) -> ({},{},{})'.format(X,Y,Z))
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2

        offsets = []
        bytecounts = []

        ts = self.frontend._TILE_SIZE

        x_tiles = numpy.arange(X[0]//ts,numpy.ceil(X[1]/ts),dtype=int)
        y_tile_stride = numpy.ceil(self.frontend.x/ts).astype(int)

        self.logger.debug('_chunk_indices(): x_tiles = {}'.format(x_tiles))
        self.logger.debug('_chunk_indices(): y_tile_stride = {}'.format(y_tile_stride))

        for z in range(Z[0],Z[1]):
            for y in range(Y[0]//ts,int(numpy.ceil(Y[1]/ts))):
                y_offset = int(y * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()

                o = [self._rdr.pages[z].dataoffsets[i] for i in ind]
                b = [self._rdr.pages[z].databytecounts[i] for i in ind]

                self.logger.debug('_chunk_indices(): offsets = {}'.format(o))
                self.logger.debug('_chunk_indices(): bytecounts = {}'.format(b))

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

        self.logger.debug('_process_chunk(): shape = {}'.format(shape))
        self.logger.debug('_process_chunk(): (w,l,d) = {},{},{}'.format(w[0],l[0],d[0]))

        out[l[0]: l[0] + shape[1],
            w[0]: w[0] + shape[2],
            d[0],0,0] = segment.squeeze()

    def _read_image(self,X,Y,Z,C,T,output):
        if (len(C)>1 and C[0]!=0) or (len(T)>0 and T[0]!=0):
            self.logger.warning('More than channel 0 was specified for either channel or timepoint data.' + \
                                'For the Python backend, only the first channel/timepoint will be loaded.')

        # Get keyframe
        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle

        # Get binary data info
        offsets,bytecounts = self._chunk_indices(X,Y,Z)

        self.logger.debug('read_image(): _tile_indices = {}'.format(self._tile_indices))

        if self.frontend.max_workers > 1:
            with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                executor.map(self._process_chunk,fh.read_segments(offsets,bytecounts))
        else:
            for args in fh.read_segments(offsets,bytecounts):
                self._process_chunk(args)

    def close(self):
        self._rdr.close()
        
    def __del__(self):
        self.close()

class PythonWriter(bfio.base_classes.AbstractWriter):
    _page_open = False
    _current_page = None

    logger = logging.getLogger('bfio.backends.PythonWriter')

    def __init__(self, frontend):
        super().__init__(frontend)

        if self.frontend.C > 1:
            self.logger.warning('The BioWriter only writes single channel ' +
                                'images, but the metadata has {} channels. '.format(self.frontend.C) +
                                'Setting the number of channels to 1.')
            self.frontend.C = 1
        if self.frontend.T > 1:
            self.logger.warning('The BioWriter only writes single timepoint ' +
                                'images, but the metadata has {} timepoints. '.format(self.frontend.T) +
                                'Setting the number of timepoints to 1.')
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
            raise ValueError(f'unknown dtype {dtype}') from exc
        rawcount = count

        if dtype == 's':
            # strings; enforce 7-bit ASCII on unicode strings
            if code == 270:
                value = tifffile.bytestr(value, 'utf-8') + b'\0'
            else:
                value = tifffile.bytestr(value, 'ascii') + b'\0'
            count = rawcount = len(value)
            rawcount = value.find(b'\0\0')
            if rawcount < 0:
                rawcount = count
            else:
                rawcount += 1  # length of string without buffer
            value = (value,)
        elif isinstance(value, bytes):
            # packed binary data
            dtsize = struct.calcsize(dtype)
            if len(value) % dtsize:
                raise ValueError('invalid packed binary data')
            count = len(value) // dtsize
        if len(dtype) > 1:
            count *= int(dtype[:-1])
            dtype = dtype[-1]
        ifdentry = [self._pack('HH', code, tifftype),
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
                    raise RuntimeError('value.size != count')
                if value.dtype.char != dtype:
                    raise RuntimeError('value.dtype.char != dtype')
                ifdvalue = value.tobytes()
            elif isinstance(value, (tuple, list)):
                ifdvalue = self._pack(str(count) + dtype, *value)
            else:
                ifdvalue = self._pack(dtype, value)
        tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

    def _init_writer(self):
        """_init_writer Initializes file writing.

        This method is called exactly once per object. Once it is
        called, all other methods of setting metadata will throw an
        error.

        """

        self._tags = []

        self.frontend._metadata.image().set_ID(Path(self.frontend._file_path).name)

        if self.frontend.X * self.frontend.Y * self.frontend.bpp > 2**31:
            big_tiff = True
        else:
            big_tiff = False

        self._writer = tifffile.TiffWriter(self.frontend._file_path,
                                           bigtiff=big_tiff,
                                           append=False)

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

        description = ''.join(['<?xml version="1.0" encoding="UTF-8"?>',
                                '<!-- Warning: this comment is an OME-XML metadata block, which contains crucial dimensional parameters and other important metadata. ',
                                'Please edit cautiously (if at all), and back up the original data before doing so. '
                                'For more information, see the OME-TIFF web site: https://docs.openmicroscopy.org/latest/ome-model/ome-tiff/. -->',
                                str(self.frontend._metadata).replace('ome:', '').replace(':ome', '')])

        self._addtag(270, 's', 0, description, writeonce=True)  # Description
        self._addtag(305, 's', 0, f'bfio v{bfio.__version__}')  # Software
        # addtag(306, 's', 0, datetime, writeonce=True)
        self._addtag(259, 'H', 1, self._compresstag)  # Compression
        self._addtag(256, 'I', 1, self._datashape[-2])  # ImageWidth
        self._addtag(257, 'I', 1, self._datashape[-3])  # ImageLength
        self._addtag(322, 'I', 1, self.frontend._TILE_SIZE)  # TileWidth
        self._addtag(323, 'I', 1, self.frontend._TILE_SIZE)  # TileLength

        sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[self._datadtype.kind]
        self._addtag(339, 'H', self._samplesperpixel,
                        (sampleformat,) * self._samplesperpixel)

        self._addtag(277, 'H', 1, self._samplesperpixel)
        self._addtag(258, 'H', 1, self._bitspersample)

        subsampling = None
        maxsampling = 1
        # PhotometricInterpretation
        self._addtag(262, 'H', 1, tifffile.TIFF.PHOTOMETRIC.MINISBLACK.value)

        if self.frontend.physical_size_x[0] is not None:
            self._addtag(282, '2I', 1,
                            rational(10000 / self.frontend.physical_size_x[0] ))  # XResolution in pixels/cm
            self._addtag(283, '2I', 1, rational(10000 / self.frontend.physical_size_y[0]))  # YResolution in pixels/cm
            self._addtag(296, 'H', 1, 3)  # ResolutionUnit = cm
        else:
            self._addtag(282, '2I', 1, (1, 1))  # XResolution
            self._addtag(283, '2I', 1, (1, 1))  # YResolution
            self._addtag(296, 'H', 1, 1)  # ResolutionUnit

        def bytecount_format(bytecounts, size=offsetsize):
            # return small bytecount format
            if len(bytecounts) == 1:
                return {4: 'I', 8: 'Q'}[size]
            bytecount = bytecounts[0] * 10
            if bytecount < 2 ** 16:
                return 'H'
            if bytecount < 2 ** 32:
                return 'I'
            if size == 4:
                return 'I'
            return 'Q'

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
            description = 'ImageJ=\nhyperstack=true\nimages={}\nchannels={}\nslices={}\nframes={}'.format(
                1, self.frontend.C, self.frontend.Z, self.frontend.T)
            self._addtag(270, 's', 0, description)  # Description
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
            fh.write(b'\0')
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
            self._ifd.write(b''.join(t[1] for t in tags))
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
                        self._ifd.write(b'\0')
                        pos += 1
                    self._ifd.seek(offset)
                    self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
                    self._ifd.seek(pos)
                    self._ifd.write(value)
                    if code == tagoffsets:
                        self._dataoffsetsoffset = offset, pos
                    elif code == tagbytecounts:
                        self._databytecountsoffset = offset, pos
                    elif code == 270 and value.endswith(b'\0\0\0\0'):
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
                self._ifd.write(b'\0')
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
            logger.warning('X or Y positions are not on tile boundary, tile may save incorrectly')

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

            return imagecodecs.deflate_encode(data,level)

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
            raise ValueError('Cannot write z layers below the current open page. (current page={},Z[0]={})'.format(
                self._current_page, Z[0]))

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
    from jpype.types import *

    class JavaReader(bfio.base_classes.AbstractReader):

        logger = logging.getLogger('bfio.backends.JavaReader')
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

        def read_metadata(self, update=False):

            return OMEXML(str(self.omexml.dumpXML()))

        def _read_image(self,X,Y,Z,C,T,output):

            out = self._image

            for ti,t in enumerate(T):
                for zi,z in enumerate(range(Z[0],Z[1])):
                    for ci,c in enumerate(C):
                        
                        # TODO: This should be changed in the future
                        # See the comment below about properly handling
                        # interleaved channel data.
                        if self.frontend.spp > 1:
                            index = self._rdr.getIndex(z, 0, t)
                        else:
                            index = self._rdr.getIndex(z, c, t)
                            
                        x_max = min([X[1],self.frontend.X])
                        for x in range(X[0],x_max,self._chunk_size):
                            x_range = min([self._chunk_size,x_max-x])

                            y_max = min([Y[1],self.frontend.Y])
                            for y in range(Y[0],y_max,self._chunk_size):
                                y_range = min([self._chunk_size,y_max-y])

                                image = self._rdr.openBytes(index, x, y, x_range, y_range)
                                image = numpy.frombuffer(bytes(image),self.frontend.dtype)
                                
                                # TODO: This should be changed in the future
                                # This reloads all channels for a tile on each
                                # loop. Ideally, there would be some better
                                # logic here to only load the necessary channel
                                # information once.
                                if self.frontend.spp > 1:
                                    image = image.reshape(self.frontend.c,y_range,x_range)
                                    image = image[c,...].squeeze()
                                    print(image.shape)
                                    
                                print(z)
                                out[y: y+y_range,
                                    x: x+x_range,
                                    zi,
                                    ci,
                                    ti] = image.reshape(y_range,x_range)

        def close(self):
            if jpype.isJVMStarted() and self._rdr != None:
                self._rdr.close()
                
        def __del__(self):
            self.close()

    class JavaWriter(bfio.base_classes.AbstractWriter):

        logger = logging.getLogger('bfio.backends.JavaWriter')

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

            # Test to see if the loci_tools.jar is present
            if bfio.JARS == None:
                raise FileNotFoundError('The loci_tools.jar could not be found.')

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
            service = ServiceFactory().getInstance(OMEXMLService);
            xml = str(self.frontend.metadata).replace('<ome:', '<').replace('</ome:', '</')
            metadata = service.createOMEXMLMetadata(xml)
            writer.setMetadataRetrieve(metadata);

            # Set the file path
            writer.setId(str(self.frontend._file_path))

            # We only save a single channel, so interleaved is unecessary
            writer.setInterleaved(False)

            # Set compression to
            writer.setCompression(CompressionType.ZLIB.getCompression())

            # Set tile sizes
            x = writer.setTileSizeX(self.frontend._TILE_SIZE)
            y = writer.setTileSizeY(self.frontend._TILE_SIZE)

            self._writer = writer

        def _process_chunk(self, dims):

            out = self._image

            X,Y,Z,C,T = dims

            x_range = min([self.frontend.X, X[1]+1024]) - X[1]
            y_range = min([self.frontend.Y, Y[1]+1024]) - Y[1]

            self.logger.debug('_process_chunk(): dims = {}'.format(dims))
            pixel_buffer = out[Y[0]:Y[0]+y_range, X[0]:X[0]+x_range, Z[0], C[0], T[0]].tobytes()
            self._writer.saveBytes(Z[1], pixel_buffer, X[1], Y[1], x_range, y_range)

        def _write_image(self,X,Y,Z,C,T,image):

            if not self.first_tile:
                args = self._tile_indices.pop(0)
                if args[0][1] > self.frontend._TILE_SIZE or args[0][1] > self.frontend._TILE_SIZE:
                    raise ValueError('The first write using the java backend ' +
                                     'must include the first tile.')
                self._process_chunk(args)
                self.first_tile = True

            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                    executor.map(self._process_chunk,self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)

        def close(self):
            if jpype.isJVMStarted() and self._writer != None:
                self._writer.close()
                
        def __del__(self):
            
            self.close()

except ModuleNotFoundError:

    logger.warning('Java backend is not available. This could be due to a ' +
                   'missing dependency (jpype).')

    class JavaReader(bfio.base_classes.AbstractReader):

        def __init__(self, frontend):

            raise ImportError('JavaReader class unavailable. Could not import' +
                              ' jpype.')

    class JavaWriter(bfio.base_classes.AbstractWriter):

        def __init__(self, frontend):

            raise ImportError('JavaWriter class unavailable. Could not import' +
                              ' jpype.')

try:
    import zarr
    from numcodecs import Blosc, blosc
    # blosc.use_threads = True

    class ZarrReader(bfio.base_classes.AbstractReader):

        logger = logging.getLogger('bfio.backends.ZarrReader')

        def __init__(self,frontend):
            super().__init__(frontend)

            self.logger.debug('__init__(): Initializing _rdr (zarr)...')

            self._root = zarr.open(str(self.frontend._file_path.resolve()),
                                   mode='r')

            self._rdr = self._root['0']

        def read_metadata(self):
            self.logger.debug('read_metadata(): Reading metadata...')
            if 'metadata' in self._root.attrs.keys():
                return OMEXML(self._root.attrs['metadata'])
            else:
                # Couldn't find OMEXML metadata, scrape metadata from file
                omexml = OMEXML()
                omexml.image(0).Name = Path(self.frontend._file_path).name
                p = omexml.image(0).Pixels

                for i,d in enumerate('XYZCT'):
                    setattr(p,'Size{}'.format(d),self._rdr.shape[4-i])
                
                p.channel_count = p.SizeC
                for i in range(0, p.SizeC):
                    p.Channel(i).Name = ''

                p.DimensionOrder = OmeXml.DO_XYZCT
                
                dtype = numpy.dtype(self._rdr.dtype.name).type
                for k,v in self.frontend._DTYPE.items():
                    if dtype==v:
                        p.PixelType = k
                        break

                return omexml

        def _process_chunk(self, dims):

            X,Y,Z,C,T = dims

            ts = self.frontend._TILE_SIZE

            data = self._rdr[T[1],C[1],
                             Z[1]:Z[1]+1,
                             Y[1]:Y[1]+ts,
                             X[1]:X[1]+ts]

            self._image[Y[0]:Y[0]+data.shape[-2],
                        X[0]:X[0]+data.shape[-1],
                        Z[0]:Z[0]+1,C[0],T[0]] = data.transpose(1,2,0)

        def _read_image(self,X,Y,Z,C,T,output):

            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                    executor.map(self._process_chunk,self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)


        def close(self):
            pass

    class ZarrWriter(bfio.base_classes.AbstractWriter):

        logger = logging.getLogger('bfio.backends.ZarrWriter')

        def __init__(self,frontend):
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
                self.frontend.X
            )

            compressor = Blosc(cname='zstd', clevel=1,shuffle=Blosc.SHUFFLE)

            self._root = zarr.group(store=str(self.frontend._file_path.resolve()))

            self._root.attrs['metadata'] = str(self.frontend._metadata)
            
            self._root.attrs['multiscales'] = [{
                "version": "0.1",
                "name": self.frontend._file_path.name,
                "datasets": [{
                    "path": "0"
                }],
                "metadata": {
                    "method": "mean"
                }
            }]

            writer = self._root.zeros('0',
                                      shape=shape,
                                      chunks=(1,1,1,self.frontend._TILE_SIZE,self.frontend._TILE_SIZE),
                                      dtype=self.frontend.dtype,
                                      compressor=compressor)

            # This is recommended to do for cloud storage to increase read/write
            # speed, but it also increases write speed locally when threading.
            zarr.consolidate_metadata(str(self.frontend._file_path.resolve()))

            self._writer = writer

        def _process_chunk(self,dims):

            out = self._image

            X,Y,Z,C,T = dims

            y1e = min([Y[1]+self.frontend._TILE_SIZE,self.frontend._DIMS["Y"]])
            y0e = min([Y[0]+self.frontend._TILE_SIZE,out.shape[0]])
            x1e = min([X[1]+self.frontend._TILE_SIZE,self.frontend._DIMS["X"]])
            x0e = min([X[0]+self.frontend._TILE_SIZE,out.shape[1]])
            self._writer[T[1]:T[1]+1,C[1]:C[1]+1,Z[1]:Z[1]+1,Y[1]:y1e,X[1]:x1e] = \
                out[Y[0]:y0e,X[0]:x0e,Z[0]:Z[0]+1,C[0]:C[0]+1,T[0]:T[0]+1].transpose(4,3,2,0,1)

        def _write_image(self,X,Y,Z,C,T,image):

            if self.frontend.max_workers > 1:
                with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                    executor.map(self._process_chunk,self._tile_indices)
            else:
                for args in self._tile_indices:
                    self._process_chunk(args)

        def close(self):
            pass

except ModuleNotFoundError:

    logger.warning('Zarr backend is not available. This could be due to a ' +
                   'missing dependency (i.e. zarr)')

    class ZarrReader(bfio.base_classes.AbstractReader):

        def __init__(self, frontend):

            raise ImportError('ZarrReader class unavailable. Could not import' +
                              ' zarr.')

    class ZarrWriter(bfio.base_classes.AbstractWriter):

        def __init__(self, frontend):

            raise ImportError('ZarrWriter class unavailable. Could not import' +
                              ' zarr.')
