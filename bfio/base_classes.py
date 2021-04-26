import abc, threading, logging
import numpy
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import multiprocessing, typing
from pathlib import Path
import bfio

class BioBase(object,metaclass=abc.ABCMeta) :
    """ Abstract class for reading/writing OME tiled tiff images
    
    Attributes:
        dtype: Gets/sets the pixel type (e.g. uint8)
        channel_names: Gets/sets the names of each channel
        samples_per_pixel: Number of samples per pixel location
        bytes_per_pixel: Number of bytes per pixel
        x: Get/set number of pixels in the x-dimension (width)
        y: Get/set number of pixels in the y-dimension (height)
        z: Get/set number of pixels in the z-dimension (depth)
        c: Get/set number of channels in the image
        t: Get/set number of timepoints in the image
        physical_size_x: Get/set the physical size of the x-dimension
        physical_size_y: Get/set the physical size of the y-dimension
        physical_size_z: Get/set the physical size of the z-dimension
        metadata: OmeXml object for the image
        cnames: Same as channel_names
        spp: Same as samples_per_pixel
        bpp: Same as bytes_per_pixel
        X: Same as x attribute
        Y: Same as y attribute
        Z: Same as z attribute
        C: Same as c attribute
        T: same as t attribute
        ps_x: Same as physical_size_x
        ps_y: Same as physical_size_y
        ps_z: Same as physical_size_z

    """
    # Set constants for reading/writing images
    _MAX_BYTES = 2 ** 30
    _DTYPE = {"uint8": numpy.uint8,
              "int8": numpy.int8,
              "uint16": numpy.uint16,
              "int16": numpy.int16,
              "uint32": numpy.uint32,
              "int32": numpy.int32,
              "float": numpy.float32,
              "double": numpy.float64}
    _BPP = {"uint8": 1,
            "int8": 1,
            "uint16": 2,
            "int16": 2,
            "uint32": 4,
            "int32": 4,
            "float": 4,
            "double": 8}
    _TILE_SIZE = 2 ** 10
    _CHUNK_SIZE = None
    
    _READ_ONLY_MESSAGE = "{} is read-only."
    
    # protected backend object for interfacing with the file on disk
    _backend = None
    
    # protected attribute to hold metadata
    _metadata = None
    
    # protected buffering variables for iterating over an image
    _raw_buffer = Queue(maxsize=1)  # only preload one supertile at a time
    _data_in_buffer = Queue(maxsize=1)
    _supertile_index = Queue()
    _pixel_buffer = None
    _fetch_thread = None
    _tile_thread = None
    _tile_x_offset = 0
    _tile_last_column = 0
    
    def __init__(self,
                 file_path: typing.Union[str,Path],
                 max_workers: typing.Optional[int] = None,
                 backend: typing.Optional[str] ="python",
                 read_only: typing.Optional[bool] = True):
        """__init__ Initialize BioBase object

         Args:
            file_path (str): Path to output file
            max_workers (int,optional): Number of threads to be used.
                Default is None.
            backend (str,optional): Backend to use, must be "python" or "java".
                Default is "python".
        """
        
        # Whether the object is read only
        self.__read_only = read_only
        
        # Internally, keep the file_path as a Path object
        if isinstance(file_path,str):
            file_path = Path(file_path)
        self._file_path = file_path

        # Set the number of workers for multi-threaded loading
        self.max_workers = max_workers if max_workers != None else max([multiprocessing.cpu_count()//2,1])
        
        # Throw an error if an invalid backend is specified
        if backend.lower() not in ["python","java"]:
            raise ValueError("Keyword argument backend must be one of ['python','java']")
        
        # Set the backend
        self._backend_name = backend.lower()
        
        # Create an thread lock for the object
        self._lock =  threading.Lock()
    
    def __setitem__(self,keys: typing.Union[list,tuple],values: numpy.ndarray):
        raise NotImplementedError("Cannot set values for {} class.".format(self.__class__.__name__))
    
    def __getitem__(self,keys):
        raise NotImplementedError("Cannot get values for {} class.".format(self.__class__.__name__))
    
    def _parse_slice(self,keys):
        
        # Dimension ordering and index initialization
        dims = "YXZCT"
        ind = {d:None for d in dims}
        
        # If an empty slice, load the whole image
        if not isinstance(keys,tuple):
            if isinstance(keys,slice) and keys.start == None and keys.stop == None and keys.step==None:
                pass
            else:
                raise ValueError("If only a single index is supplied, it must be an empty slice ([:]).")
            
        # If not an empty slice, parse the key tuple
        else:
            
            # At most, 5 indices can be indicated
            if len(keys) > 5:
                raise ValueError("Found {} indices, but at most 5 indices may be supplied.".format(len(keys)))
            
            # If the first key is an ellipsis, read backwards
            if keys[0] == Ellipsis:
                
                # If the last key is an ellipsis, throw an error
                if keys[-1]==Ellipsis:
                    raise ValueError("Ellipsis (...) may be used in either the first or last index, not both.")
                
                dims = "".join([d for d in reversed(dims)])
                keys = [k for k in reversed(keys)]
            
            # Get key values
            for dim,key in zip(dims,keys):
                
                if isinstance(key,slice):
                    start = 0 if key.start == None else key.start
                    stop = getattr(self,dim) if key.stop == None else key.stop
                    
                    # For CT dimensions, generate a list from slices
                    if dim in "CT":
                        step = 1 if key.step == None else key.step
                        ind[dim] = list(range(start,stop,step))
                    
                    # For XYZ dimensions, get start and stop of slice, ignore step
                    else:
                        ind[dim] = [start,stop]
                        
                elif isinstance(key,(int,tuple,list)):
                    # Only the last two dimensions can use int, tuple, or list indexing
                    if dim in "CT":
                        if isinstance(key,int):
                            ind[dim] = [key]
                        else:
                            ind[dim] = key
                    else:
                        raise ValueError("The index in position {} must be a slice type.".format(dims.find(dim)))
                elif key==Ellipsis:
                    if dims.find(dim)+1 < len(keys):
                        raise ValueError("Ellipsis may only be used in the first or last index.")
                else:
                    raise ValueError("Did not recognize indexing value of type: {}".format(type(key)))
                
        return ind
                
    @property
    def read_only(self) -> bool:
        """Returns true if object is ready only"""
        return self.__read_only

    @read_only.setter
    def read_only(self):
        raise AttributeError(self._READ_ONLY_MESSAGE.format("read_only"))
    
    def __getattribute__(self,name):
        # Get image dimensions using num_x, x, or X
        if (name.startswith("num_") and name[-1] in "xyzct"):
            raise PendingDeprecationWarning(("num_{0} will be deprecated in bfio version 2.1.0.\n" + \
                                             "Currently, num_{0} can only be used to get the dimension." + \
                                             "\tTo get/set the image dimension, " + \
                                             "use the new get/set attribute BioReader.{0}").format(name[-1]))
        if (len(name)==1 and name.lower() in "xyzct"):
            return getattr(self._metadata.image().Pixels,"get_Size{}".format(name.upper()))()
        else:
            return object.__getattribute__(self,name)
        
    def __setattr__(self,name,args):
        # Set image dimensions, for example, using x or X
        if len(name)==1 and name.lower() in "xyzct":
            self.__xyzct_setter(name,args)
        else:
            object.__setattr__(self,name,args)
    
    def __xyzct_setter(self,dimension,value):
        assert not self.__read_only, self._READ_ONLY_MESSAGE.format(dimension.lower())
        assert value >= 1, "{} must be >= 0".format(dimension.upper())
        setattr(self._metadata.image(0).Pixels,"Size{}".format(dimension.upper()),value)
        if dimension.upper() == "C":
            self._metadata.image(0).Pixels.channel_count = value
        self._metadata.image().Pixels.tiffdata_count = self.Z * self.C * self.T
        
        count = 0
        for z in range(self.Z):
            for c in range(self.C):
                for t in range(self.T):
                    self._metadata.image().Pixels.tiffdata(count).FirstZ = z
                    self._metadata.image().Pixels.tiffdata(count).FirstC = c
                    self._metadata.image().Pixels.tiffdata(count).FirstT = t
                    self._metadata.image().Pixels.tiffdata(count).IFD = count
                    self._metadata.image().Pixels.tiffdata(count).plane_count = 1
                    count += 1
    
    """ ------------------------------ """
    """ -Get/Set Dimension Properties- """
    """ ------------------------------ """
    @property
    def channel_names(self) -> typing.List[str]:
        """Get the channel names for the image"""
        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0, self.C)]
        
    @channel_names.setter
    def channel_names(self,cnames: typing.List[str]):
        assert not self.__read_only, self._READ_ONLY_MESSAGE.format("channel_names")
        assert len(cnames) == self.C, "Number of names does not match number of channels."
        for i in range(0, len(cnames)):
            self._metadata.image(0).Pixels.Channel(i).Name = cnames[i]

    @property
    def shape(self) -> typing.Tuple[int,int,int,int,int]:
        """The 5-dimensional shape of the image

        Returns:
            (:attr:`~.Y`, :attr:`~.X`, :attr:`~.Z`, :attr:`~.C`, :attr:`~.T`)
            shape of the image
        """
        return tuple(getattr(self,d) for d in "yxzct")
        
    @shape.setter
    def shape(self,new_shape: typing.Tuple[int,int,int,int,int]):
        assert len(new_shape) == 5
        for s,d in zip(new_shape,"yxzct"):
            setattr(self,d,s)

    @property
    def cnames(self) -> typing.List[str]:
        """Same as :attr:`~.channel_names` """
        return self.channel_names
        
    @cnames.setter
    def cnames(self,cnames: typing.List[str]):
        assert not self.__read_only, self._READ_ONLY_MESSAGE.format("cnames")
        self.channel_names = cnames
            
    def __physical_size(self,dimension,psize,units):
        if psize != None and units != None:
            assert not self.__read_only, self._READ_ONLY_MESSAGE.format("physical_size_{}".format(dimension.lower()))
            setattr(self._metadata.image(0).Pixels,"PhysicalSize{}".format(dimension.upper()),psize)
            setattr(self._metadata.image(0).Pixels,"PhysicalSize{}Unit".format(dimension.upper()),units)

    @property
    def physical_size_x(self) -> typing.Tuple[float,str]:
        """Physical size of pixels in x-dimension

        Returns:
            Units per pixel, Units (i.e. "cm" or "mm")
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeX, self._metadata.image(0).Pixels.PhysicalSizeXUnit)

    @physical_size_x.setter
    def physical_size_x(self,size_units: tuple):
        self.__physical_size("X",*size_units)
        
    @property
    def ps_x(self) -> typing.Tuple[float,str]:
        """Same as :attr:`~bfio.bfio.BioReader.physical_size_x`"""
        return self.physical_size_x

    @ps_x.setter
    def ps_x(self,size_units: tuple):
        self.__physical_size("X",*size_units)
        
    @property
    def physical_size_y(self) -> typing.Tuple[float,str]:
        """Physical size of pixels in y-dimension

        Returns:
            Units per pixel, Units (i.e. "cm" or "mm")
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeY, self._metadata.image(0).Pixels.PhysicalSizeYUnit)

    @physical_size_y.setter
    def physical_size_y(self,size_units: tuple):
        self.__physical_size("Y",*size_units)
        
    @property
    def ps_y(self):
        """Same as :attr:`~bfio.bfio.BioReader.physical_size_y`"""
        return self.physical_size_y

    @ps_y.setter
    def ps_y(self,size_units: tuple):
        self.__physical_size("Y",*size_units)
        
    @property
    def physical_size_z(self) -> typing.Tuple[float,str]:
        """Physical size of pixels in z-dimension

        Returns:
            Units per pixel, Units (i.e. "cm" or "mm")
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeZ, self._metadata.image(0).Pixels.PhysicalSizeZUnit)

    @physical_size_z.setter
    def physical_size_z(self,size_units: tuple):
        self.__physical_size("Z",*size_units)
        
    @property
    def ps_z(self):
        """Same as :attr:`~.physical_size_z`"""
        return self.physical_size_z

    @ps_z.setter
    def ps_z(self,size_units: tuple):
        self.__physical_size("Z",*size_units)

    """ -------------------- """
    """ -Validation methods- """
    """ -------------------- """
    
    def _val_xyz(self, xyz: int, axis: str) -> typing.List[int]:
        """_val_xyz Utility function for validating image dimensions

        Args:
            xyz: Pixel value of x, y, or z dimension.
                If None, returns the maximum range of the dimension
            axis: Must be "X", "Y", or "Z"

        Returns:
            list of ints indicating the first and last index in the dimension
        """
        assert axis in "XYZ"
        
        if xyz == None:
            xyz = [0,getattr(self,axis)]
        else:
            assert len(xyz) == 2, \
                "{} must be a list or tuple of length 2.".format(axis)
            assert xyz[0] >= 0, \
                "{}[0] must be greater than or equal to 0.".format(axis)
            assert xyz[1] <= getattr(self,axis), \
                "{}[1] cannot be greater than the maximum of the dimension ({}).".format(axis, getattr(self,axis))
                
        return xyz

    def _val_ct(self, ct: typing.Union[int,list], axis: str) -> typing.List[int]:
        """_val_ct Utility function for validating image dimensions

        Args:
            ct: List of ints indicating the channels or timepoints to load
                If None, returns a list of ints
            axis: Must be "C", "T"

        Returns:
            list of ints indicating the first and last index in the dimension
        """

        assert axis in "CT"
        
        if ct == None:
            # number of timepoints
            ct = list(range(0, getattr(self,axis)))
        else:
            assert numpy.any(numpy.greater(getattr(self,axis), ct)), \
            "At least one of the {}-indices was larger than largest index ({}).".format(axis, getattr(self,axis) - 1)
            assert numpy.any(numpy.less_equal(0, ct)), \
            "At least one of the {}-indices was less than 0.".format(axis)
            assert len(ct) != 0, \
            "At least one {}-index must be selected.".format(axis)
            
        return ct

    """ ------------------- """
    """ -Pixel information- """
    """ ------------------- """
    
    @property
    def dtype(self) -> numpy.dtype:
        """The numpy pixel type of the data"""
        return self._DTYPE[self._metadata.image(0).Pixels.PixelType]
    
    @dtype.setter
    def dtype(self,dtype):
        assert not self.__read_only, self._READ_ONLY_MESSAGE.format("dtype")
        if dtype in [numpy.uint64, numpy.int64]:
            self.logger.warning(f"{dtype} is not supported by Bioformats, saving as numpy.float64.")
            dtype = numpy.float64
        assert dtype in self._DTYPE.values(), "Invalid data type."
        for k,v in self._DTYPE.items():
            if dtype==v:
                self._metadata.image(0).Pixels.PixelType = k
                return
        
    @property
    def samples_per_pixel(self) -> int:
        """Number of samples per pixel"""
        return self._metadata.image().Pixels.Channel().SamplesPerPixel
    
    @samples_per_pixel.setter
    def samples_per_pixel(self,
                          samples_per_pixel: int):
        self._metadata.image().Pixels.Channel().SamplesPerPixel = samples_per_pixel

    @property
    def spp(self):
        """Same as :attr:`.samples_per_pixel`"""
        return self.samples_per_pixel
    
    @spp.setter
    def spp(self,samples_per_pixel):
        self.samples_per_pixel(samples_per_pixel)
        
    @property
    def bytes_per_pixel(self) -> int:
        """Number of bytes per pixel"""
        return self._BPP[self._metadata.image().Pixels.get_PixelType()]
    
    @bytes_per_pixel.setter
    def bytes_per_pixel(self,
                        bytes_per_pixel: int):
        raise AttributeError("Bytes per pixel cannot be set. Change the dtype instead")
    
    @property
    def bpp(self):
        """Same as :attr:`.bytes_per_pixel`"""
        return self.bytes_per_pixel
    
    @bpp.setter
    def bpp(self,bytes_per_pixel: int):
        self.bytes_per_pixel = bytes_per_pixel

    """ -------------------------- """
    """ -Other Methods/Properties- """
    """ -------------------------- """
    @property
    def metadata(self) -> bfio.OmeXml.OMEXML:
        """Get the metadata for the image

        This function calls the Bioformats metadata parser, which extracts
        metadata from an image. This returns a reference to an OMEXML class,
        which is a convenient handler for the complex xml metadata created by
        Bioformats.

        Most basic metadata information have their own BioReader methods, such
        as image dimensions(i.e. x, y, etc). However, in some cases it may be
        necessary to access the underlying metadata class.
        
        Minor changes have been made to the original OMEXML class created for
        python-bioformats, so the original OMEXML documentation should assist
        those interested in directly accessing the metadata. In general, it is
        best to assign data using the object properties to ensure the metadata
        stays in sync with the file.

        For information on the OMEXML class:
        https://github.com/CellProfiler/python-bioformats/blob/master/bioformats/omexml.py

        Returns:
            OMEXML object for the image
        """        
        return self._metadata
    
    @metadata.setter
    def metadata(self,value):
        raise AttributeError("The metadata attribute is read-only. Components" +
                             " of the metadata can be modified by getting" +
                             " the metadata object and making changes, or by" +
                             " changing the attriutes of the image.")

    def maximum_batch_size(self,
                           tile_size: typing.List[int],
                           tile_stride: typing.Union[typing.List[int],None] = None) -> int:
        """maximum_batch_size Maximum allowable batch size for tiling
        
        The pixel buffer only loads at most two supertiles at a time. If the
        batch size is too large, then the tiling function will attempt to create
        more tiles than what the buffer holds. To prevent the tiling function
        from doing this, there is a limit on the number of tiles that can be
        retrieved in a single call. This function determines what the largest
        number of retreivable batches is.
        
        Args:
            tile_size: The height and width of the tiles to retrieve
            tile_stride: If None, defaults to tile_size. *Defaults to None.*
        
        Returns:
            Maximum allowed number of batches that can be retrieved by the
                iterate method.
        """
        if tile_stride == None:
            tile_stride = tile_size

        xyoffset = [(tile_size[0] - tile_stride[0]) / 2, (tile_size[1] - tile_stride[1]) / 2]

        num_tile_rows = int(numpy.ceil(self.Y / tile_stride[0]))
        num_tile_cols = (1024 - xyoffset[1]) // tile_stride[1]
        if num_tile_cols == 0:
            num_tile_cols = 1

        return int(num_tile_cols * num_tile_rows)
    
    def close(self):
        """Close the image"""
        if self._backend is not None:
            self._backend.close()
        
    def __enter__(self):
        """Handle entrance to a context manager
        
        This code is called when a `with` statement is used. This allows a
        BioBase object to be used like this:
        
        with bfio.BioReader("Path/To/File.ome.tif") as reader:
            ...
            
        with bfio.BioWriter("Path/To/File.ome.tif") as writer:
            ...
        """
        return self
    
    def __del__(self):
        """Handle file deletion

        This code runs when an object is deleted..
        """
        self.close()
        
    
    def __exit__(self, type_class, value, traceback):
        """Handle exit from the context manager

        This code runs when exiting a `with` statement.
        """
        self.close()
        
class AbstractBackend(object,metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self,frontend):
        self.frontend = frontend
        self._lock = threading.Lock()

    def _image_io(self,X,Y,Z,C,T,image):
            
        # Define tile bounds
        ts = self.frontend._TILE_SIZE
        X_tile_shape = X[1] - X[0]
        Y_tile_shape = Y[1] - Y[0]
        Z_tile_shape = Z[1] - Z[0]
        
        # Set the output for asynchronous reading
        self._image = image

        # Set up the tile indices
        self._tile_indices = []
        for t in range(len(T)):
            for c in range(len(C)):
                for z in range(Z_tile_shape):
                    for y in range(0,Y_tile_shape,ts):
                        for x in range(0,X_tile_shape,ts):
                            self._tile_indices.append(((x,X[0]+x),
                                                       (y,Y[0]+y),
                                                       (z,Z[0]+z),
                                                       (c,C[c]),
                                                       (t,T[t])))
        
        self.logger.debug("_image_io(): _tile_indices = {}".format(self._tile_indices))
        
    @abc.abstractmethod
    def close(self):
        pass

class AbstractReader(AbstractBackend):
    
    @abc.abstractmethod
    def __init__(self,frontend):
        super().__init__(frontend)
    
    @abc.abstractmethod
    def read_metadata(self):
        pass
    
    def read_image(self,*args):
        with self._lock:
            self._image_io(*args)
            self._read_image(*args)

    @abc.abstractmethod
    def _read_image(self,X,Y,Z,C,T,output):
        pass

class AbstractWriter(AbstractBackend):
    
    _writer = None
    
    @abc.abstractmethod
    def __init__(self,frontend):
        super().__init__(frontend)
        self.initialized = False
    
    def write_image(self,*args):
        with self._lock:
            if not self.initialized:
                self._init_writer()
                self.frontend.__read_only = True
                self.initialized = True
            
            self._image_io(*args)
            self._write_image(*args)
        
    @abc.abstractmethod
    def _init_writer(self):
        pass
    
    @abc.abstractmethod
    def _write_image(*args):
        pass