=========================================
What is bfio? An explanation and roadmap.
=========================================

----------------------------------------
The Web Image Processing Pipeline (WIPP)
----------------------------------------

WIPP is an open source, cloud computing system for processing image data. It was
developed at the National Institute of Standards and Technology (NIST) with the
express mission of producing traceable, reproducible, and scalable image
analysis pipelines. The file format used for the platform is the OME tiled tiff,
a tiff file specification created by the Open Microscopy Foundation as their
standard file format. Although image tiling is not an explicit component of the
OME tiff specification, the tiles permit loading of small, compressed sections
of an image for processing. This capability is especially useful for large
images that cannot be loaded into memory, and increases the speed of loading
small sections of an image.

The ``bfio`` package was created for reading and writing OME tiled tiffs.
Although created for use within the WIPP platform, it can also be used as a 
powerful file standardization tool to convert images into the scalable
OME tiled tiff format.

---------------------------
Why not use other packages?
---------------------------

The ``bfio`` package makes use of a variety of existing packages to perform
image reading and writing. Below is a rational for what separates bfio apart
from the underlying libraries it runs on.

~~~~~~~~~~~~~~~~~
python-bioformats
~~~~~~~~~~~~~~~~~

``python-bioformats``, along with ``python-javabridge``, are packages developed
for CellProfiler to handle image input and output. The basic way it works is
that ``python-javabridge`` is used to start a Java session, and
``python-bioformats`` interacts with the BioFormats library to read and write
images. The BioFormats library is very robust at reading images and parsing
metadata from a wide array of data types.

The upside to the BioFormats OME tiff specification is that it stores metadata
in a scalable file format. The OME tiff specification is especially attractive
for handling large images when using tiled storage, since pixels close to each
other can be stored with compression and retrieved with low memory and
processing requirements.

The downside to BioFormats is that it is slow, limited to loading only ~2GB of
pixels at a time due to the way it indexes files, and prone to memory leaks.
Thus, for big data BioFormats can be a bottleneck.

~~~~~~~~
tifffile
~~~~~~~~

``tifffile`` is an excellent package for reading and writing tiff files,
supporting a broad range of tiff specifications. The downside to ``tifffile``
has been its ability to write tiled tiffs, and it isn't thread safe or
multi-threaded. Further, as of this writing, it isn't possible to load
subsections of an image out of the box. The other deficit of ``tifffile`` is
that while it can provide access to OME metadata, it does so as a raw XML string
and can be difficult to modify.

---------------
The bfio Vision
---------------

What ``bfio`` does is bring the best ``python-bioformats`` and ``tifffile``
together while adding functionality that is necessary for processing large
images. ``bfio`` utilizes BioFormats to read images and process metadata from a
wide variety of file formats using ``python-bioformats``, and uses portions of
``tifffile`` to accelerate reading/writing chunks of larges images that may not
fit into memory. The ``bfio`` modifies the way ``python-bioformats`` accesses
the BioFormats library to overcome some of the indexing issues in Bioformats,
and uses multi-threaded reading of images. For OME Tiffs, ``bfio`` has a special
Python backend (that uses parts of ``tifffile``) that doesn't require Java.
The ``bfio`` tiff reader/writer is thread safe, uses threaded reading/writing,
and is 4-15x faster than ``python-bioformats`` when reading/writing data files.

The general workflow for ``bfio`` is intended to be:

1. Load an image using ``bfio`` with the Java backend
2. Save as an OME tiled tiff
3. Create image processing components that use the ``bfio`` Python backend to
load and save images.

----------------
What comes next?
----------------

For reasons explained below, ``bfio`` will be creating a zarr backend. In
summary, zarr is a more appropriate file specification for handling large data
due to the way data is stored on disk. Currently, there are a number of
challenges to using the OME tiff specification for large, 3d images. So why
still use ``bfio`` if zarr can handle things better? Mainly for metadata and
consistency of zarr file format.

In addition to adding a zarr backend, the following things will be implemented:

1. A more scalable tiled tiff format
2. Better unit testing
3. Backends to read/write pyramid formats (deepzoom, neuroglancer)
4. Better error handling for threaded reading/writing
5. Load image collection with stitching file (on the fly, in memory assembly)
6. Cython accelerated type conversion upon loading

----------------------------------
Challenges For 3D Data in OME Tiff
----------------------------------

One of the deficits of the OME tiff specification is the way that data is
stored, which is not ideal for large volumetric images. Pixel/voxel data in an
image is stored in 2-dimensional tiff pages in the OME tiff specification, which
is problematic for storing 3d data when using compression. The reason this is
problematic is that the information about each pages storage is located at the
beginning of each page, and if the size of a page on disk cannot be known at the
time of writing (because of variable compression rates), then each page must be
written serially. Thus, when writing a z-stack in the OME tiff spec, the entire
first z-slice must be written before the second z-slice can be written. For
volumes that can fit into memory, this isn't an issue, but for large volumes
this may be impossible. 