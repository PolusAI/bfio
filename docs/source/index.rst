=====================================
BioFormats Input/Ouput (bfio) Utility
=====================================

The ``bfio`` utility is an easy to use, image input/output utility optimized to
scalably read and write OME TIFF and OME Zarr images as well as all formats supported by
`Bioformats <https://www.openmicroscopy.org/bio-formats/>`_
. Reading of data makes direct usage of BioFormats (when using the `bioformats` install
option) using JPype, giving ``bfio`` the ability to read any of the 150+ formats
supported by BioFormats.

For file outputs, only two file formats are supported: tiled OME TIFF, and OME
Zarr. These two file formats are supported because they are scalable, permitting
tiled writing. This package was created out of a need for speed and scalability
so that plugins could be created for the 
`Web Image Processing Pipeline (WIPP) <https://github.com/usnistgov/wipp>`_.

``bfio`` has a simple to use interface that allows programs to access images
as if they were memory mapped numpy arrays operating on the original data file.
There are a lot of caveats to data reading and writing, but the goal of this
package was to lower the barrier to working with image data that may not fit
into memory.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Examples
   Reference
