.. bfio documentation master file, created by
   sphinx-quickstart on Fri Oct 30 11:58:14 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
BioFormats Input/Ouput (bfio) Utility
=====================================

The ``bfio`` utility builds off of multiple existing projects to streamline and
optimize the reading of formats supported by
`Bioformats <https://www.openmicroscopy.org/bio-formats/>`_
. Writing of files is done using the the OME-TIFF file specification, using
compressed image tiling to make the file conform to the
`Web Image Processing Pipeline (WIPP) <https://github.com/usnistgov/wipp>`_
standard image format.

This tool makes extensive usage of
`tifffile <https://pypi.org/project/tifffile/>`_,
`python-bioformats <https://pypi.org/project/python-bioformats/>`_,
and `python-javabridge <https://pypi.org/project/python-javabridge/>`_. The need
for the ``bfio`` utility stems from issues in dealing with large images in
``python-bioformats``, slow loading and saving of files in ``ome.tif`` format,
and difficulty of maintaining OME metadata. Individually, each package has
advantages and disadvantages, but ``bfio`` merges and optimizes the best
qualities of each package and simplifies working with large images.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Examples
   Reference
   BFIO