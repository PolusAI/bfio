Install
=======

The ``bfio`` package can be installed with a variety of options designed in
consideration of containerization.

Base Package
------------

The base package can only read/write tiled, OME TIFF images. It can be installed
using:

``pip install bfio``

Java and Bioformats
-------------------

To use Bioformats, it is necessary to install Java 8 or later. Once Java is
installed, ``bfio`` can be installed with support for Java using:

pip install bfio['jpype']

Zarr
----

To use the zarr backend, you need to have Zarr already installed, or you can
install it when installing bfio using:

pip install bfio['jpype']