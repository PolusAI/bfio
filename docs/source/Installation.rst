Install
=======

The ``bfio`` package can be installed with a variety of options designed in
consideration of containerization.

Base Package
------------

The base package can only read/write tiled, OME TIFF images. It can be installed using:

``pip install bfio``

To install all reading and writing options, install with all dependencies:

``pip install bfio[all]``

NOTE: See the Java and Bioformats note about licensing considerations when installing
using the `bfio[all]` option.

Java and Bioformats
-------------------

To use Bioformats, it is necessary to install Java 8 or later. Once Java is
installed, ``bfio`` can be installed with support for Java using:

pip install bfio['bioformats']

NOTE: The `bioformats_jar` package and BioFormats are licensed under GPL, while `bfio`
is licensed under MIT. This may have consequences when packaging any software that uses
`bfio` as a dependency when this option is used.

Zarr
----

To use the zarr backend, you need to have Zarr already installed, or you can
install it when installing bfio using:

pip install bfio['zarr']
