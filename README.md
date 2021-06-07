# **B**io**F**ormats **I**nput/**O**utput utility (bfio v2.1.6)

[![Documentation Status](https://readthedocs.org/projects/bfio/badge/?version=latest)](https://bfio.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bfio)](https://pypi.org/project/filepattern/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/bfio)
![Bower](https://img.shields.io/bower/l/MI)

This tool is a simplified but powerful interface to the
[Bioformats java library](https://www.openmicroscopy.org/bio-formats/).
It makes use of parts of Cell Profilers
[python-bioformats](https://github.com/CellProfiler/python-bioformats)
package to access the Bioformats library. One of the issues with using the
`python-bioformats` package is reading and writing large image planes (>2GB).
The challenge lies in the way Bioformats reads and writes large image planes,
using an `int` value to index the file. To get around this, files can be read or
written in chunks and the classes provided in `bfio` handle this automatically.
The `BioWriter` class in this package only writes files in the `.ome.tif`
format, and automatically sets the tile sizes to 1024.

Docker containers with all necessary components are available (see
**Docker Containers** section).

## Summary

  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Installation

### Setting up Java

**Note:** `bfio` can be used without Java, but only the `python` and `zarr`
backends will be useable. Only files in tiled OME Tiff or OME Zarr format can be
read/written.

In order to use the `Java` backend, it is necessary to first install the JDK.
The `bfio` package is generally tested with
[JDK 8](https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html),
but JDK 11 and later also appear to work.

### Installing bfio

The `bfio` package and the core dependencies (numpy, tifffile, imagecodecs) can
be installed using pip:

`pip install bfio`

Additionally, `bfio` with other dependencies can be installed:

1. `pip install bfio[jpype]` - Adds support for BioFormats/Java
2. `pip install bfio[zarr]` - Adds support for OME Zarr
3. `pip install bfio[all]` - Installs all dependencies.

## Documentation

Documentation and examples are available on
[Read the Docs](https://bfio.readthedocs.io/en/latest/).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

Nick Schaub (nick.schaub@nih.gov, nick.schaub@labshare.org)

## License

This project is licensed under the [MIT License](LICENSE)
Creative Commons License - see the [LICENSE](LICENSE) file for
details

## Acknowledgments

  - Parts of this code were written/modified from existing code found in
    `python-bioformats` and `tifffile`.
