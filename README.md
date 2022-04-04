# **B**io**F**ormats **I**nput/**O**utput utility (bfio 2.3.0-dev0)

[![Documentation Status](https://readthedocs.org/projects/bfio/badge/?version=latest)](https://bfio.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bfio)](https://pypi.org/project/filepattern/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/bfio)
![Bower](https://img.shields.io/bower/l/MI)

This tool is a simplified but powerful interface to
[Bioformats](https://www.openmicroscopy.org/bio-formats/)
using jpype for direct access to the library. This tool is designed with
scalable image analysis in mind, with a simple interface to treat any image
like a memory mapped array.

Docker containers with all necessary components are available (see
**Docker Containers** section).

## Summary

- [Installation](#installation)
- [Docker](#docker)
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

## Docker

### labshare/polus-bfio-util:2.3.0-dev0

Ubuntu based container with bfio and all dependencies (including Java).

### labshare/polus-bfio-util:2.3.0-dev0-imagej

Same as above, except comes with ImageJ and PyImageJ.

### labshare/polus-bfio-util:2.3.0-dev0-tensorflow

Tensorflow container with bfio isntalled.

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
    `tifffile`.
