# **B**io**F**ormats **I**nput/**O**utput utility (bfio 2.4.3-dev0)

[![Documentation Status](https://readthedocs.org/projects/bfio/badge/?version=latest)](https://bfio.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/bfio)](https://pypi.org/project/bfio/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/bfio)
[![Conda](https://img.shields.io/conda/v/conda-forge/bfio)](https://anaconda.org/conda-forge/bfio)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/bfio?label=Conda%20downloads)](https://anaconda.org/conda-forge/bfio)
![Bower](https://img.shields.io/bower/l/MI)

This tool is a simplified but powerful interface to
[Bio-Formats](https://www.openmicroscopy.org/bio-formats/)
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
backends will be usable. This means only files in tiled OME Tiff or OME Zarr format can be
read/written.

In order to use the `bioformats` backend, it is necessary to first install the JDK and Maven.
The `bfio` package is generally tested with
[JDK 8](https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html),
but JDK 11 and later also appear to work.
Here are some info on installing Maven on various OS ([Windows](https://phoenixnap.com/kb/install-maven-windows) | [Linux](https://www.digitalocean.com/community/tutorials/install-maven-linux-ubuntu) | [Mac](https://www.digitalocean.com/community/tutorials/install-maven-mac-os))

### Installing bfio

The `bfio` package and the core dependencies (numpy, tifffile, imagecodecs, scyjava) can
be installed using pip:

`pip install bfio`

## Docker

### polusai/bfio:2.4.3-dev0

Ubuntu based container with bfio and all dependencies (including Java).

### polusai/bfio:2.4.3-dev0-imagej

Same as above, except comes with ImageJ and PyImageJ.

### polusai/bfio:2.4.3-dev0-tensorflow

Tensorflow container with bfio installed.

## Documentation

Documentation and examples are available on
[Read the Docs](https://bfio.readthedocs.io/en/latest/).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

Nick Schaub (nick.schaub@nih.gov, nick.schaub@axleinfo.com)
Sameeul B Samee (sameeul.samee@axleinfo.com)

## License

This project is licensed under the [MIT License](LICENSE)
Creative Commons License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Parts of this code were written/modified from existing code found in
    `tifffile`.
