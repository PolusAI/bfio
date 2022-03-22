from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read()
    with open("./bfio/VERSION", "w") as fw:
        fw.write(version)

setup(
    name="bfio",
    version=version,
    author="Nick Schaub",
    author_email="nick.schaub@nih.gov",
    description="Simple reading and writing classes for tiled tiffs using Bioformats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://bfio.readthedocs.io/en/latest/",
        "Source": "https://github.com/labshare/bfio",
    },
    entry_points={"napari.plugin": "bfio = bfio.bfio"},
    packages=find_packages(),
    package_data={"bfio": ["jars/*.jar", "jars/*.xml", "VERSION"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: napari",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tifffile>=2021.8.30",
        "imagecodecs>=2021.2.26",
        "numpy>=1.20.1",
        "ome-types>=0.2.10",
        "lxml",  # remove this when upgrading to the next version of ome-types
    ],
    extras_require={
        "jpype": ["jpype1>=1.2.1"],
        "zarr": ["zarr>=2.6.1"],
        "all": ["jpype1>=1.2.1", "zarr>=2.6.1"],
        "dev": ["jpype1>=1.2.1", "zarr>=2.6.1", "requests>=2.26.0"],
    },
)
