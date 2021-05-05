from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("VERSION",'r') as fh:
    version = fh.read()
    with open("./bfio/VERSION",'w') as fw:
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
        'Documentation': 'https://bfio.readthedocs.io/en/latest/',
        'Source': 'https://github.com/Nicholas-Schaub/polus-plugins/tree/master/utils/polus-bfio-util'
    },
    packages=find_packages(),
    package_data = {"bfio": ['jars/*.jar','jars/*.properties','VERSION']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ['tifffile==2020.11.18',
                        'imagecodecs>=2021.2.26',
                        'numpy>=1.20.1'],
)