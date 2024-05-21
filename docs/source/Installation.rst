Install
=======

The ``bfio`` package and the core dependencies (``numpy``, ``tifffile``, ``imagecodecs``, ``scyjava``) can
be installed using pip:

``pip install bfio``


A conda distribution is also available in the conda-forge channel. To install it in conda environment, use this:

``conda install bfio -c conda-forge``

Java and Bio-Formats
-------------------
``bfio`` can be used without Java, but only the ``python`` and ``zarr``
backends will be usable. This means only files in tiled OME Tiff or OME Zarr format can be
read/written.

In order to use the ``bioformats`` backend, it is necessary to first install the JDK and Maven.
The ``bfio`` package is generally tested with
`JDK 8 <https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html>`_,
but JDK 11 and later also appear to work.
Here are some info on installing Maven on various OS (`Windows <https://phoenixnap.com/kb/install-maven-windows>`_ | `Linux <https://www.digitalocean.com/community/tutorials/install-maven-linux-ubuntu>`_ | `Mac <https://www.digitalocean.com/community/tutorials/install-maven-mac-os>`_).



NOTE: ``Bio-Formats`` is licensed under GPL, while ``bfio`` is licensed under MIT. This may have consequences when packaging any software that uses
``bfio`` as a dependency. During the first invocation of ``bfio``, ``scyjava`` will try to download ``Bio-Formats`` package from the Maven repository. The current version of ``bfio`` uses ``Bio-Formats`` v7.3.0 .