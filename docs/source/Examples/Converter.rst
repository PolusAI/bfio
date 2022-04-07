=============================
Create a Tiled Tiff Converter
=============================

------------
Introduction
------------

The bfio package is designed to make it easy to process arbitrarily sized images
in a fast, scalable way. The two core classes, :doc:`/Reference/BioReader` and
:doc:`/Reference/BioWriter`, can use one of three different backends depending
on the file type that will be read. Usually, the proper backend will be selected
when opening a file, but periodically it will not select the proper backend so
it is useful to mention them here:

1. ``backend="python"`` can only be used to read/write OME tiled tiff images.
   Tiled tiff is the preferred file format for reading/writing arbitrarily sized
   images.
2. ``backend="bioformats"`` can be used to read any
   `any format supported by Bioformats <https://docs.openmicroscopy.org/bio-formats/6.1.0/supported-formats.html>`_.
   The BioWriter with java backend will only save images as OME tiled tiff.
3. ``backend="zarr"`` can be used to read/write a subset of Zarr files following
   the `OME Zarr spec. <https://ngff.openmicroscopy.org/latest/>`_.

The advantage to using the ``python`` and ``zarr`` backends are speed and
scalability at the expense of a rigid file structure, while the ``bioformats`` backend
provides broad access to a wide array of file types but is considerably slower.

In this example, the basic useage of two core classes are demonstrated by
creating a "universal" converter from any Bioformats supported file format to
OME tiled tiff.

---------------
Getting Started
---------------

~~~~~~~~~~~~~~~~~~~~
Install Dependencies
~~~~~~~~~~~~~~~~~~~~

To run this example, a few dependencies are required. First, ensure Java is
installed. Then install ``bfio`` using:

``pip install bfio['bioformats']``

.. note::

    JPype and Bioformats require Java 8 or later. ``bfio`` has been tested
    using Java 8.

.. note::

    Bioformats is licensed under GPL. If you plan to package ``bfio`` using this option,
    make sure to consider the licensing ramifications.

~~~~~~~~~~~~~~~~~~~~~
Download a Test Image
~~~~~~~~~~~~~~~~~~~~~

Download a test image from the Web Image Processing Pipeline (WIPP) github set
of test images.

.. code-block:: python

    from pathlib import Path
    import requests

    """ Get an example image """
    # Set up the directories
    PATH = Path("data")
    PATH.mkdir(parents=True, exist_ok=True)

    # Download the data if it doesn't exist
    URL = "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/"
    FILENAME = "img_r001_c001.ome.tif"
    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)

---------------------------------
Reading Images with the BioReader
---------------------------------

The first step in loading an image with the BioReader is to initialize a
BioReader object. The BioReader tries to predict what backend to use based on
the file extension, and it will throw an error if it predicts incorrectly. In
this case, the test file downloaded from the WIPP repository is not in OME tiled
tiff format even though it has the ``.ome.tif`` extension, meaning it will try
to use the Python backend:

.. code-block:: python

    from bfio import BioReader

    br = BioReader(PATH / FILENAME)

The output should be something like this::
    
    TypeError: img_r001_c001.ome.tif is not a tiled tiff. The python backend of
    the BioReader only supports OME tiled tiffs. Use the bioformats backend to load
    this image.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually Specifying a Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the error message above, we need to use the ``bioformats`` backend. To do
this, just use the keyword argument ``backend=bioformats`` when initializing the
BioReader.

.. code-block:: python

    # Set up the BioReader
    br = BioReader(PATH / FILENAME,backend='bioformats')
    
    # Print off some information about the image before loading it
    print('br.shape: {}'.format(br.shape))
    print('br.dtype: {}'.format(br.dtype))
    
    br.close()

Behind the scenes, what happens is that JPype starts Java and loads the
Bioformats jar. If Java is not installed, the above code will raise an error.

.. Note::

    Currently, the mechanism for starting Java is likely not thread safe.

~~~~~~~~~~~~~~~~~~~
Using the BioReader
~~~~~~~~~~~~~~~~~~~

In the above code, a ``BioReader`` object is initialized, the shape and data
type is printed, and then the BioReader object is closed. The closing of the
``BioReader`` object is necessary to ensure that the Java object is cleaned up
properly. To ensure that this happens, it is recommended to put image access 
into a ``with`` block, which will automatically perform file cleanup.

.. code-block:: python

    from bfio import BioReader

    # Initialize the BioReader inside a ``with`` block to handle file cleanup
    with BioReader(PATH / FILENAME,backend='bioformats') as br:

        # Print off some information about the image before loading it
        print('br.shape: {}'.format(br.shape))
        print('br.dtype: {}'.format(br.dtype))

To read an entire image, use the :attr:`~bfio.bfio.BioReader.read` method
without any arguments.

.. code-block:: python

    I = br.read()

Alternatively, the
:attr:`~bfio.bfio.BioReader.X`,
:attr:`~bfio.bfio.BioReader.Y`, 
:attr:`~bfio.bfio.BioReader.Z`, 
:attr:`~bfio.bfio.BioReader.C`, and 
:attr:`~bfio.bfio.BioReader.T` values can be specified to load only a subsection
of the image. If the BioReader is reading from an OME tiled tiff, then the file
reading should be faster and require less memory than other formats. This has to
do with how data is stored in the OME tiled tiff.

For the current file, to load only the first 100x100 pixels:

.. code-block:: python

    I = br.read(X=[0,100],Y=[0,100])

The above code will return a 5-dimensional numpy array with
``shape=(100,100,1,1,1)``. If this file had multiple z-slices, channels, or
timepoint information stored in it, then the first 100x100 pixels in every
z-slice, channel, and timepoint would all be loaded since Z, C, and T were not
included as keyword arguments.

To make it easier to load data, data can be fetch from a file using NumPy like
indexing. However, there are some caveats. Step sizes in slices are ignored for
the first three indices. Thus, the following three lines of code will load data
exactly the same as the above line using ``read`` to load the first 100 rows and
columns of pixels:

.. code-block:: python

    I = br[0:100,0:100,:,:,:]
    I = br[:100,:100,...]
    I = br[:100:2,:100:2]

    print(I.shape) # Should return (100,100)

.. note::

    When using NumPy like indexing, trailing dimensions with size=1 are squeezed. So,
    while ``read`` will always return a 5-dimensional array, the NumPy indexing in this
    case will return a 2-dimensional array since there are no Z, C, or T dimensions.

---------------------------------
Writing Images With the BioWriter
---------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~
Initializing the BioWriter
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initializing the :doc:`/Reference/BioWriter` requires a little more thought than
the ``BioReader`` because the properties of the file have to be set prior to
writing any data. In many cases, starting the BioWriter with the same metadata
as the BioReader will get you most of the way there.

.. code-block:: python

    from bfio import BioWriter

    bw = BioWriter(PATH / 'out.ome.tif',metadata=br.metadata)

The above code copies all the metadata from a BioReader object to the
BioWriter object. If the data type needs to be changed for the file, simply
set the object property.

.. code-block:: python

    bw.dtype = np.uint8 # Must be a numpy data type
    bw.X = 1000 # set the image width
    bw.Y = 500  # set the image height
    bw.channel_names = ['PI','phalloidin','DAPI'] # if your image has three channels, name each of them

For more information on the settable properties, see the
:doc:`/Reference/BioWriter` documentation.

~~~~~~~~~~~~~~~~~
Writing the Image
~~~~~~~~~~~~~~~~~

As with the BioReader, the BioWriter needs to be properly closed using the
:attr:`~bfio.bfio.BioWriter.close` method. Closing the BioWriter finalizes the
file, and if code exits without a file being close then the image may not open
properly. To help prevent this scenario, use a ``with`` block.

.. code-block:: python

    with BioWriter(PATH / 'out.ome.tif',metadata=br.metadata) as bw:

        original_image = br[:]
        bw.write(original_image)

This code reads an image and savess it as an OME tiled tiff!

As with the BioReader, it is possible to use numpy-like indexing. An alternative
to the above code block would be:

.. code-block:: python

    with BioWriter(PATH / 'out.ome.tif',metadata=br.metadata) as bw:

        bw[:] = br[:]

.. note::

    After the first ``write`` call, most BioWriter attributes become 
    :attr:`~bfio.bfio.BioWriter.read_only`.

--------------------------------------------
An Efficient, Scalable, Tiled Tiff Converter
--------------------------------------------

In the above example, the demo image was relatively small, so opening the entire
image and saving it was trivial. However, the ``bfio`` classes can be used to
to convert an arbitrarily large image on a resource constrained system. This is
done by reading/writing images in subsections and controlling the number of
threads used for processing. Both the BioReader and BioWriter use multiple
threads to read/write data, one thread per tile on individual tiles. By default,
the number of threads is half the number of detected CPU cores, and this can be
changed when a BioReader or BioWriter object is created by using the
``max_workers`` keyword argument.

To get started, let's transform the previous examples into something more
scalable. Something more scalable will read in a small part of one image, and
save it into the tiled tiff format. 

.. note::

    The BioWriter always saves images in 1024x1024 tiles. So, it is important to
    save images in multiples of 1024 (height or width) in order for the image to
    save correctly. In the future, the tiled tiff tile size may become a user
    defined parameter, but for now the WIPP OME tiled tiff standard of 1024x1024
    tile size is used exclusively.

.. code-block:: python
    
    # Number of tiles to process at a time
    # This value squared is the total number of tiles processed at a time
    tile_grid_size = 1

    # Do not change this, the number of pixels to be saved at a time must
    # be a multiple of 1024
    tile_size = tile_grid_size * 1024

    with BioReader(PATH / 'file.czi',backend='bioformats') as br, \
        BioWriter(PATH / 'out.ome.tif',backend='bioformats',metadata=br.metadata) as bw:
    
        # Loop through timepoints
        for t in range(br.T):

            # Loop through channels
            for c in range(br.C):

                # Loop through z-slices
                for z in range(br.Z):

                    # Loop across the length of the image
                    for y in range(0,br.Y,tile_size):
                        y_max = min([br.Y,y+tile_size])

                        # Loop across the depth of the image
                        for x in range(0,br.X,tile_size):
                            x_max = min([br.X,x+tile_size])
                            
                            bw[y:y_max,x:x_max,z:z+1,c,t] = br[y:y_max,x:x_max,z:z+1,c,t]


The above code has a lot of for loops. What makes the above code more scalable
than just a simple piece of code like ``bw[:] = br[:]``? The for loops and the
``tile_size`` variable make it so that only a small portion of the image is
loaded into memory. In the above code, ``tile_grid_size = 1``, meaning that
individual tiles are being stored one by one, which is the most memory efficient
way of converting to tiled tiff.

One thing to note in the above example is that both the BioReader and BioWriter
are using the Java backend. This ensures a direct, 1-to-1 file conversion can
take place. While the Python backend for both the BioReader and BioWriter can read OME
TIFF files with any number of XYZCT dimensions, the WIPP platform expects each file to
only contain XYZ data. To make the above tiled tiff converter export WIPP compliant
files, the code should be changed as follows:

.. code-block:: python
    
    # Number of tiles to process at a time
    # This value squared is the total number of tiles processed at a time
    tile_grid_size = 1

    # Do not change this, the number of pixels to be saved at a time must
    # be a multiple of 1024
    tile_size = tile_grid_size * 1024

    with BioReader(PATH / 'file.czi',backend='bioformats') as br:
    
        # Loop through timepoints
        for t in range(br.T):

            # Loop through channels
            for c in range(br.C):
            
                with BioWriter(PATH / 'out_c{c:03d}_t{t:03d}.ome.tif',
                               backend='bioformats',
                               metadata=br.metadata) as bw:
                    bw.C = 1
                    bw.T = 1

                    # Loop through z-slices
                    for z in range(br.Z):

                        # Loop across the length of the image
                        for y in range(0,br.Y,tile_size):
                            y_max = min([br.Y,y+tile_size])

                            # Loop across the depth of the image
                            for x in range(0,br.X,tile_size):
                                x_max = min([br.X,x+tile_size])
                                
                                bw[y:y_max,x:x_max,z:z+1,0,0] = br[y:y_max,x:x_max,z:z+1,c,t]

---------------------
Complete Example Code
---------------------

~~~~~~~~~~~~~~~~~~~~~~
Self Contained Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bfio import BioReader, BioWriter
    from pathlib import Path
    import requests
    import numpy as np

    """ Get an example image """
    # Set up the directories
    PATH = Path("data")
    PATH.mkdir(parents=True, exist_ok=True)

    # Download the data if it doesn't exist
    URL = "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/"
    FILENAME = "img_r001_c001.ome.tif"
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    """ Convert the tif to tiled tiff """
    # Set up the BioReader
    with BioReader(PATH / FILENAME,backend='bioformats') as br, \
        BioWriter(PATH / 'out.ome.tif',metadata=br.metadata,backend='python') as bw:
    
        # Print off some information about the image before loading it
        print('br.shape: {}'.format(br.shape))
        print('br.dtype: {}'.format(br.dtype))
        
        # Read in the original image, then save
        original_image = br[:]
        bw[:] = original_image

    # Compare the original and saved images using the Python backend
    br = BioReader(PATH.joinpath('out.ome.tif'))

    new_image = br.read()

    br.close()

    print('original and saved images are identical: {}'.format(np.array_equal(new_image,original_image)))

~~~~~~~~~~~~~~~~~~~
Scalable Tiled Tiff
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bfio import BioReader, BioWriter
    import math
    from pathlib import Path
    from multiprocessing import cpu_count

    """ Define the path to the file to convert """
    # Set up the directories
    PATH = Path("path/to/file").joinpath('file.tif')


    """ Convert the tif to tiled tiff """
    # Number of tiles to process at a time
    # This value squared is the total number of tiles processed at a time
    tile_grid_size = math.ceil(math.sqrt(cpu_count()))

    # Do not change this, the number of pixels to be saved at a time must
    # be a multiple of 1024
    tile_size = tile_grid_size * 1024
    
    # Set up the BioReader
    with BioReader(PATH,backend='bioformats',max_workers=cpu_count()) as br:

        # Loop through timepoints
        for t in range(br.T):

            # Loop through channels
            for c in range(br.C):
            
                with BioWriter(PATH.with_name(f'out_c{c:03}_t{t:03}.ome.tif'),
                            backend='python',
                            metadata=br.metadata,
                            max_workers = cpu_count()) as bw:

                    # Loop through z-slices
                    for z in range(br.Z):

                        # Loop across the length of the image
                        for y in range(0,br.Y,tile_size):
                            y_max = min([br.Y,y+tile_size])

                            # Loop across the depth of the image
                            for x in range(0,br.X,tile_size):
                                x_max = min([br.X,x+tile_size])
                                
                                bw[y:y_max,x:x_max,z:z+1,0,0] = br[y:y_max,x:x_max,z:z+1,c,t]
