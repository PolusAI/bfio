from bfio import BioReader, BioWriter, LOG4J, JARS
import javabridge
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
javabridge.start_vm(args=["-Dlog4j.configuration=file:{}".format(LOG4J)],
                    class_path=JARS,
                    run_headless=True)

try:
    # Set up the BioReader
    with BioReader(PATH / FILENAME,backend='java') as br, \
        BioWriter(PATH / 'out.ome.tif',metadata=br.metadata,backend='python') as bw:
    
        # Print off some information about the image before loading it
        print('br.shape: {}'.format(br.shape))
        print('br.dtype: {}'.format(br.dtype))
        
        # Read in the original image, then save
        original_image = br[:]
        bw[:] = original_image
    
finally:
    # Close the javabridge. Since this is in the finally block, it is always run
    javabridge.kill_vm()

# Compare the original and saved images using the Python backend
br = BioReader(PATH.joinpath('out.ome.tif'))

new_image = br.read()

br.close()

print('original and saved images are identical: {}'.format(np.array_equal(new_image,original_image)))
