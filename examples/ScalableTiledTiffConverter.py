from bfio import BioReader, BioWriter, LOG4J, JARS
import javabridge, math, requests
from pathlib import Path
from multiprocessing import cpu_count

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
                
    # Number of tiles to process at a time
    # This value squared is the total number of tiles processed at a time
    tile_grid_size = math.ceil(math.sqrt(cpu_count()))

    # Do not change this, the number of pixels to be saved at a time must
    # be a multiple of 1024
    tile_size = tile_grid_size * 1024
    
    # Set up the BioReader
    with BioReader(PATH,backend='java',max_workers=cpu_count()) as br:

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
    
finally:
    # Close the javabridge. Since this is in the finally block, it is always run
    javabridge.kill_vm()
