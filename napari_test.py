import bfio
import napari
import requests
from pathlib import Path
import numpy as np

r = requests.get('https://downloads.openmicroscopy.org/images/Zeiss-CZI/idr0011/Plate1-Blue-A_TS-Stinger/Plate1-Blue-A-12-Scene-3-P3-F2-03.czi')

with open(Path('Plate1-Blue-A-12-Scene-3-P3-F2-03.czi'),'wb') as fw:
    
    fw.write(r.content)
                
# filename = 'IowaFull_z1483.ome.tif'  # 163 GB compressed
filename = 'IowaFull.mrc'  # 163 GB compressed

class Reader:
    
    def __init__(self,file):
        
        self.br = bfio.BioReader(file)
        
        self.indices = [i for i in reversed(range(5))]
        
    def __getitem__(self,keys):
        
        new_keys = [0 for _ in range(5)]
        for i,k in enumerate(keys):
            new_keys[self.indices[i]] = k
            
        keys = tuple(new_keys)
        return self.br[keys]
    
    @property
    def dtype(self):
        return self.br.dtype
    
    @property
    def shape(self):
        shape = tuple(self.br.shape[i] for i in [4,3,2,1,0])
        return shape

# br = Reader('Plate1-Blue-A-12-Scene-3-P3-F2-03.czi')
br = Reader(filename)
print(br.shape)
viewer = napari.view_image([br], rgb=False, contrast_limits=[np.iinfo(br.dtype).min, np.iinfo(br.dtype).max])
napari.run()
    