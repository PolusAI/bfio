import time

def simple_read(reader):
    print()
    print('Using reading: {}'.format(reader))
    image_read = reader.read(X=[0,1024],Y=[0,1024])
    image_get = reader[0:1024,0:1024]
    
    print('image.shape: {}'.format(reader.shape))
    print('image.dtype: {}'.format(reader.dtype))
    print('image.read(X=[0,1024],Y=[0,1024]): ({},{})'.format(image_read.shape,image_read.dtype))
    print('image[0:1024,0:1024]: ({},{})'.format(image_get.shape,image_get.dtype))
    
def full_read(reader):
    print()
    
    start = time.time()
    for _ in range(10):
        image_read = reader.read()
    
    print('Finished loading the test image 10 times in {:.1f}ms.'.format((time.time() - start)*100))

class TestPythonReader():
    
    def test_read(self,python_reader):
        simple_read(python_reader)
        
    def test_time_read(self,python_reader):
        full_read(python_reader)
        
class TestJavaReader():
    
    def test_read(self,java_reader,jvm):
        simple_read(java_reader)
        
    def test_time_read(self,java_reader):
        full_read(java_reader)
