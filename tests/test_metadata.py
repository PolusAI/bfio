import logging
import pytest
import javabridge as jutil
import bioformats
import bfio
from pathlib import Path

""" Metadata tests to run on each backend """
def get_dims(reader):
    print()
    for dim in 'xyzct':
        print('image.{} = {}'.format(dim,getattr(reader,dim)))
    for dim in 'xyzct'.upper():
        print('image.{} = {}'.format(dim,getattr(reader,dim)))
    print('image.shape = {}'.format(reader.shape))
    
def get_pixel_size(reader):
    print()
    for dim in 'xyz':
        attribute = 'physical_size_{}'.format(dim)
        print('image.physical_size_{} = {}'.format(dim,
                                                   getattr(reader,attribute)))
    for dim in 'xyz':
        attribute = 'ps_{}'.format(dim)
        print('image.ps_{} = {}'.format(dim,
                                        getattr(reader,attribute)))
        
def get_pixel_info(reader):
    print()
    print('image.samples_per_pixel={}'.format(reader.samples_per_pixel))
    print('image.spp={}'.format(reader.spp))
    print('image.bytes_per_pixel={}'.format(reader.bytes_per_pixel))
    print('image.bpp={}'.format(reader.bpp))
    print('image.dtype={}'.format(reader.dtype))
    
def get_channel_names(reader):
    print()
    print('image.channel_names={}'.format(reader.channel_names))
    print('image.cnames={}'.format(reader.cnames))

""" Test classes (where the testing actually happens) """
class TestVersion():
    
    def test_bfio_version(self):
        print()
        print('__version__ = {}'.format(bfio.__version__))
        assert bfio.__version__ != '0.0.0'
        
    def test_jar_version(self):
        print()
        print('JAR_VERSION = {}'.format(bfio.JAR_VERSION))
        assert bfio.__version__ != None
        
class TestPythonReader():
    
    def test_get_dims(self,python_reader):
        get_dims(python_reader)
        
    def test_get_pixel_size(self,python_reader):
        get_pixel_size(python_reader)
        
    def test_get_pixel_info(self,python_reader):
        get_pixel_info(python_reader)
        
    def test_get_channel_names(self,python_reader):
        get_channel_names(python_reader)
            
class TestJavaReader():
    
    def test_dims(self,java_reader,jvm):
        get_dims(java_reader)
        
    def test_get_pixel_size(self,java_reader,jvm):
        get_pixel_size(java_reader)
        
    def test_get_pixel_info(self,java_reader,jvm):
        get_pixel_info(java_reader)
        
    def test_get_channel_names(self,java_reader,jvm):
        get_channel_names(java_reader)