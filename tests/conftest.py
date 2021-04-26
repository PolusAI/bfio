import pytest
import javabridge as jutil
from pathlib import Path
import bfio

""" Image path to test """
image_path = Path('Path/To/Test/Image')

""" Fixtures """ 
@pytest.fixture(scope="class")
def java_reader():
    return bfio.BioReader(image_path,backend='java')

@pytest.fixture
def python_reader(scope="class"):
    return bfio.BioReader(image_path,backend='python')

@pytest.fixture(scope="session")
def jvm():
    print()
    print('Starting javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(bfio.LOG4J)],class_path=bfio.JARS)
    yield
    print()
    print('Closing javabridge...')
    jutil.kill_vm()