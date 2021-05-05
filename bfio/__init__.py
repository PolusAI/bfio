from __future__ import absolute_import, unicode_literals
import pathlib, logging

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("bfio.init")

log_level = logging.WARNING

try:
    with open(pathlib.Path(__file__).parent.joinpath("VERSION"),'r') as fh:
        __version__ = fh.read()
except:
    logger.info('Could not find VERSION. This is likely due to using a local/cloned version of bfio.')
    __version__ = "0.0.0"
logger.info('VERSION = {}'.format(__version__))

if pathlib.Path(__file__).parent.joinpath('jars').is_dir():

    try:
        import jpype

        _jars_dir = pathlib.Path(__file__).parent.joinpath('jars')

        JAR_VERSION = "Use bfio.start() to get JAR_VERSION."

        JARS = [str(_jars_dir.joinpath('loci_tools.jar').absolute())]

        LOG4J = str(_jars_dir.joinpath('log4j.properties').absolute())
        
    except ModuleNotFoundError:
        JAR_VERSION = None
        JARS = None
        LOG4J = None
        logger.info('jpype has not been installed. Can only use Python backend for reading/writing images.')

else:
    JAR_VERSION = None
    JARS = None
    LOG4J = None
    logger.info('The loci_tools.jar is not present. Can only use Python backend for reading/writing images.')

from .bfio import BioReader, BioWriter, start
