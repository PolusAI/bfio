from __future__ import absolute_import, unicode_literals

import logging
import pathlib

JAR_VERSION = None
JARS = None
LOGBACK = None

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.init")

log_level = logging.WARNING

try:
    with open(pathlib.Path(__file__).parent.joinpath("VERSION"), "r") as fh:
        __version__ = fh.read()
except FileNotFoundError:
    logger.info(
        "Could not find VERSION. "
        + "This is likely due to using a local/cloned version of bfio."
    )
    __version__ = "0.0.0"
logger.info("VERSION = {}".format(__version__))

if pathlib.Path(__file__).parent.joinpath("jars").is_dir():

    try:

        _jars_dir = pathlib.Path(__file__).parent.joinpath("jars")

        JAR_VERSION = "Use bfio.start() to get JAR_VERSION."

        JARS = [str(_jars_dir.joinpath("bioformats_package.jar").absolute())]

        LOGBACK = str(_jars_dir.joinpath("logback.xml").absolute())

    except ModuleNotFoundError:
        JAR_VERSION = None
        JARS = None
        LOGBACK = None
        logger.info(
            "jpype has not been installed. "
            + "Can only use Python backend for reading/writing images."
        )

else:
    logger.info(
        "The bioformats_package.jar is not present."
        + "Can only use Python backend for reading/writing images."
    )

# Must import last after jpype and version logic
from .bfio import BioReader, BioWriter, start  # NOQA: F401, E402
