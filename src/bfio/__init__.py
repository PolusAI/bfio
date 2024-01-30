# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import logging
import pathlib

JAR_VERSION = None

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


from .bfio import BioReader, BioWriter, start  # NOQA: F401, E402
