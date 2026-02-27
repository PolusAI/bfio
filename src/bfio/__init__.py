# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import logging
import pathlib
from typing import Any

JAR_VERSION = None

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.init")

log_level = logging.WARNING

try:
    with open(pathlib.Path(__file__).parent.joinpath("VERSION"), "r") as fh:
        __version__ = fh.read().strip()
except FileNotFoundError:
    logger.info(
        "Could not find VERSION. "
        "This is likely due to using a local/cloned version of bfio."
    )
    __version__ = "0.0.0"

logger.info("VERSION = %s", __version__)

# same public API but with lazy imports
__all__ = ["BioReader", "BioWriter", "start", "__version__", "JAR_VERSION", "log_level"]


def __getattr__(name: str) -> Any:
    if name in ("BioReader", "BioWriter"):
        from .bfio import BioReader, BioWriter  # local import (lazy)

        return BioReader if name == "BioReader" else BioWriter

    if name == "start":
        from .utils import start  # local import (lazy)

        return start

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
