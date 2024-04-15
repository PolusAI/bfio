# -*- coding: utf-8 -*-
# import core packages
import logging
from typing import Dict


# Third party packages
import ome_types
from xml.etree import ElementTree as ET


from bfiocpp import TSTiffReader, Seq
import bfio.base_classes
from bfio.utils import clean_ome_xml_for_known_issues


class TsOmeTiffReader(bfio.base_classes.TSAbstractReader):
    logger = logging.getLogger("bfio.backends.TsOmeTiffReader")

    _rdr = None
    _offsets_bytes = None
    _STATE_DICT = ["_metadata", "frontend"]

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug("__init__(): Initializing _rdr (TSTiffReader)...")
        self._rdr = TSTiffReader(str(self.frontend._file_path))
        self.X = self._rdr._X
        self.Y = self._rdr._Y
        self.Z = self._rdr._Z
        self.C = self._rdr._C
        self.T = self._rdr._T

        # do test for strip mages

        # do test for interleaved images

        # do test for dimension order

    def __getstate__(self) -> Dict:
        state_dict = {n: getattr(self, n) for n in self._STATE_DICT}
        state_dict.update({"file_path": self.frontend._file_path})

        return state_dict

    def __setstate__(self, state) -> None:
        for k, v in state.items():
            if k == "file_path":
                pass
            else:
                setattr(self, k, v)

    def read_metadata(self):

        self.logger.debug("read_metadata(): Reading metadata...")
        if self._metadata is None:
            try:
                self._metadata = ome_types.from_xml(
                    self._rdr.ome_metadata(), validate=False
                )
            except (ET.ParseError, ValueError):
                if self.frontend.clean_metadata:
                    cleaned = clean_ome_xml_for_known_issues(self._rdr.ome_metadata())
                    self._metadata = ome_types.from_xml(cleaned, validate=False)
                    self.logger.warning(
                        "read_metadata(): OME XML required reformatting."
                    )
                else:
                    raise

        return self._metadata

    def read_image(self, X, Y, Z, C, T):

        cols = Seq(X[0], X[-1] - 1, 1)
        rows = Seq(Y[0], Y[-1] - 1, 1)
        layers = Seq(Z[0], Z[-1] - 1, 1)
        if len(C) == 1:
            channels = Seq(C[0], C[0], 1)
        else:
            channels = Seq(C[0], C[-1], 1)
        if len(T) == 1:
            tsteps = Seq(T[0], T[0], 1)
        else:
            tsteps = Seq(T[0], T[-1], 1)

        return self._rdr.data(rows, cols, layers, channels, tsteps)

    def close(self):
        pass

    def __del__(self):
        self.close()
