#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By: Lucas Ribeiro Borges
# ---------------------------------------------------------------------------
""" Concrete reader class module for nearest neighbor datasets"""

import logging
from .BinaryFileReader import BinaryFileReader


class BvecsReader(BinaryFileReader):
    """
    Concrete implementation of abstract Reader class for bytes binary data.

    http://corpus-texmex.irisa.fr/
    The vectors are stored in raw little endian.
    Each vector takes 4+d bytes for .bvecs formats, where d is the dimensionality of the vector, as shown below.

    field       field type          description
    d           int                 the vector dimension
    components  (unsigned char)*d   the vector components
    """

    def __init__(self, filePath: str) -> None:
        """
        The constructor for concrete Reader class for bytes binary data.

        Parameters:
            filepath: str
                Path of file to be read.
        """
        super().__init__(filePath)

        logging.debug(f"BvecsReader created for {filePath}")

    @property
    def dtypeStr(self):
        """
        Property describing binary data on the file.

        <i4 - 32-bit signed integer in little endian

        ({dim},)<B - array of {dim} unsigned bytes in little endian
        """
        return "<i4, ({dim},)<B"

