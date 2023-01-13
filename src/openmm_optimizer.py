#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG Gulp optimizer.

Author: Andrew Tarzia

"""

import os
import re
import logging
from rdkit.Chem import AllChem as rdkit

from optimizer import CGOptimizer


class MDEmptyTrajcetoryError(Exception):
    ...


class CGOMMOptimizer(CGOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        param_pool,
        custom_torsion_set,
        bonds,
        angles,
        torsions,
        vdw,
        max_cycles=500,
        conjugate_gradient=False,
    ):
        super().__init__(
            fileprefix,
            output_dir,
            param_pool,
            bonds,
            angles,
            torsions,
            vdw,
        )
        raise NotImplementedError()
        self._custom_torsion_set = custom_torsion_set

    def optimize(self, molecule):
        raise NotImplementedError()
