#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG optimizer.

Author: Andrew Tarzia

"""

import logging

import stk

from .forcefield import Forcefield

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class CGOptimizer:
    def __init__(
        self,
        force_field: Forcefield,
    ) -> None:
        self._force_field = force_field
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10

    def optimize(self, molecule: stk.Molecule) -> stk.Molecule:
        raise NotImplementedError()
