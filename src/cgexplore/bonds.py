#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling bonds.

Author: Andrew Tarzia

"""

import itertools
import logging
from openmm import openmm
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class TargetBond:
    class1: str
    class2: str
    eclass1: str
    eclass2: str
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity


@dataclass
class TargetBondRange:
    class1: str
    class2: str
    eclass1: str
    eclass2: str
    bond_rs: tuple[openmm.unit.Quantity]
    bond_ks: tuple[openmm.unit.Quantity]

    def yield_bonds(self):
        for r, k in itertools.product(self.bond_rs, self.bond_ks):
            yield TargetBond(
                class1=self.class1,
                class2=self.class2,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                bond_k=k,
                bond_r=r,
            )
