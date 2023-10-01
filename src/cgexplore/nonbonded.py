#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling nobonded interactions.

Author: Andrew Tarzia

"""

import itertools
import logging
from dataclasses import dataclass

from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Nonbonded:
    bead_class: str
    bead_element: str
    sigma: openmm.unit.Quantity
    epsilon: openmm.unit.Quantity
    force: str


@dataclass
class TargetNonbonded:
    bead_class: str
    bead_element: str
    sigma: openmm.unit.Quantity
    epsilon: openmm.unit.Quantity
    force: str


@dataclass
class TargetNonbondedRange:
    bead_class: str
    bead_element: str
    sigmas: tuple[openmm.unit.Quantity]
    epsilons: tuple[openmm.unit.Quantity]
    force: str

    def yield_nonbondeds(self):
        for sigma, epsilon in itertools.product(self.sigmas, self.epsilons):
            yield TargetNonbonded(
                bead_class=self.bead_class,
                bead_element=self.bead_element,
                epsilon=epsilon,
                sigma=sigma,
                force=self.force,
            )
