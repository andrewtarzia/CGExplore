#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling nobonded interactions.

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
class TargetNonbonded:
    search_string: str
    search_estring: str
    sigma: openmm.unit.Quantity
    epsilon: openmm.unit.Quantity


@dataclass
class TargetNonbondedRange:
    search_string: str
    search_estring: str
    sigmas: tuple[openmm.unit.Quantity]
    epsilons: tuple[openmm.unit.Quantity]

    def yield_nonbondeds(self):
        for sigma, epsilon in itertools.product(self.sigmas, self.epsilons):
            yield TargetNonbonded(
                search_string=self.search_string,
                search_estring=self.search_estring,
                epsilon=epsilon,
                sigma=sigma,
            )
