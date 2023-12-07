#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for handling nobonded interactions.

Author: Andrew Tarzia

"""

import itertools as it
import logging
from dataclasses import dataclass

from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Nonbonded:
    atom_id: int
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

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.bead_class}, "
            f"{self.bead_element}, "
            f"{self.sigma.in_units_of(openmm.unit.angstrom)}, "
            f"{self.epsilon.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            f"{self.force}, "
            ")"
        )


@dataclass
class TargetNonbondedRange:
    bead_class: str
    bead_element: str
    sigmas: tuple[openmm.unit.Quantity]
    epsilons: tuple[openmm.unit.Quantity]
    force: str

    def yield_nonbondeds(self):
        for sigma, epsilon in it.product(self.sigmas, self.epsilons):
            yield TargetNonbonded(
                bead_class=self.bead_class,
                bead_element=self.bead_element,
                epsilon=epsilon,
                sigma=sigma,
                force=self.force,
            )
