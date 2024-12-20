# Distributed under the terms of the MIT License.

"""Module for handling nobonded interactions.

Author: Andrew Tarzia

"""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass(frozen=True, slots=True)
class Nonbonded:
    """Class containing term defintion.

    Parameters:
        atom_id:
            ID of bead.

        bead_class:
            Class of bead - usually the first letter of bead type.

        bead_element:
            Element of atom.

        sigma:
            `sigma` quantity to be implemented in a forcefield.

        epsilons:
            `epsilon` quantity to be implemented in a forcefield.

        force:
            Which nonbonded force to use in this term.

    """

    atom_id: int
    bead_class: str
    bead_element: str
    sigma: openmm.unit.Quantity
    epsilon: openmm.unit.Quantity
    force: str


@dataclass(frozen=True, slots=True)
class TargetNonbonded:
    """Defines a target term to search for in a molecule.

    Parameters:
        bead_class:
            Class of bead - usually the first letter of bead type.

        bead_element:
            Element of atom.

        sigma:
            `sigma` quantity to be implemented in a forcefield.

        epsilons:
            `epsilon` quantity to be implemented in a forcefield.

        force:
            Which nonbonded force to use in this term.

    """

    bead_class: str
    bead_element: str
    sigma: openmm.unit.Quantity
    epsilon: openmm.unit.Quantity
    force: str

    def vector_key(self) -> str:
        """Return key for vector defining this target term."""
        return self.bead_class

    def vector(self) -> tuple[float, float]:
        """Return vector defining this target term."""
        return (
            self.sigma.value_in_unit(openmm.unit.angstrom),
            self.epsilon.value_in_unit(openmm.unit.kilojoules_per_mole),
        )

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.bead_class}, "
            f"{self.bead_element}, "
            f"{self.sigma.in_units_of(openmm.unit.angstrom)}, "
            f"{self.epsilon.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            f"{self.force}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class TargetNonbondedRange:
    """Defines a target term and ranges in parameters to search for.

    Parameters:
        bead_class:
            Class of bead - usually the first letter of bead type.

        bead_element:
            Element of atom.

        sigmas:
            Each `sigma` quantity to be implemented in a forcefield
            library.

        epsilons:
            Each `epsilon` quantity to be implemented in a forcefield
            library.

        force:
            Which nonbonded force to use in this term.

    """

    bead_class: str
    bead_element: str
    sigmas: tuple[openmm.unit.Quantity]
    epsilons: tuple[openmm.unit.Quantity]
    force: str

    def yield_nonbondeds(self) -> abc.Iterable[TargetNonbonded]:
        """Find interactions matching target."""
        for sigma, epsilon in it.product(self.sigmas, self.epsilons):
            yield TargetNonbonded(
                bead_class=self.bead_class,
                bead_element=self.bead_element,
                epsilon=epsilon,
                sigma=sigma,
                force=self.force,
            )
