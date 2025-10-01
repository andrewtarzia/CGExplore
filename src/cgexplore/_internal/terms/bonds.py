# Distributed under the terms of the MIT License.

"""Module for handling bonds.

Author: Andrew Tarzia

"""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

import stk
from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_bond_k_unit = openmm.unit.kilojoules_per_mole / openmm.unit.nanometer**2


@dataclass(frozen=True, slots=True)
class Bond:
    """Class containing bond defintion used in Forcefield.

    Parameters:
        atom_names:
            Element and ID+1 of atoms in bond.

        atom_ids:
            ID of atoms in bond.

        bond_r:
            `r` quantity of bond force.

        bond_k:
            `k` quantity of bond force.

        atoms:
            `stk.Atom` instances of the atoms in bond.

        force:
            The force to apply to a bond, usually "HarmonicBondForce".

        funct:
            For some forcefields (e.g., Martini), this term can change the
            force function. For harmonic bonds, set to `0`.

    """

    atom_names: abc.Sequence[str]
    atom_ids: abc.Sequence[int]
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity
    atoms: tuple[stk.Atom, ...] | None
    force: str | None
    funct: int = 0


@dataclass(frozen=True, slots=True)
class TargetBond:
    """Defines a target term to search for in a molecule.

    Parameters:
        type1:
            Atom/bead type of atom.

        type2:
            Atom/bead type of atom.

        element1:
            Element string of atom.

        element2:
            Element string of atom.

        bond_r:
            `r` quantity of bond force.

        bond_k:
            `k` quantity of bond force.

        funct:
            For some forcefields (e.g., Martini), this term can change the
            force function. For harmonic bonds, set to `0`.

    """

    type1: str
    type2: str
    element1: str
    element2: str
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity
    funct: int = 0

    def vector_key(self) -> str:
        """Return key for vector defining this target term."""
        return f"{self.type1}{self.type2}"

    def vector(self) -> tuple[float, float]:
        """Return vector defining this target term."""
        return (
            self.bond_r.value_in_unit(openmm.unit.angstrom),
            self.bond_k.value_in_unit(_bond_k_unit),
        )

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}, "
            f"{self.element1}{self.element2}, "
            f"{self.bond_r.in_units_of(openmm.unit.angstrom)}, "
            f"{self.bond_k.in_units_of(_bond_k_unit)}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class TargetBondRange:
    """Defines a target term and ranges in parameters to search for.

    Parameters:
        type1:
            Atom/bead type of atom.

        type2:
            Atom/bead type of atom.

        element1:
            Element string of atom.

        element2:
            Element string of atom.

        bond_rs:
            Each `r` quantity of bond force to be implemented in a forcefield
            library.

        bond_ks:
            Each `k` quantity of bond force to be implemented in a forcefield
            library.

    """

    type1: str
    type2: str
    element1: str
    element2: str
    bond_rs: abc.Sequence[openmm.unit.Quantity]
    bond_ks: abc.Sequence[openmm.unit.Quantity]

    def yield_bonds(self) -> abc.Iterable[TargetBond]:
        """Find interactions matching target."""
        for r, k in it.product(self.bond_rs, self.bond_ks):
            yield TargetBond(
                type1=self.type1,
                type2=self.type2,
                element1=self.element1,
                element2=self.element2,
                bond_k=k,
                bond_r=r,
            )


@dataclass(frozen=True, slots=True)
class TargetMartiniBond:
    """Defines a target angle to search for in a molecule."""

    type1: str
    type2: str
    element1: str
    element2: str
    funct: int
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}, "
            f"{self.element1}{self.element2}, "
            f"{self.funct},"
            f"{self.bond_r.in_units_of(openmm.unit.angstrom)}, "
            f"{self.bond_k.in_units_of(_bond_k_unit)}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class MartiniBondRange:
    """Defines a target bond and ranges in parameters to search for."""

    type1: str
    type2: str
    element1: str
    element2: str
    funct: int
    bond_rs: abc.Sequence[openmm.unit.Quantity]
    bond_ks: abc.Sequence[openmm.unit.Quantity]

    def yield_bonds(self) -> abc.Iterable[TargetMartiniBond]:
        """Find bonds matching target."""
        for r, k in it.product(self.bond_rs, self.bond_ks):
            yield TargetMartiniBond(
                type1=self.type1,
                type2=self.type2,
                element1=self.element1,
                element2=self.element2,
                funct=self.funct,
                bond_k=k,
                bond_r=r,
            )
