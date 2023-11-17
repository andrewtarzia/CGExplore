#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for handling bonds.

Author: Andrew Tarzia

"""

import itertools
import logging
from dataclasses import dataclass

import stk
from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def bond_k_unit():
    return openmm.unit.kilojoules_per_mole / openmm.unit.nanometer**2


@dataclass
class Bond:
    atom_names: tuple[str, ...]
    atom_ids: tuple[int, ...]
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity
    atoms: tuple[stk.Atom, ...] | None
    force: str | None


@dataclass
class TargetBond:
    class1: str
    class2: str
    eclass1: str
    eclass2: str
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.class1}{self.class2}, "
            f"{self.eclass1}{self.eclass2}, "
            f"{self.bond_r.in_units_of(openmm.unit.angstrom)}, "
            f"{self.bond_k.in_units_of(bond_k_unit())}, "
            ")"
        )


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
