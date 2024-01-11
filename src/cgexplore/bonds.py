#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for handling bonds.

Author: Andrew Tarzia

"""

import itertools as it
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
    funct: int = 0


@dataclass
class TargetBond:
    type1: str
    type2: str
    element1: str
    element2: str
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity
    funct: int = 0

    def vector_key(self) -> str:
        return f"{self.type1}{self.type2}"

    def vector(self) -> tuple[float]:
        return (
            self.bond_r.value_in_unit(openmm.unit.angstrom),
            self.bond_k.value_in_unit(bond_k_unit()),
        )

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}, "
            f"{self.element1}{self.element2}, "
            f"{self.bond_r.in_units_of(openmm.unit.angstrom)}, "
            f"{self.bond_k.in_units_of(bond_k_unit())}, "
            ")"
        )


@dataclass
class TargetBondRange:
    type1: str
    type2: str
    element1: str
    element2: str
    bond_rs: tuple[openmm.unit.Quantity]
    bond_ks: tuple[openmm.unit.Quantity]

    def yield_bonds(self):
        for r, k in it.product(self.bond_rs, self.bond_ks):
            yield TargetBond(
                type1=self.type1,
                type2=self.type2,
                element1=self.element1,
                element2=self.element2,
                bond_k=k,
                bond_r=r,
            )


@dataclass
class TargetPairedBondRange:
    type1s: tuple[str]
    type2s: tuple[str]
    element1s: tuple[str]
    element2s: tuple[str]
    bond_rs: tuple[openmm.unit.Quantity]
    bond_ks: tuple[openmm.unit.Quantity]

    def yield_bonds(self):
        raise NotImplementedError
        for r, k in it.product(self.bond_rs, self.bond_ks):
            for type1, type2, element1, element2 in zip(
                self.type1s,
                self.type2s,
                self.element1s,
                self.element2s,
                strict=True,
            ):
                print(type1, type2, element1, element2)
                yield TargetBond(
                    type1=type1,
                    type2=type2,
                    element1=element1,
                    element2=element2,
                    bond_k=k,
                    bond_r=r,
                )


@dataclass
class TargetMartiniBond:
    type1: str
    type2: str
    element1: str
    element2: str
    funct: int
    bond_r: openmm.unit.Quantity
    bond_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}, "
            f"{self.element1}{self.element2}, "
            f"{self.funct},"
            f"{self.bond_r.in_units_of(openmm.unit.angstrom)}, "
            f"{self.bond_k.in_units_of(bond_k_unit())}, "
            ")"
        )


@dataclass
class MartiniBondRange:
    type1: str
    type2: str
    element1: str
    element2: str
    funct: int
    bond_rs: tuple[openmm.unit.Quantity]
    bond_ks: tuple[openmm.unit.Quantity]

    def yield_bonds(self):
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
