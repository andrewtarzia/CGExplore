#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling torsions.

Author: Andrew Tarzia

"""

import itertools
import logging
import typing
from dataclasses import dataclass

import stk
from openmm import openmm
from rdkit.Chem import AllChem as rdkit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Torsion:
    atom_names: tuple[str, ...]
    atom_ids: tuple[int, ...]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int
    force: str | None


@dataclass
class TargetTorsion:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f'{"".join(self.search_string)}, '
            f'{"".join(self.search_estring)}, '
            f"{str(self.measured_atom_ids)}, "
            f"{self.phi0.in_units_of(openmm.unit.degrees)}, "
            f"{self.torsion_k.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            f"{self.torsion_n}, "
            ")"
        )


@dataclass
class TargetTorsionRange:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0s: tuple[openmm.unit.Quantity]
    torsion_ks: tuple[openmm.unit.Quantity]
    torsion_ns: tuple[int]

    def yield_torsions(self):
        for phi0, k, n in itertools.product(
            self.phi0s, self.torsion_ks, self.torsion_ns
        ):
            yield TargetTorsion(
                search_string=self.search_string,
                search_estring=self.search_estring,
                measured_atom_ids=self.measured_atom_ids,
                phi0=phi0,
                torsion_k=k,
                torsion_n=n,
            )


@dataclass
class FoundTorsion:
    atoms: tuple[stk.Atom, ...]
    atom_ids: tuple[int, ...]


def find_torsions(
    molecule: stk.Molecule,
    chain_length: int,
) -> typing.Iterator[FoundTorsion]:
    paths = rdkit.FindAllPathsOfLengthN(
        mol=molecule.to_rdkit_mol(),
        length=chain_length,
        useBonds=False,
        useHs=True,
    )
    for atom_ids in paths:
        atoms = tuple(molecule.get_atoms(atom_ids=[i for i in atom_ids]))
        yield FoundTorsion(
            atoms=atoms,
            atom_ids=tuple(i.get_id() for i in atoms),
        )
