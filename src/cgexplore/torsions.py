#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for handling torsions.

Author: Andrew Tarzia

"""

import itertools as it
import logging
from collections import abc
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
    funct: int = 0


@dataclass
class TargetTorsion:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int
    funct: int = 0

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f'{"".join(self.search_string)}, '
            f'{"".join(self.search_estring)}, '
            f"{self.measured_atom_ids!s}, "
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
        for phi0, k, n in it.product(
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
) -> abc.Iterator[FoundTorsion]:
    paths = rdkit.FindAllPathsOfLengthN(
        mol=molecule.to_rdkit_mol(),
        length=chain_length,
        useBonds=False,
        useHs=True,
    )
    for atom_ids in paths:
        atoms = tuple(molecule.get_atoms(atom_ids=list(atom_ids)))
        yield FoundTorsion(
            atoms=atoms,
            atom_ids=tuple(i.get_id() for i in atoms),
        )


@dataclass
class TargetMartiniTorsion:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int
    funct: int

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f'{"".join(self.search_string)}, '
            f'{"".join(self.search_estring)}, '
            f"{self.measured_atom_ids!s}, "
            f"{self.funct},"
            f"{self.phi0.in_units_of(openmm.unit.degrees)}, "
            f"{self.torsion_k.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            f"{self.torsion_n}, "
            ")"
        )


@dataclass
class MartiniTorsionRange:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0s: tuple[openmm.unit.Quantity]
    torsion_ks: tuple[openmm.unit.Quantity]
    torsion_ns: tuple[int]
    funct: int

    def yield_torsions(self):
        raise NotImplementedError("handle torsions")
        for phi0, k, n in it.product(
            self.phi0s, self.torsion_ks, self.torsion_ns
        ):
            yield TargetMartiniTorsion(
                search_string=self.search_string,
                search_estring=self.search_estring,
                measured_atom_ids=self.measured_atom_ids,
                phi0=phi0,
                torsion_k=k,
                torsion_n=n,
                funct=self.funct,
            )
