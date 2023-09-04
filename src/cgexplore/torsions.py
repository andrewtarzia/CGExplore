#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling torsions.

Author: Andrew Tarzia

"""

import logging
import typing
from dataclasses import dataclass
import itertools
import stk
from rdkit.Chem import AllChem as rdkit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Torsion:
    atom_names: tuple[str, ...]
    atom_ids: tuple[int, ...]
    phi0: float
    torsion_k: float
    torsion_n: float


@dataclass
class TargetTorsion:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0: float
    torsion_k: float
    torsion_n: float


@dataclass
class TargetTorsionRange:
    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0s: tuple[float]
    torsion_ks: tuple[float]
    torsion_ns: tuple[float]

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
