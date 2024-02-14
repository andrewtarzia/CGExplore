# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

from collections import abc

import stk
from rdkit.Chem import AllChem

from .angles import FoundAngle
from .torsions import FoundTorsion


def find_angles(molecule: stk.Molecule) -> abc.Iterator[FoundAngle]:
    """Find angles based on bonds in molecule."""
    paths = AllChem.FindAllPathsOfLengthN(
        mol=molecule.to_rdkit_mol(),
        length=3,
        useBonds=False,
        useHs=True,
    )
    for atom_ids in paths:
        atoms = tuple(molecule.get_atoms(atom_ids=list(atom_ids)))
        yield FoundAngle(
            atoms=atoms,
            atom_ids=tuple(i.get_id() for i in atoms),
        )


def find_torsions(
    molecule: stk.Molecule,
    chain_length: int,
) -> abc.Iterator[FoundTorsion]:
    """Find torsions based on bonds in molecule."""
    paths = AllChem.FindAllPathsOfLengthN(
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
