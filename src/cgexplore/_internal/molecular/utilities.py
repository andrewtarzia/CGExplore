# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging

import numpy as np
import spindry as spd
import stk

from .beads import string_to_atom_number

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_rotation(radians: float) -> np.ndarray:
    """Get rotation of radians."""
    c, s = np.cos(radians), np.sin(radians)
    return np.array(((c, -s), (s, c)))


def vnorm_r(v: np.ndarray, distance: float) -> np.ndarray:
    """Normalise rotation."""
    return v / np.linalg.norm(v) * distance


def spd_to_stk(supramolecule: spd.SupraMolecule) -> stk.Molecule:
    """Convert SpinDry molecule to stk molecule."""
    atoms = [
        stk.Atom(
            id=i.get_id(),
            atomic_number=string_to_atom_number(i.get_element_string()),
        )
        for i in supramolecule.get_atoms()
    ]
    bonds = [
        stk.Bond(
            atom1=atoms[i.get_atom1_id()],
            atom2=atoms[i.get_atom2_id()],
            order=1,
        )
        for i in supramolecule.get_bonds()
    ]

    return stk.BuildingBlock.init(
        atoms=atoms,
        bonds=bonds,
        position_matrix=supramolecule.get_position_matrix(),
    )
