# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging

import stk
from scipy.spatial.distance import euclidean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_atom_distance(
    molecule: stk.Molecule,
    atom1_id: int,
    atom2_id: int,
) -> float:
    """Return the distance between atom1 and atom2.

    Parameters:
        molecule:

        atom1_id:
            The id of atom1.

        atom2_id:
            The id of atom2.

    Returns:
        The euclidean distance between two atoms.

    """
    position_matrix = molecule.get_position_matrix()

    distance = euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id],
    )

    return float(distance)
