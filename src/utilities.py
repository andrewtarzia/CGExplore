#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Utilities module.

Author: Andrew Tarzia

"""

import numpy as np
import json
import logging
from rdkit.Chem import AllChem as rdkit
from scipy.spatial.distance import euclidean


def get_distances(optimizer, cage):
    bond_set = optimizer.define_bond_potentials()
    set_ks = tuple(bond_set.get_keys())
    distances = {"".join(i): [] for i in set_ks}
    for bond in cage.get_bonds():
        a1 = bond.get_atom1()
        a2 = bond.get_atom2()
        a1name = a1.__class__.__name__
        a2name = a2.__class__.__name__
        pair = tuple(sorted([a1name, a2name]))
        if pair in set_ks:
            a1id = a1.get_id()
            a2id = a2.get_id()
            distances["".join(pair)].append(
                get_atom_distance(cage, a1id, a2id)
            )

    return distances


def get_angles(optimizer, cage):
    angle_set = optimizer.define_angle_potentials()
    set_ks = tuple(angle_set.get_keys())
    angles = {"".join(i): [] for i in set_ks}
    pos_mat = cage.get_position_matrix()

    angle_atoms = get_all_angles(cage)
    for angle_trip in angle_atoms:
        triplet = tuple(
            sorted([i.__class__.__name__ for i in angle_trip])
        )
        if triplet in set_ks:
            a1id = angle_trip[0].get_id()
            a2id = angle_trip[1].get_id()
            a3id = angle_trip[2].get_id()
            vector1 = pos_mat[a2id] - pos_mat[a1id]
            vector2 = pos_mat[a2id] - pos_mat[a3id]
            angles["".join(triplet)].append(
                np.degrees(angle_between(vector1, vector2))
            )

    return angles


def get_atom_distance(molecule, atom1_id, atom2_id):
    """
    Return the distance between atom1 and atom2.

    Parameters
    ----------
    molecule : :class:`stk.Molecule`

    atom1_id : :class:`int`
        The id of atom1.

    atom2_id : :class:`int`
        The id of atom2.

    Returns
    -------
    :class:`float`
        The euclidean distance between two atoms.

    """

    position_matrix = molecule.get_position_matrix()

    distance = euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id],
    )

    return float(distance)


def unit_vector(vector):
    """
    Returns the unit vector of the vector.

    https://stackoverflow.com/questions/2827393/
    angles-between-two-n-dimensional-vectors-in-python/
    13849249#13849249

    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, normal=None):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

    https://stackoverflow.com/questions/2827393/
    angles-between-two-n-dimensional-vectors-in-python/
    13849249#13849249

    If normal is given, the angle polarity is determined using the
    cross product of the two vectors.

    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if normal is not None:
        # Get normal vector and cross product to determine sign.
        cross = np.cross(v1_u, v2_u)
        if np.dot(normal, cross) < 0:
            angle = -angle
    return angle


def read_lib(lib_file):
    """
    Read lib file.
    Returns dictionary.
    """

    logging.info(f"reading {lib_file}")
    with open(lib_file, "rb") as f:
        lib = json.load(f)

    return lib


def get_all_angles(molecule):

    paths = rdkit.FindAllPathsOfLengthN(
        mol=molecule.to_rdkit_mol(),
        length=3,
        useBonds=False,
        useHs=True,
    )
    angles = []
    for atom_ids in paths:
        atoms = list(molecule.get_atoms(atom_ids=[i for i in atom_ids]))
        atom1 = atoms[0]
        atom2 = atoms[1]
        atom3 = atoms[2]
        angles.append((atom1, atom2, atom3))

    return angles
