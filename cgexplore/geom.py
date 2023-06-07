#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for geometry analysis.

Author: Andrew Tarzia

"""

import numpy as np
from rdkit.Chem import AllChem as rdkit
from collections import defaultdict
from scipy.spatial.distance import pdist

from .utilities import get_atom_distance, angle_between, get_dihedral


class GeomMeasure:
    def __init__(self, torsion_set=None):
        if torsion_set is None:
            self._torsion_set = None
        else:
            self._torsion_set = tuple(torsion_set)

    def _get_paths(self, molecule, path_length):
        return rdkit.FindAllPathsOfLengthN(
            mol=molecule.to_rdkit_mol(),
            length=path_length,
            useBonds=False,
            useHs=True,
        )

    def calculate_minb2b(self, molecule):
        pair_dists = pdist(molecule.get_position_matrix())
        return np.min(pair_dists.flatten())

    def calculate_bonds(self, molecule):
        lengths = defaultdict(list)
        for bond in molecule.get_bonds():
            a1id = bond.get_atom1().get_id()
            a2id = bond.get_atom2().get_id()
            length_type = "_".join(
                (
                    bond.get_atom1().__class__.__name__,
                    bond.get_atom2().__class__.__name__,
                )
            )
            lengths[length_type].append(
                get_atom_distance(molecule, a1id, a2id)
            )

        return lengths

    def calculate_angles(self, molecule):
        pos_mat = molecule.get_position_matrix()
        angles = defaultdict(list)
        for a_ids in self._get_paths(molecule, 3):
            atoms = list(
                molecule.get_atoms(atom_ids=[i for i in a_ids])
            )
            atom1 = atoms[0]
            atom2 = atoms[1]
            atom3 = atoms[2]
            angle_type = "_".join(
                (
                    atom1.__class__.__name__,
                    atom2.__class__.__name__,
                    atom3.__class__.__name__,
                )
            )
            vector1 = pos_mat[atom2.get_id()] - pos_mat[atom1.get_id()]
            vector2 = pos_mat[atom2.get_id()] - pos_mat[atom3.get_id()]
            angles[angle_type].append(
                np.degrees(angle_between(vector1, vector2))
            )

        return angles

    def calculate_torsions(self, molecule):
        if self._torsion_set is None:
            return []

        torsions = defaultdict(list)
        for a_ids in self._get_paths(molecule, 5):
            atoms = list(
                molecule.get_atoms(atom_ids=[i for i in a_ids])
            )
            estrings = tuple([i.__class__.__name__ for i in atoms])
            if estrings != self._torsion_set:
                continue
            torsion_type = "_".join(
                (
                    atoms[0].__class__.__name__,
                    atoms[1].__class__.__name__,
                    atoms[3].__class__.__name__,
                    atoms[4].__class__.__name__,
                )
            )
            torsion = get_dihedral(
                pt1=tuple(molecule.get_atomic_positions(a_ids[0]))[0],
                pt2=tuple(molecule.get_atomic_positions(a_ids[1]))[0],
                pt3=tuple(molecule.get_atomic_positions(a_ids[3]))[0],
                pt4=tuple(molecule.get_atomic_positions(a_ids[4]))[0],
            )
            torsions[torsion_type].append(abs(torsion))

        return torsions

    def calculate_radius_gyration(self, molecule):
        centroid = molecule.get_centroid()
        pos_mat = molecule.get_position_matrix()
        vectors = pos_mat - centroid
        distances2 = np.square(np.linalg.norm(vectors, axis=1))

        rg2 = (1 / molecule.get_num_atoms()) * np.sum(distances2)
        return np.sqrt(rg2)

    def calculate_max_diameter(self, molecule):
        return molecule.get_maximum_diameter()
