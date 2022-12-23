#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for geometry analysis.

Author: Andrew Tarzia

"""

import numpy as np
from rdkit.Chem import AllChem as rdkit

from utilities import get_atom_distance, angle_between, get_dihedral


class GeomMeasure:
    def __init__(self, torsion_set):
        self._torsion_set = tuple(torsion_set)

    def _get_paths(self, molecule, path_length):
        return rdkit.FindAllPathsOfLengthN(
            mol=molecule.to_rdkit_mol(),
            length=path_length,
            useBonds=False,
            useHs=True,
        )

    def calculate_bonds(self, molecule):
        lengths = []
        for bond in molecule.get_bonds():
            a1id = bond.get_atom1().get_id()
            a2id = bond.get_atom2().get_id()
            lengths.append(get_atom_distance(molecule, a1id, a2id))

        return lengths

    def calculate_angles(self, molecule):
        pos_mat = molecule.get_position_matrix()
        angles = []
        for a_ids in self._get_paths(molecule, 3):
            atoms = list(
                molecule.get_atoms(atom_ids=[i for i in a_ids])
            )
            atom1 = atoms[0]
            atom2 = atoms[1]
            atom3 = atoms[2]
            vector1 = pos_mat[atom2.get_id()] - pos_mat[atom1.get_id()]
            vector2 = pos_mat[atom2.get_id()] - pos_mat[atom3.get_id()]
            angles.append(np.degrees(angle_between(vector1, vector2)))

        return angles

    def calculate_dihedrals(self, molecule):
        torsions = []
        for a_ids in self._get_paths(molecule, 5):
            atoms = list(
                molecule.get_atoms(atom_ids=[i for i in a_ids])
            )
            estrings = tuple([i.__class__.__name__ for i in atoms])
            if estrings != self._torsion_set:
                continue
            torsion = get_dihedral(
                pt1=tuple(molecule.get_atomic_positions(a_ids[0]))[0],
                pt2=tuple(molecule.get_atomic_positions(a_ids[1]))[0],
                pt3=tuple(molecule.get_atomic_positions(a_ids[3]))[0],
                pt4=tuple(molecule.get_atomic_positions(a_ids[4]))[0],
            )
            torsions.append(abs(torsion))

        return torsions
