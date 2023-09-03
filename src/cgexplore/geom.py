#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for geometry analysis.

Author: Andrew Tarzia

"""

import typing
from collections import defaultdict

import numpy as np
import stk
from rdkit.Chem import AllChem as rdkit
from scipy.spatial.distance import pdist

from .torsions import find_torsions
from .utilities import angle_between, get_atom_distance, get_dihedral


class GeomMeasure:
    def __init__(self, target_torsions: typing.Iterable | None = None) -> None:
        if target_torsions is None:
            self._target_torsions = None
        else:
            self._target_torsions = tuple(target_torsions)

    def _get_paths(
        self,
        molecule: stk.Molecule,
        path_length: int,
    ) -> tuple[tuple[int]]:
        return rdkit.FindAllPathsOfLengthN(
            mol=molecule.to_rdkit_mol(),
            length=path_length,
            useBonds=False,
            useHs=True,
        )

    def calculate_minb2b(self, molecule: stk.Molecule) -> float:
        pair_dists = pdist(molecule.get_position_matrix())
        return np.min(pair_dists.flatten())

    def calculate_bonds(
        self,
        molecule: stk.Molecule,
    ) -> dict[str, list[float]]:
        lengths = defaultdict(list)
        for bond in molecule.get_bonds():
            a1id = bond.get_atom1().get_id()
            a2id = bond.get_atom2().get_id()
            length_type = "_".join(
                sorted(
                    (
                        bond.get_atom1().__class__.__name__,
                        bond.get_atom2().__class__.__name__,
                    )
                )
            )
            lengths[length_type].append(
                get_atom_distance(molecule, a1id, a2id)
            )

        return lengths

    def calculate_angles(
        self,
        molecule: stk.Molecule,
    ) -> dict[str, list[float]]:
        pos_mat = molecule.get_position_matrix()
        angles = defaultdict(list)
        for a_ids in self._get_paths(molecule, 3):
            atoms = list(molecule.get_atoms(atom_ids=[i for i in a_ids]))
            atom1 = atoms[0]
            atom2 = atoms[1]
            atom3 = atoms[2]
            angle_type_option1 = "_".join(
                (
                    atom1.__class__.__name__,
                    atom2.__class__.__name__,
                    atom3.__class__.__name__,
                )
            )
            angle_type_option2 = "_".join(
                (
                    atom3.__class__.__name__,
                    atom2.__class__.__name__,
                    atom1.__class__.__name__,
                )
            )

            vector1 = pos_mat[atom2.get_id()] - pos_mat[atom1.get_id()]
            vector2 = pos_mat[atom2.get_id()] - pos_mat[atom3.get_id()]

            if angle_type_option1 in angles:
                angles[angle_type_option1].append(
                    np.degrees(angle_between(vector1, vector2))
                )
            elif angle_type_option2 in angles:
                angles[angle_type_option2].append(
                    np.degrees(angle_between(vector1, vector2))
                )
            else:
                angles[angle_type_option1].append(
                    np.degrees(angle_between(vector1, vector2))
                )

        return angles

    def calculate_torsions(
        self,
        molecule: stk.Molecule,
        absolute: bool,
    ) -> dict[str, list[float]]:
        if self._target_torsions is None:
            return {}

        torsions = defaultdict(list)
        for target_torsion in self._target_torsions:
            for torsion in find_torsions(
                molecule, len(target_torsion.search_estring)
            ):
                estrings = tuple([i.__class__.__name__ for i in torsion.atoms])
                if estrings not in (
                    target_torsion.search_estring,
                    tuple(reversed(target_torsion.search_estring)),
                ):
                    continue
                torsion_type_option1 = "_".join(
                    tuple(
                        estrings[i] for i in target_torsion.measured_atom_ids
                    )
                )
                torsion_type_option2 = "_".join(
                    tuple(
                        estrings[i]
                        for i in reversed(target_torsion.measured_atom_ids)
                    )
                )

                if torsion_type_option1 in torsions:
                    key_string = torsion_type_option1
                    new_ids = tuple(
                        torsion.atom_ids[i]
                        for i in target_torsion.measured_atom_ids
                    )
                elif torsion_type_option2 in torsions:
                    key_string = torsion_type_option2
                    new_ids = tuple(
                        torsion.atom_ids[i]
                        for i in reversed(target_torsion.measured_atom_ids)
                    )
                else:
                    key_string = torsion_type_option1
                    new_ids = tuple(
                        torsion.atom_ids[i]
                        for i in target_torsion.measured_atom_ids
                    )

                torsion_value = get_dihedral(
                    pt1=tuple(molecule.get_atomic_positions(new_ids[0]))[0],
                    pt2=tuple(molecule.get_atomic_positions(new_ids[1]))[0],
                    pt3=tuple(molecule.get_atomic_positions(new_ids[2]))[0],
                    pt4=tuple(molecule.get_atomic_positions(new_ids[3]))[0],
                )

                if absolute:
                    torsion_value = abs(torsion_value)
                torsions[key_string].append(torsion_value)

        return torsions

    def calculate_radius_gyration(self, molecule: stk.Molecule) -> float:
        centroid = molecule.get_centroid()
        pos_mat = molecule.get_position_matrix()
        vectors = pos_mat - centroid
        distances2 = np.square(np.linalg.norm(vectors, axis=1))

        rg2 = (1 / molecule.get_num_atoms()) * np.sum(distances2)
        return np.sqrt(rg2)

    def calculate_max_diameter(self, molecule: stk.Molecule) -> float:
        return molecule.get_maximum_diameter()
