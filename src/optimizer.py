#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG optimizer.

Author: Andrew Tarzia

"""

import numpy as np
from itertools import combinations
import logging

from utilities import get_all_angles, angle_between, get_all_torsions


def lorentz_berthelot_sigma_mixing(sigma1, sigma2):
    return (sigma1 + sigma2) / 2


class CGOptimizer:
    def __init__(
        self,
        fileprefix,
        output_dir,
        param_pool,
        bonds,
        angles,
        torsions,
        vdw,
    ):
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._param_pool = {i.element_string: i for i in param_pool}
        self._bonds = bonds
        self._angles = angles
        self._torsions = torsions
        self._vdw = vdw
        self._mass = 1
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10

    def _yield_bonds(self, mol):
        if self._bonds is False:
            return ""
        logging.info(
            "OPT: you are not yet assigning different k values"
        )
        bond_k = 10
        bonds = list(mol.get_bonds())

        for bond in bonds:
            atom1 = bond.get_atom1()
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            atom2 = bond.get_atom2()
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__

            try:
                cgbead1 = self._param_pool[estring1]
                cgbead2 = self._param_pool[estring2]
                bond_r = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.sigma,
                    sigma2=cgbead2.sigma,
                )
                yield (name1, name2, bond_k, bond_r)
            except KeyError:
                logging.info(
                    f"OPT: {(name1, name2)} bond not assigned."
                )
                continue

    def _yield_angles(self, mol):
        if self._angles is False:
            return ""
        logging.info(
            "OPT: you are not yet assigning different k values"
        )
        angle_k = 20

        angles = get_all_angles(mol)
        pos_mat = mol.get_position_matrix()

        for angle in angles:
            outer_atom1, centre_atom, outer_atom2 = angle
            outer_name1 = (
                f"{outer_atom1.__class__.__name__}"
                f"{outer_atom1.get_id()+1}"
            )
            centre_name = (
                f"{centre_atom.__class__.__name__}"
                f"{centre_atom.get_id()+1}"
            )
            outer_name2 = (
                f"{outer_atom2.__class__.__name__}"
                f"{outer_atom2.get_id()+1}"
            )
            # outer_estring1 = outer_atom1.__class__.__name__
            centre_estring = centre_atom.__class__.__name__
            # outer_estring2 = outer_atom2.__class__.__name__
            try:
                # outer_cgbead1 = self._param_pool[outer_estring1]
                centre_cgbead = self._param_pool[centre_estring]
                # outer_cgbead2 = self._param_pool[outer_estring2]

                acentered = centre_cgbead.angle_centered
                if isinstance(acentered, int) or isinstance(
                    acentered, float
                ):
                    angle_theta = acentered

                elif isinstance(acentered, tuple):
                    min_angle, max_angle, cut_angle = acentered
                    vector1 = (
                        pos_mat[centre_atom.get_id()]
                        - pos_mat[outer_atom1.get_id()]
                    )
                    vector2 = (
                        pos_mat[centre_atom.get_id()]
                        - pos_mat[outer_atom2.get_id()]
                    )
                    curr_angle = np.degrees(
                        angle_between(vector1, vector2)
                    )
                    if curr_angle < cut_angle:
                        angle_theta = min_angle
                    elif curr_angle >= cut_angle:
                        angle_theta = max_angle

                yield (
                    centre_name,
                    outer_name1,
                    outer_name2,
                    angle_k,
                    angle_theta,
                )

            except KeyError:
                logging.info(
                    f"OPT: {(outer_name1, centre_name, outer_name2)} "
                    f"angle not assigned (centered on {centre_name})."
                )
                continue

    def _yield_torsions(self, mol):
        if self._torsions is False:
            return ""
        logging.info("OPT: not setting torsion ks, ns yet.")
        torsion_k = 1
        torsion_n = 1

        torsions = get_all_torsions(mol)
        for torsion in torsions:
            atom1, atom2, atom3, atom4 = torsion
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            name3 = f"{atom3.__class__.__name__}{atom3.get_id()+1}"
            name4 = f"{atom4.__class__.__name__}{atom4.get_id()+1}"

            atom2_estring = atom2.__class__.__name__
            atom3_estring = atom3.__class__.__name__

            try:
                cgbead2 = self._param_pool[atom2_estring]
                cgbead3 = self._param_pool[atom3_estring]

                phi0 = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead2.angle_centered,
                    sigma2=cgbead3.angle_centered,
                )

                yield (
                    name1,
                    name2,
                    name3,
                    name4,
                    torsion_k,
                    torsion_n,
                    phi0,
                )

            except KeyError:
                logging.info(
                    f"OPT: {(name1, name2, name3, name4)} "
                    f"angle not assigned."
                )
                continue

    def _yield_nonbondeds(self, mol):
        if self._vdw is False:
            return ""
        logging.info(
            "OPT: only vdw interactions between host and guest."
        )

        pairs = combinations(mol.get_atoms(), 2)
        for pair in pairs:
            atom1, atom2 = pair
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__
            guest_estrings = tuple(
                i
                for i in (estring1, estring2)
                if i in self._vdw_on_types
            )
            if len(guest_estrings) != 1:
                continue

            try:
                cgbead1 = self._param_pool[estring1]
                cgbead2 = self._param_pool[estring2]
                sigma = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.sigma,
                    sigma2=cgbead2.sigma,
                )
                epsilon = self._param_pool[guest_estrings[0]].epsilon
                yield (name1, name2, epsilon, sigma)

            except KeyError:
                # logging.info(f"OPT: {sorted_name} vdw not assigned.")
                continue

    def optimize(self, molecule):
        raise NotImplementedError()
