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
from heapq import nsmallest
from rdkit.Chem import AllChem as rdkit

from utilities import (
    get_all_angles,
    angle_between,
    get_all_torsions,
    convert_pyramid_angle,
)


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
        self._param_pool = param_pool
        self._bonds = bonds
        self._angles = angles
        self._torsions = torsions
        self._vdw = vdw
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10

    def _get_cgbead_from_element(self, estring):
        for i in self._param_pool:
            bead = self._param_pool[i]
            if bead.element_string == estring:
                return bead

    def _get_new_torsions(self, molecule, chain_length):
        paths = rdkit.FindAllPathsOfLengthN(
            mol=molecule.to_rdkit_mol(),
            length=chain_length,
            useBonds=False,
            useHs=True,
        )
        torsions = []
        for atom_ids in paths:
            atoms = list(
                molecule.get_atoms(atom_ids=[i for i in atom_ids])
            )
            atom1 = atoms[0]
            atom2 = atoms[1]
            atom3 = atoms[2]
            atom4 = atoms[3]
            atom5 = atoms[4]
            torsions.append((atom1, atom2, atom3, atom4, atom5))

        return torsions

    def _yield_bonds(self, molecule):
        if self._bonds is False:
            return ""

        bonds = list(molecule.get_bonds())
        for bond in bonds:
            atom1 = bond.get_atom1()
            id1 = atom1.get_id()
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            atom2 = bond.get_atom2()
            id2 = atom2.get_id()
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__

            try:
                cgbead1 = self._get_cgbead_from_element(estring1)
                cgbead2 = self._get_cgbead_from_element(estring2)
                bond_r = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.bond_r,
                    sigma2=cgbead2.bond_r,
                )
                bond_k = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.bond_k,
                    sigma2=cgbead2.bond_k,
                )
                yield (name1, name2, id1, id2, bond_k, bond_r)
            except KeyError:
                logging.info(
                    f"OPT: {(name1, name2)} bond not assigned."
                )
                continue

    def _yield_angles(self, molecule):
        if self._angles is False:
            return ""

        angles = get_all_angles(molecule)
        pos_mat = molecule.get_position_matrix()

        saved_angles = {}
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
            outer_id1 = outer_atom1.get_id()
            centre_id = centre_atom.get_id()
            outer_id2 = outer_atom2.get_id()
            centre_estring = centre_atom.__class__.__name__

            try:
                centre_cgbead = self._get_cgbead_from_element(
                    estring=centre_estring,
                )

                angle_theta = centre_cgbead.angle_centered
                angle_k = centre_cgbead.angle_k
                if centre_cgbead.coordination == 4:
                    if centre_name not in saved_angles:
                        saved_angles[centre_name] = []
                    saved_angles[centre_name].append(
                        (
                            angle_theta,
                            angle_k,
                            outer_atom1,
                            outer_atom2,
                            centre_atom,
                            centre_id,
                            outer_id1,
                            outer_id2,
                        )
                    )
                    continue

                yield (
                    centre_name,
                    outer_name1,
                    outer_name2,
                    centre_id,
                    outer_id1,
                    outer_id2,
                    angle_k,
                    angle_theta,
                )

            except KeyError:
                logging.info(
                    f"OPT: {(outer_name1, centre_name, outer_name2)} "
                    f"angle not assigned (centered on {centre_name})."
                )
                continue

        # For four coordinate systems, only apply angles between
        # neighbouring atoms.
        for centre_name in saved_angles:
            sa_d = saved_angles[centre_name]

            all_angles = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X[4].get_id()]
                        - pos_mat[X[2].get_id()],
                        v2=pos_mat[X[4].get_id()]
                        - pos_mat[X[3].get_id()],
                    )
                )
                for i, X in enumerate(sa_d)
            }
            four_smallest = nsmallest(4, all_angles, key=all_angles.get)
            for used_ang_id in four_smallest:
                used_ang = sa_d[used_ang_id]
                (
                    angle_theta,
                    angle_k,
                    outer_atom1,
                    outer_atom2,
                    centre_atom,
                    centre_id,
                    outer_id1,
                    outer_id2,
                ) = used_ang

                outer_name1 = (
                    f"{outer_atom1.__class__.__name__}"
                    f"{outer_atom1.get_id()+1}"
                )
                outer_name2 = (
                    f"{outer_atom2.__class__.__name__}"
                    f"{outer_atom2.get_id()+1}"
                )
                yield (
                    centre_name,
                    outer_name1,
                    outer_name2,
                    centre_id,
                    outer_id1,
                    outer_id2,
                    angle_k,
                    angle_theta,
                )

            for used_ang_id in all_angles:
                if used_ang_id in four_smallest:
                    continue
                used_ang = sa_d[used_ang_id]
                (
                    angle_theta,
                    angle_k,
                    outer_atom1,
                    outer_atom2,
                    centre_atom,
                    centre_id,
                    outer_id1,
                    outer_id2,
                ) = used_ang
                angle_theta = convert_pyramid_angle(angle_theta)
                outer_name1 = (
                    f"{outer_atom1.__class__.__name__}"
                    f"{outer_atom1.get_id()+1}"
                )
                outer_name2 = (
                    f"{outer_atom2.__class__.__name__}"
                    f"{outer_atom2.get_id()+1}"
                )
                yield (
                    centre_name,
                    outer_name1,
                    outer_name2,
                    centre_id,
                    outer_id1,
                    outer_id2,
                    angle_k,
                    angle_theta,
                )

    def _yield_torsions(self, molecule):
        if self._torsions is False:
            return ""
        logging.info("OPT, WARNING: torsions are hardcoded!.")
        phi0 = 180
        torsion_n = 1
        torsion_k = 50

        torsions = get_all_torsions(molecule)
        for torsion in torsions:
            atom1, atom2, atom3, atom4 = torsion
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            name3 = f"{atom3.__class__.__name__}{atom3.get_id()+1}"
            name4 = f"{atom4.__class__.__name__}{atom4.get_id()+1}"
            id1 = atom1.get_id()
            id2 = atom2.get_id()
            id3 = atom3.get_id()
            id4 = atom4.get_id()

            try:
                yield (
                    name1,
                    name2,
                    name3,
                    name4,
                    id1,
                    id2,
                    id3,
                    id4,
                    torsion_k,
                    torsion_n,
                    phi0,
                )

            except KeyError:
                continue

    def _yield_nonbondeds(self, molecule):
        raise NotImplementedError()
        if self._vdw is False:
            return ""
        logging.info(
            "OPT: only vdw interactions between host and guest."
        )

        pairs = combinations(molecule.get_atoms(), 2)
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
                cgbead1 = self._get_cgbead_from_element(estring1)
                cgbead2 = self._get_cgbead_from_element(estring2)
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
