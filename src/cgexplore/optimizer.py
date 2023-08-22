#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG optimizer.

Author: Andrew Tarzia

"""

import logging
import typing
from heapq import nsmallest
from itertools import combinations

import numpy as np
import stk

from .beads import CgBead, get_cgbead_from_element
from .torsions import Torsion, find_torsions
from .utilities import (
    angle_between,
    convert_pyramid_angle,
    get_all_angles,
    get_all_torsions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def lorentz_berthelot_sigma_mixing(sigma1: float, sigma2: float) -> float:
    return (sigma1 + sigma2) / 2


class CGOptimizer:
    def __init__(
        self,
        bead_set: dict[str, CgBead],
        custom_torsion_set: tuple | None,
        custom_vdw_set: tuple | None,
        bonds: bool,
        angles: bool,
        torsions: bool,
        vdw: bool,
    ) -> None:
        self._bead_set = bead_set
        self._custom_torsion_set = custom_torsion_set
        self._custom_vdw_set = custom_vdw_set
        self._bonds = bonds
        self._angles = angles
        self._torsions = torsions
        self._vdw = vdw
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10

    def _yield_bonds(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[tuple]:
        if self._bonds is False:
            return ()

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
                cgbead1 = get_cgbead_from_element(
                    estring=estring1,
                    bead_set=self._bead_set,
                )
                cgbead2 = get_cgbead_from_element(
                    estring=estring2,
                    bead_set=self._bead_set,
                )
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
                logging.info(f"OPT: {(name1, name2)} bond not assigned.")
                continue

    def _yield_angles(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[tuple]:
        if self._angles is False:
            return ()

        angles = get_all_angles(molecule)
        pos_mat = molecule.get_position_matrix()

        pyramid_angles: dict[str, list] = {}
        octahedral_angles: dict[str, list] = {}
        for angle in angles:
            outer_atom1, centre_atom, outer_atom2 = angle
            outer_name1 = (
                f"{outer_atom1.__class__.__name__}" f"{outer_atom1.get_id()+1}"
            )
            centre_name = (
                f"{centre_atom.__class__.__name__}" f"{centre_atom.get_id()+1}"
            )
            outer_name2 = (
                f"{outer_atom2.__class__.__name__}" f"{outer_atom2.get_id()+1}"
            )
            outer_id1 = outer_atom1.get_id()
            centre_id = centre_atom.get_id()
            outer_id2 = outer_atom2.get_id()
            centre_estring = centre_atom.__class__.__name__

            try:
                centre_cgbead = get_cgbead_from_element(
                    estring=centre_estring,
                    bead_set=self._bead_set,
                )

                angle_theta = centre_cgbead.angle_centered
                angle_k = centre_cgbead.angle_k
                if centre_cgbead.coordination == 4:
                    if centre_name not in pyramid_angles:
                        pyramid_angles[centre_name] = []
                    pyramid_angles[centre_name].append(
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

                elif centre_cgbead.coordination == 6:
                    if centre_name not in octahedral_angles:
                        octahedral_angles[centre_name] = []
                    octahedral_angles[centre_name].append(
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

        # For four coordinate systems, apply standard angle theta to
        # neighbouring atoms, then compute pyramid angle for opposing
        # interaction.
        for centre_name in pyramid_angles:
            sa_d = pyramid_angles[centre_name]

            all_angles = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X[4].get_id()] - pos_mat[X[2].get_id()],
                        v2=pos_mat[X[4].get_id()] - pos_mat[X[3].get_id()],
                    )
                )
                for i, X in enumerate(sa_d)
            }
            four_smallest = nsmallest(
                n=4,
                iterable=all_angles,
                key=all_angles.get,  # type: ignore[arg-type]
            )
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

        # For six coordinate systems, assume octahedral geometry.
        # So 90 degrees with 12 smallest angles, 180 degrees for the rest.
        for centre_name in octahedral_angles:
            sa_d = octahedral_angles[centre_name]

            all_angles = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X[4].get_id()] - pos_mat[X[2].get_id()],
                        v2=pos_mat[X[4].get_id()] - pos_mat[X[3].get_id()],
                    )
                )
                for i, X in enumerate(sa_d)
            }
            smallest = nsmallest(
                n=12,
                iterable=all_angles,
                key=all_angles.get,  # type: ignore[arg-type]
            )
            for used_ang_id in smallest:
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
                if used_ang_id in smallest:
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
                angle_theta = 180
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

    def _yield_torsions(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[tuple]:
        if self._torsions is False:
            return ()
        raise NotImplementedError()

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

            # Here would be where you capture the torsions in some
            # provided forcefield.
            phi0 = None
            torsion_n = None
            torsion_k = None

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

    def _yield_custom_torsions(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[Torsion]:
        if self._custom_torsion_set is None:
            return ()

        # Iterate over the different path lengths, and find all torsions
        # for that lengths.
        path_lengths = set(
            len(i.search_string) for i in self._custom_torsion_set
        )
        for pl in path_lengths:
            for found_torsion in find_torsions(molecule, pl):
                atom_estrings = list(
                    i.__class__.__name__ for i in found_torsion.atoms
                )
                cgbeads = list(
                    get_cgbead_from_element(i, self._bead_set)
                    for i in atom_estrings
                )
                cgbead_string = tuple(i.bead_type[0] for i in cgbeads)
                for target_torsion in self._custom_torsion_set:
                    if target_torsion.search_string != cgbead_string:
                        continue
                    yield Torsion(
                        atom_names=tuple(
                            f"{found_torsion.atoms[i].__class__.__name__}"
                            f"{found_torsion.atoms[i].get_id()+1}"
                            for i in target_torsion.measured_atom_ids
                        ),
                        atom_ids=tuple(
                            found_torsion.atoms[i].get_id()
                            for i in target_torsion.measured_atom_ids
                        ),
                        phi0=target_torsion.phi0,
                        torsion_n=target_torsion.torsion_n,
                        torsion_k=target_torsion.torsion_k,
                    )

    def _yield_nonbondeds(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[tuple]:
        raise NotImplementedError()
        if self._vdw is False:
            return ()
        logging.info("OPT: only vdw interactions between host and guest.")

        pairs = combinations(molecule.get_atoms(), 2)
        for pair in pairs:
            atom1, atom2 = pair
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__
            guest_estrings = tuple(
                i for i in (estring1, estring2) if i in self._vdw_on_types
            )
            if len(guest_estrings) != 1:
                continue

            try:
                cgbead1 = get_cgbead_from_element(
                    estring=estring1,
                    bead_set=self._bead_set,
                )
                cgbead2 = get_cgbead_from_element(
                    estring=estring2,
                    bead_set=self._bead_set,
                )
                sigma = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.sigma,
                    sigma2=cgbead2.sigma,
                )
                epsilon = self._bead_set[guest_estrings[0]].epsilon
                yield (name1, name2, epsilon, sigma)

            except KeyError:
                # logging.info(f"OPT: {sorted_name} vdw not assigned.")
                continue

    def optimize(self, molecule: stk.Molecule) -> stk.Molecule:
        raise NotImplementedError()
