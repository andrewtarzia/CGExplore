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

import numpy as np
import stk

from .beads import get_cgbead_from_element
from .forcefield import Forcefield
from .utilities import (
    angle_between,
    convert_pyramid_angle,
    get_all_angles,
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
        force_field: Forcefield,
    ) -> None:
        self._force_field = force_field
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10

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

    def optimize(self, molecule: stk.Molecule) -> stk.Molecule:
        raise NotImplementedError()
