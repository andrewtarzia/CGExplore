#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for Beads in GA cage optimisation.

Author: Andrew Tarzia

"""

from dataclasses import dataclass
from typing import Union, Tuple
import itertools
import logging

from gulp_optimizer import HarmBond, ThreeAngle, CheckedThreeAngle


@dataclass
class CgBead:
    element_string: str
    sigma: float
    connectivity: int
    angle_centered: Union[float, Tuple[float]]


# Mn = CgBead("F", sigma=2.0, connectivity=0, angle_centered=180)
# Hg = CgBead("Hg", sigma=2.0, connectivity=0, angle_centered=180)
# Mo = CgBead("Mo", sigma=2.0, connectivity=0, angle_centered=180)
# Nd = CgBead("Nd", sigma=2.0, connectivity=0, angle_centered=180)
# Ne = CgBead("Ne", sigma=2.0, connectivity=0, angle_centered=180)
# Ni = CgBead("Cs", sigma=2.0, connectivity=0, angle_centered=180)
# Nb = CgBead("Nb", sigma=2.0, connectivity=0, angle_centered=180)
# N = CgBead("N", sigma=2.0, connectivity=0, angle_centered=180)
# Os = CgBead("Os", sigma=2.0, connectivity=0, angle_centered=180)
# O = CgBead("Li", sigma=2.0, connectivity=0, angle_centered=180)
# Pd = CgBead("Pd", sigma=2.0, connectivity=0, angle_centered=180)
# P = CgBead("P", sigma=2.0, connectivity=0, angle_centered=180)
# Pt = CgBead("Pt", sigma=2.0, connectivity=0, angle_centered=180)
# K = CgBead("K", sigma=2.0, connectivity=0, angle_centered=180)
# CgBead("H", sigma=2.0, connectivity=3, angle_centered=60),
# CgBead("Kr", sigma=2.0, connectivity=3, angle_centered=60),
# CgBead("La", sigma=2.0, connectivity=3, angle_centered=60),
# CgBead("Dy", sigma=2.0, connectivity=3, angle_centered=60),
# CgBead("Cl", sigma=2.0, connectivity=3, angle_centered=60),
# CgBead("Lu", sigma=2.0, connectivity=3, angle_centered=60),


def core_beads():
    return (
        CgBead("O", sigma=3.0, connectivity=1, angle_centered=40),
        CgBead("Cr", sigma=3.0, connectivity=1, angle_centered=60),
        CgBead("Co", sigma=3.0, connectivity=1, angle_centered=90),
        CgBead("Cu", sigma=3.0, connectivity=1, angle_centered=120),
        CgBead("Pb", sigma=3.0, connectivity=1, angle_centered=140),
        CgBead("Er", sigma=3.0, connectivity=1, angle_centered=160),
        CgBead("Eu", sigma=3.0, connectivity=1, angle_centered=180),
        CgBead("Mn", sigma=4.0, connectivity=1, angle_centered=40),
        CgBead("Gd", sigma=4.0, connectivity=1, angle_centered=60),
        CgBead("Ga", sigma=4.0, connectivity=1, angle_centered=90),
        CgBead("Ge", sigma=4.0, connectivity=1, angle_centered=120),
        CgBead("Au", sigma=4.0, connectivity=1, angle_centered=140),
        CgBead("Ni", sigma=4.0, connectivity=1, angle_centered=160),
        CgBead("He", sigma=4.0, connectivity=1, angle_centered=180),
    )


def beads_2c():
    return (
        CgBead("Al", sigma=3.0, connectivity=2, angle_centered=40),
        CgBead("Sb", sigma=3.0, connectivity=2, angle_centered=60),
        CgBead("Ar", sigma=3.0, connectivity=2, angle_centered=90),
        CgBead("As", sigma=3.0, connectivity=2, angle_centered=120),
        CgBead("Ba", sigma=3.0, connectivity=2, angle_centered=140),
        CgBead("Be", sigma=3.0, connectivity=2, angle_centered=160),
        CgBead("Bi", sigma=3.0, connectivity=2, angle_centered=180),
        CgBead("B", sigma=4.0, connectivity=2, angle_centered=40),
        CgBead("Mg", sigma=4.0, connectivity=2, angle_centered=60),
        CgBead("Cd", sigma=4.0, connectivity=2, angle_centered=90),
        CgBead("Hf", sigma=4.0, connectivity=2, angle_centered=120),
        CgBead("Ca", sigma=4.0, connectivity=2, angle_centered=140),
        CgBead("C", sigma=4.0, connectivity=2, angle_centered=160),
        CgBead("Ce", sigma=4.0, connectivity=2, angle_centered=180),
    )


def beads_3c():
    return (
        CgBead("Ho", sigma=2.0, connectivity=3, angle_centered=120),
        CgBead("Fe", sigma=2.5, connectivity=3, angle_centered=120),
        CgBead("In", sigma=3.0, connectivity=3, angle_centered=120),
        CgBead("I", sigma=3.5, connectivity=3, angle_centered=120),
        CgBead("Ir", sigma=4.0, connectivity=3, angle_centered=120),
    )


def beads_4c():
    return (CgBead("Pt", 2.0, 4, (90, 180, 130)),)


def lorentz_berthelot_sigma_mixing(sigma1, sigma2):
    return (sigma1 + sigma2) / 2


def bond_library():
    logging.info("you are not yet assigning different k values")
    bond_k = 10

    all_beads = core_beads() + beads_2c() + beads_3c() + beads_4c()

    bonds = [
        HarmBond(
            atom1_type=i.element_string,
            atom2_type=i.element_string,
            bond_r=i.sigma,
            bond_k=bond_k,
        )
        for i in all_beads
    ]
    for pair in itertools.combinations(all_beads, r=2):
        sorted_ids = tuple(
            sorted([pair[0].element_string, pair[1].element_string])
        )
        mixed_radii = lorentz_berthelot_sigma_mixing(
            sigma1=pair[0].sigma,
            sigma2=pair[1].sigma,
        )
        bonds.append(
            HarmBond(
                atom1_type=sorted_ids[0],
                atom2_type=sorted_ids[1],
                bond_r=mixed_radii,
                bond_k=bond_k,
            )
        )

    return tuple(bonds)


def angle_library():
    logging.info("you are not yet assigning different k values")
    angle_k = 20

    all_beads = core_beads() + beads_2c() + beads_3c() + beads_4c()

    angles = []
    for triplet in itertools.combinations_with_replacement(
        all_beads, 3
    ):
        # Atom1 type defines the centre atom.
        acentered = triplet[1].angle_centered
        if isinstance(acentered, int) or isinstance(acentered, float):
            angles.append(
                ThreeAngle(
                    atom1_type=triplet[1].element_string,
                    atom2_type=triplet[0].element_string,
                    atom3_type=triplet[2].element_string,
                    theta=acentered,
                    angle_k=angle_k,
                )
            )
        elif isinstance(acentered, tuple):
            angles.append(
                CheckedThreeAngle(
                    atom1_type=triplet[1].element_string,
                    atom2_type=triplet[0].element_string,
                    atom3_type=triplet[2].element_string,
                    cut_angle=acentered[2],
                    min_angle=acentered[0],
                    max_angle=acentered[1],
                    angle_k=angle_k,
                )
            )

    return tuple(angles)
