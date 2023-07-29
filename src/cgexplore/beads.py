#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for beads.

Author: Andrew Tarzia

"""

import itertools
import logging
from collections import Counter
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_cgbead_from_string(string, bead_set):
    return bead_set[string]


def get_cgbead_from_element(estring, bead_set):
    for i in bead_set:
        bead = bead_set[i]
        if bead.element_string == estring:
            return bead


def periodic_table():
    return {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Pa": 91,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Cf": 98,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Db": 105,
    }


def string_to_atom_number(string):
    return periodic_table()[string]


@dataclass
class CgBead:
    element_string: str
    bead_type: str
    bond_r: float
    bond_k: float
    angle_centered: float
    angle_k: float
    sigma: float
    epsilon: float
    coordination: int


@dataclass
class GuestBead:
    element_string: str
    sigma: float
    epsilon: float


def guest_beads():
    return (GuestBead("Li", sigma=3.0, epsilon=1.0),)


def bead_library_check(bead_libraries):
    logging.info(f"there are {len(bead_libraries)} beads")
    used_names = tuple(i.bead_type for i in bead_libraries)
    counts = Counter(used_names)
    if any((i > 1 for i in counts.values())):
        raise ValueError(f"you used a bead twice in your library: {counts}")

    used_strings = tuple(i.element_string for i in bead_libraries)
    for string in used_strings:
        if string not in periodic_table():
            raise ValueError(
                f"you used a bead not available in PoreMapper: {string}"
            )

    bl_string = "bead library:\n"
    bl_string += (
        "element name bond_r[A] bond_k[kJ/mol/nm^2] angle[deg.] "
        "angle_k[kJ/mol/radian2] sigma[A] epsilon[kJ/mol] coord.\n"
    )
    for bead in bead_libraries:
        bl_string += (
            f"{bead.element_string} {bead.bead_type} "
            f"{bead.bond_r} {bead.bond_k} "
            f"{bead.angle_centered} {bead.angle_k} "
            f"{bead.sigma} {bead.epsilon} {bead.coordination}\n"
        )
    logging.info(bl_string)


def produce_bead_library(
    type_prefix,
    element_string,
    bond_rs,
    angles,
    bond_ks,
    angle_ks,
    sigma,
    epsilon,
    coordination,
):
    return {
        f"{type_prefix}{idx1}{idx2}{idx3}{idx4}": CgBead(
            element_string=element_string,
            bead_type=f"{type_prefix}{idx1}{idx2}{idx3}{idx4}",
            bond_r=bond_r,
            angle_centered=angle,
            bond_k=bond_k,
            angle_k=angle_k,
            sigma=sigma,
            epsilon=epsilon,
            coordination=coordination,
        )
        for (idx1, bond_r), (idx2, angle), (idx3, bond_k), (
            idx4,
            angle_k,
        ) in itertools.product(
            enumerate(bond_rs),
            enumerate(angles),
            enumerate(bond_ks),
            enumerate(angle_ks),
        )
    }
