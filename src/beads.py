#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for Beads in GA cage optimisation.

Author: Andrew Tarzia

"""

from dataclasses import dataclass
import logging
from collections import Counter


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
    sigma: float
    bond_k: float
    angle_centered: float
    angle_k: float
    coordination: int


@dataclass
class GuestBead:
    element_string: str
    sigma: float
    epsilon: float


def guest_beads():
    return (GuestBead("Li", sigma=3.0, epsilon=1.0),)


def sets_in_library(library, sigma=None, angle=None):
    set_beads = set()
    for cgbead in library:
        sig_p = False
        if sigma is None:
            sig_p = True
        else:
            if cgbead.sigma == sigma:
                sig_p = True
            else:
                sig_p = False

        if angle is None:
            ang_p = True
        else:
            if cgbead.angle_centered == angle:
                ang_p = True
            else:
                ang_p = False

        if ang_p and sig_p:
            set_beads.add(cgbead.element_string)

    return set_beads


def bead_library_check(bead_libraries):
    logging.info(f"there are {len(bead_libraries)} beads")
    used_names = tuple(i.bead_type for i in bead_libraries)
    counts = Counter(used_names)
    if any((i > 1 for i in counts.values())):
        raise ValueError(
            f"you used a bead twice in your library: {counts}"
        )

    used_strings = tuple(i.element_string for i in bead_libraries)
    for string in used_strings:
        if string not in periodic_table():
            raise ValueError(
                f"you used a bead not available in PoreMapper: {string}"
            )
