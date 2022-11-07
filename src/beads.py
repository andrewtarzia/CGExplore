#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for Beads in GA cage optimisation.

Author: Andrew Tarzia

"""

from dataclasses import dataclass
from typing import Union, Tuple


def string_to_atom_number(string):
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
        "Tb": 65,
        "Dy": 66,
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
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Uut": 113,
        "Fl": 114,
        "Uup": 115,
        "Lv": 116,
        "Uus": 117,
        "Uuo": 118,
    }[string]


@dataclass
class CgBead:
    element_string: str
    sigma: float
    angle_centered: Union[float, Tuple[float]]


@dataclass
class GuestBead:
    element_string: str
    sigma: float
    epsilon: float


# Mn = CgBead("F", sigma=2.0, connectivity=0, angle_centered=180)
# Hg = CgBead("Hg", sigma=2.0, connectivity=0, angle_centered=180)
# Mo = CgBead("Mo", sigma=2.0, connectivity=0, angle_centered=180)
# Nd = CgBead("Nd", sigma=2.0, connectivity=0, angle_centered=180)
# Ne = CgBead("Ne", sigma=2.0, connectivity=0, angle_centered=180)
# Ni = CgBead("Cs", sigma=2.0, connectivity=0, angle_centered=180)
# Nb = CgBead("Nb", sigma=2.0, connectivity=0, angle_centered=180)
# N = CgBead("N", sigma=2.0, connectivity=0, angle_centered=180)
# Os = CgBead("Os", sigma=2.0, connectivity=0, angle_centered=180)
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
# CgBead("Ni", sigma=2.0, angle_centered=160),
# CgBead("Be", sigma=3.0, angle_centered=160),
# CgBead("Er", sigma=5.0, angle_centered=160),
# CgBead("C", sigma=4.0, angle_centered=160),


def guest_beads():
    return (GuestBead("Li", sigma=3.0, epsilon=1.0),)


def beads_2c():
    return (
        CgBead("Mn", sigma=2.0, angle_centered=40),
        CgBead("Gd", sigma=2.0, angle_centered=60),
        CgBead("Ga", sigma=2.0, angle_centered=90),
        CgBead("Ge", sigma=2.0, angle_centered=120),
        CgBead("Au", sigma=2.0, angle_centered=150),
        CgBead("He", sigma=2.0, angle_centered=180),
        CgBead("Al", sigma=3.0, angle_centered=40),
        CgBead("Sb", sigma=3.0, angle_centered=60),
        CgBead("Ar", sigma=3.0, angle_centered=90),
        CgBead("As", sigma=3.0, angle_centered=120),
        CgBead("Ba", sigma=3.0, angle_centered=150),
        CgBead("Bi", sigma=3.0, angle_centered=180),
        CgBead("B", sigma=4.0, angle_centered=40),
        CgBead("Mg", sigma=4.0, angle_centered=60),
        CgBead("Cd", sigma=4.0, angle_centered=90),
        CgBead("Hf", sigma=4.0, angle_centered=120),
        CgBead("Ca", sigma=4.0, angle_centered=150),
        CgBead("Ce", sigma=4.0, angle_centered=180),
        CgBead("O", sigma=5.0, angle_centered=40),
        CgBead("Cr", sigma=5.0, angle_centered=60),
        CgBead("Co", sigma=5.0, angle_centered=90),
        CgBead("Be", sigma=5.0, angle_centered=120),
        CgBead("Pb", sigma=5.0, angle_centered=150),
        CgBead("Eu", sigma=5.0, angle_centered=180),
    )


def beads_3c():
    return (
        CgBead("Ho", sigma=2.0, angle_centered=120),
        CgBead("Fe", sigma=2.5, angle_centered=120),
        CgBead("In", sigma=3.0, angle_centered=120),
        CgBead("I", sigma=3.5, angle_centered=120),
        CgBead("Ir", sigma=4.0, angle_centered=120),
        CgBead("Ni", sigma=4.5, angle_centered=120),
        CgBead("Cu", sigma=5.0, angle_centered=120),
        CgBead("Er", sigma=5.5, angle_centered=120),
        CgBead("C", sigma=6.0, angle_centered=120),
    )


def beads_4c():
    return (CgBead("Pt", 2.0, 4, (90, 180, 130)),)
