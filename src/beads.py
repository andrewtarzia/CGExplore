#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for Beads in GA cage optimisation.

Author: Andrew Tarzia

"""

from dataclasses import dataclass
from typing import Union, Tuple


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
