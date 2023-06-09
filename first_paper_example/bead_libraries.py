#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module containing bead libraries.

Author: Andrew Tarzia

"""

from cgexplore.beads import produce_bead_library


def bond_k():
    return 1e5


def angle_k():
    return 1e2


def core_2c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        angles=(180,),
        bond_rs=(2,),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def arm_2c_beads():
    return produce_bead_library(
        type_prefix="a",
        element_string="Ba",
        bond_rs=(1,),
        angles=range(90, 181, 5),
        # angles=(90, 100, 120, 140, 160, 180),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def binder_beads():
    return produce_bead_library(
        type_prefix="b",
        element_string="Pb",
        bond_rs=(1,),
        angles=(180,),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def beads_3c():
    return produce_bead_library(
        type_prefix="n",
        element_string="C",
        bond_rs=(2,),
        angles=(50, 60, 70, 80, 90, 100, 110, 120),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=3,
    )


def beads_4c():
    return produce_bead_library(
        type_prefix="m",
        element_string="Pd",
        bond_rs=(2,),
        angles=(50, 60, 70, 80, 90),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=4,
    )
