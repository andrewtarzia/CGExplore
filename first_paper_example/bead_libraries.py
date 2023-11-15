#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module containing bead libraries.

Author: Andrew Tarzia

"""

from cgexplore.beads import CgBead


def core_bead() -> CgBead:
    return CgBead(
        element_string="Ag",
        bead_class="c",
        bead_type="c1",
        coordination=2,
    )


def arm_bead() -> CgBead:
    return CgBead(
        element_string="Ba",
        bead_class="a",
        bead_type="a1",
        coordination=2,
    )


def binder_bead() -> CgBead:
    return CgBead(
        element_string="Pb",
        bead_class="b",
        bead_type="b1",
        coordination=2,
    )


def trigonal_bead() -> CgBead:
    return CgBead(
        element_string="C",
        bead_class="n",
        bead_type="n1",
        coordination=3,
    )


def tetragonal_bead() -> CgBead:
    return CgBead(
        element_string="Pd",
        bead_class="m",
        bead_type="m1",
        coordination=4,
    )
