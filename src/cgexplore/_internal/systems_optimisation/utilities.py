# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging

from openmm import openmm

from cgexplore.molecular import CgBead
from cgexplore.terms import (
    TargetAngle,
    TargetBond,
    TargetCosineAngle,
    TargetNonbonded,
    TargetTorsion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def element_from_type(
    bead_type: str,
    present_beads: tuple[CgBead, ...],
) -> str:
    """Get element of cgbead from type of cgbead."""
    return next(
        i.element_string for i in present_beads if i.bead_type == bead_type
    )


def define_bond(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[CgBead, ...],
) -> TargetBond:
    """Define target from a known structured list.

    TODO: Handle other forces.
    TODO: Use generic dataclasses.
    """
    return TargetBond(
        type1=interaction_key[0],
        type2=interaction_key[1],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        bond_r=openmm.unit.Quantity(
            value=interaction_list[1], unit=openmm.unit.angstrom
        ),
        bond_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule
            / openmm.unit.mole
            / openmm.unit.nanometer**2,
        ),
    )


def define_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[CgBead, ...],
) -> TargetAngle:
    """Define target from a known structured list.

    TODO: Handle other forces.
    TODO: Use generic dataclasses.
    """
    return TargetAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        angle=openmm.unit.Quantity(
            value=interaction_list[1], unit=openmm.unit.degrees
        ),
        angle_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule
            / openmm.unit.mole
            / openmm.unit.radian**2,
        ),
    )


def define_cosine_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[CgBead, ...],
) -> TargetCosineAngle:
    """Define target from a known structured list.

    TODO: Handle other forces.
    TODO: Use generic dataclasses.
    """
    return TargetCosineAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        n=interaction_list[1],
        b=interaction_list[2],
        angle_k=openmm.unit.Quantity(
            value=interaction_list[3],
            unit=openmm.unit.kilojoule / openmm.unit.mole,
        ),
    )


def define_torsion(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[CgBead, ...],
) -> TargetTorsion:
    """Define target from a known structured list.

    TODO: Handle other forces.
    TODO: Use generic dataclasses.
    """
    return TargetTorsion(
        search_string=tuple(i for i in interaction_key),
        search_estring=tuple(
            element_from_type(test, present_beads) for test in interaction_key
        ),
        measured_atom_ids=interaction_list[1],
        phi0=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.degrees,
        ),
        torsion_k=openmm.unit.Quantity(
            value=interaction_list[3],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        torsion_n=interaction_list[4],
    )


def define_nonbonded(
    interaction_key: str,
    interaction_list: list,
    present_beads: tuple[CgBead, ...],
) -> TargetNonbonded:
    """Define target from a known structured list.

    TODO: Handle other forces.
    TODO: Use generic dataclasses.
    """
    return TargetNonbonded(
        bead_class=interaction_key[0],
        bead_element=element_from_type(interaction_key[0], present_beads),
        epsilon=openmm.unit.Quantity(
            value=interaction_list[1],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        sigma=openmm.unit.Quantity(
            value=interaction_list[2], unit=openmm.unit.angstrom
        ),
        force="custom-excl-vol",
    )
