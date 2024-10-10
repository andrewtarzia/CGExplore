# Distributed under the terms of the MIT License.

"""Module containing force field definitions."""

from collections import abc

from openmm import openmm

from cgexplore.forcefields import ForceFieldLibrary
from cgexplore.molecular import CgBead
from cgexplore.terms import (
    PyramidAngleRange,
    TargetAngleRange,
    TargetBondRange,
    TargetNonbondedRange,
    TargetTorsionRange,
)


def define_forcefield_library(
    present_beads: abc.Sequence[CgBead],
    prefix: str,
) -> ForceFieldLibrary:
    """Define a forcefield library for iteration."""
    forcefieldlibrary = ForceFieldLibrary(
        present_beads=present_beads,
        vdw_bond_cutoff=2,
        prefix=prefix,
    )

    if "2p3" in prefix:
        bonds = (
            ("a1", "c1", "Ba", "Ag", (1.5,), (1e5,)),
            ("a1", "b1", "Ba", "Pb", (1.0,), (1e5,)),
            ("b1", "n1", "Pb", "C", (1.5,), (1e5,)),
        )
    elif "2p4" in prefix:
        bonds = (
            ("a1", "c1", "Ba", "Ag", (1.5,), (1e5,)),
            ("a1", "b1", "Ba", "Pb", (1.0,), (1e5,)),
            ("b1", "m1", "Pb", "Pd", (1.5,), (1e5,)),
        )
    elif "3p4" in prefix:
        bonds = (
            ("b1", "b1", "Pb", "Pb", (1.0,), (1e5,)),
            ("b1", "n1", "Pb", "C", (1.5,), (1e5,)),
            ("b1", "m1", "Pb", "Pd", (1.5,), (1e5,)),
        )
    for bond in bonds:
        r_range = tuple(
            openmm.unit.Quantity(value=b, unit=openmm.unit.angstrom)
            for b in bond[4]
        )
        k_range = tuple(
            openmm.unit.Quantity(
                value=k,
                unit=openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.nanometer**2,
            )
            for k in bond[5]
        )
        forcefieldlibrary.add_bond_range(
            TargetBondRange(
                type1=bond[0],
                type2=bond[1],
                element1=bond[2],
                element2=bond[3],
                bond_rs=r_range,
                bond_ks=k_range,
            )
        )

    if "2p3" in prefix:
        angles = (
            ("a1", "c1", "a1", "Ba", "Ag", "Ba", (180,), (1e2,)),
            ("b1", "a1", "c1", "Pb", "Ba", "Ag", range(90, 181, 5), (1e2,)),
            ("n1", "b1", "a1", "C", "Pb", "Ba", (180,), (1e2,)),
            (
                "b1",
                "n1",
                "b1",
                "Pb",
                "C",
                "Pb",
                (50, 60, 70, 80, 90, 100, 110, 120),
                (1e2,),
            ),
        )
        pyramids = ()
    elif "2p4" in prefix:
        angles = (
            ("a1", "c1", "a1", "Ba", "Ag", "Ba", (180,), (1e2,)),
            ("b1", "a1", "c1", "Pb", "Ba", "Ag", range(90, 181, 5), (1e2,)),
            ("m1", "b1", "a1", "Pd", "Pb", "Ba", (180,), (1e2,)),
        )
        pyramids = (
            ("b1", "m1", "b1", "Pb", "Pd", "Pb", (50, 60, 70, 80, 90), (1e2,)),
        )
    elif "3p4" in prefix:
        angles = (
            ("n1", "b1", "b1", "C", "Pb", "Pb", (180,), (1e2,)),
            (
                "b1",
                "n1",
                "b1",
                "Pb",
                "C",
                "Pb",
                (50, 60, 70, 80, 90, 100, 110, 120),
                (1e2,),
            ),
            ("m1", "b1", "b1", "Pd", "Pb", "Pb", (180,), (1e2,)),
        )
        pyramids = (
            ("b1", "m1", "b1", "Pb", "Pd", "Pb", (50, 60, 70, 80, 90), (1e2,)),
        )
    for angle in angles:
        a_range = tuple(
            openmm.unit.Quantity(value=a, unit=openmm.unit.degrees)
            for a in angle[6]
        )
        k_range = tuple(
            openmm.unit.Quantity(
                value=k,
                unit=openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.radian**2,
            )
            for k in angle[7]
        )
        forcefieldlibrary.add_angle_range(
            TargetAngleRange(
                type1=angle[0],
                type2=angle[1],
                type3=angle[2],
                element1=angle[3],
                element2=angle[4],
                element3=angle[5],
                angles=a_range,
                angle_ks=k_range,
            )
        )
    for pyramid in pyramids:
        a_range = tuple(
            openmm.unit.Quantity(value=a, unit=openmm.unit.degrees)
            for a in pyramid[6]
        )
        k_range = tuple(
            openmm.unit.Quantity(
                value=k,
                unit=openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.radian**2,
            )
            for k in pyramid[7]
        )
        forcefieldlibrary.add_angle_range(
            PyramidAngleRange(
                type1=pyramid[0],
                type2=pyramid[1],
                type3=pyramid[2],
                element1=pyramid[3],
                element2=pyramid[4],
                element3=pyramid[5],
                angles=a_range,
                angle_ks=k_range,
            )
        )

    if prefix in ("2p3", "2p4"):
        forcefieldlibrary.add_torsion_range(
            TargetTorsionRange(
                search_string=("b1", "a1", "c1", "a1", "b1"),
                search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                measured_atom_ids=[0, 1, 3, 4],
                phi0s=(
                    openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
                ),
                torsion_ks=(
                    openmm.unit.Quantity(
                        value=50,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                    openmm.unit.Quantity(
                        value=0,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                ),
                torsion_ns=(1,),
            )
        )

    if "2p3" in prefix:
        nonbondeds = (
            ("a", "Ba", 10.0, 1.0),
            ("c", "Ag", 10.0, 1.0),
            ("b", "Pb", 10.0, 1.0),
            ("n", "C", 10.0, 1.0),
        )
    elif "2p4" in prefix:
        nonbondeds = (
            ("a", "Ba", 10.0, 1.0),
            ("c", "Ag", 10.0, 1.0),
            ("b", "Pb", 10.0, 1.0),
            ("m", "Pd", 10.0, 1.0),
        )
    elif "3p4" in prefix:
        nonbondeds = (
            ("n", "C", 10.0, 1.0),
            ("b", "Pb", 10.0, 1.0),
            ("m", "Pd", 10.0, 1.0),
        )
    for nb in nonbondeds:
        forcefieldlibrary.add_nonbonded_range(
            TargetNonbondedRange(
                bead_class=nb[0],
                bead_element=nb[1],
                epsilons=(
                    openmm.unit.Quantity(
                        value=nb[2], unit=openmm.unit.kilojoules_per_mole
                    ),
                ),
                sigmas=(
                    openmm.unit.Quantity(
                        value=nb[3], unit=openmm.unit.angstrom
                    ),
                ),
                force="custom-excl-vol",
            )
        )

    return forcefieldlibrary


def neighbour_2p3_library(ffnum: int) -> list[int]:
    """Define neighbour from forcefield library."""
    new_nums = []
    # Change bnb angle.
    new_nums.append(ffnum - 14)
    new_nums.append(ffnum + 14)
    # Change bac angle.
    new_nums.append(ffnum - 2)
    new_nums.append(ffnum + 2)
    # Change torsion, even is ton.
    if ffnum % 2 == 0:
        new_nums.append(ffnum + 1)
    elif ffnum % 2 != 0:
        new_nums.append(ffnum - 1)
    return new_nums


def neighbour_2p4_library(ffnum: int) -> list[int]:
    """Define neighbour from forcefield library."""
    new_nums = []
    # Change bmb angle.
    new_nums.append(ffnum - 10)
    new_nums.append(ffnum + 10)
    # Change bac angle.
    new_nums.append(ffnum - 2)
    new_nums.append(ffnum + 2)
    # Change torsion, even is ton.
    if ffnum % 2 == 0:
        new_nums.append(ffnum + 1)
    elif ffnum % 2 != 0:
        new_nums.append(ffnum - 1)
    return new_nums


def neighbour_3p4_library(ffnum: int) -> list[int]:
    """Define neighbour from forcefield library."""
    new_nums = []
    # Change bmb angle.
    new_nums.append(ffnum - 1)
    new_nums.append(ffnum + 1)
    # Change bnb angle.
    new_nums.append(ffnum - 5)
    new_nums.append(ffnum + 5)
    return new_nums


def get_neighbour_library(ffnum: int, fftype: str) -> list[int]:
    """Get neighbour library."""
    if fftype == "2p3":
        return neighbour_2p3_library(ffnum)
    if fftype == "2p4":
        return neighbour_2p4_library(ffnum)
    if fftype == "3p4":
        return neighbour_3p4_library(ffnum)

    msg = f"{fftype} not known"
    raise ValueError(msg)
