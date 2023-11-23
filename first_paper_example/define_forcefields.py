#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module containing force field definitions.

Author: Andrew Tarzia

"""

from cgexplore.angles import PyramidAngleRange, TargetAngleRange
from cgexplore.bonds import TargetBondRange
from cgexplore.forcefield import ForceFieldLibrary
from cgexplore.nonbonded import TargetNonbondedRange
from cgexplore.torsions import TargetTorsionRange
from openmm import openmm


def define_3p4_forcefield_library(full_bead_library, prefix):
    forcefieldlibrary = ForceFieldLibrary(
        bead_library=full_bead_library,
        vdw_bond_cutoff=2,
        prefix=prefix,
    )

    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="b1",
            class2="b1",
            eclass1="Pb",
            eclass2="Pb",
            bond_rs=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="b1",
            class2="n1",
            eclass1="Pb",
            eclass2="C",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="b1",
            class2="m1",
            eclass1="Pb",
            eclass2="Pd",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="n1",
            class2="b1",
            class3="b1",
            eclass1="C",
            eclass2="Pb",
            eclass3="Pb",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="b1",
            class2="n1",
            class3="b1",
            eclass1="Pb",
            eclass2="C",
            eclass3="Pb",
            angles=(
                openmm.unit.Quantity(value=50, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=60, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=70, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=80, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=90, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=100, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=110, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=120, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="m1",
            class2="b1",
            class3="b1",
            eclass1="Pd",
            eclass2="Pb",
            eclass3="Pb",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        PyramidAngleRange(
            class1="b1",
            class2="m1",
            class3="b1",
            eclass1="Pb",
            eclass2="Pd",
            eclass3="Pb",
            angles=(
                openmm.unit.Quantity(value=50, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=60, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=70, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=80, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=90, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="n",
            bead_element="C",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="m",
            bead_element="Pd",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="b",
            bead_element="Pb",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )

    return forcefieldlibrary


def define_2p4_forcefield_library(full_bead_library, prefix):
    forcefieldlibrary = ForceFieldLibrary(
        bead_library=full_bead_library,
        vdw_bond_cutoff=2,
        prefix=prefix,
    )

    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="a1",
            class2="c1",
            eclass1="Ba",
            eclass2="Ag",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="a1",
            class2="b1",
            eclass1="Ba",
            eclass2="Pb",
            bond_rs=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="b1",
            class2="m1",
            eclass1="Pb",
            eclass2="Pd",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="a1",
            class2="c1",
            class3="a1",
            eclass1="Ba",
            eclass2="Ag",
            eclass3="Ba",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="b1",
            class2="a1",
            class3="c1",
            eclass1="Pb",
            eclass2="Ba",
            eclass3="Ag",
            angles=tuple(
                openmm.unit.Quantity(value=i, unit=openmm.unit.degrees)
                for i in range(90, 181, 5)
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="m1",
            class2="b1",
            class3="a1",
            eclass1="Pd",
            eclass2="Pb",
            eclass3="Ba",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        PyramidAngleRange(
            class1="b1",
            class2="m1",
            class3="b1",
            eclass1="Pd",
            eclass2="Pb",
            eclass3="Pd",
            angles=(
                openmm.unit.Quantity(value=50, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=60, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=70, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=80, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=90, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_torsion_range(
        TargetTorsionRange(
            search_string=("b1", "a1", "c1", "a1", "b1"),
            search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
            measured_atom_ids=[0, 1, 3, 4],
            phi0s=(openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),),
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

    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="a",
            bead_element="Ba",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="c",
            bead_element="Ag",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="m",
            bead_element="Pd",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="b",
            bead_element="Pb",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )

    return forcefieldlibrary


def define_2p3_forcefield_library(full_bead_library, prefix):
    forcefieldlibrary = ForceFieldLibrary(
        bead_library=full_bead_library,
        vdw_bond_cutoff=2,
        prefix=prefix,
    )

    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="a1",
            class2="c1",
            eclass1="Ba",
            eclass2="Ag",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="a1",
            class2="b1",
            eclass1="Ba",
            eclass2="Pb",
            bond_rs=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="b1",
            class2="n1",
            eclass1="Pb",
            eclass2="C",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="a1",
            class2="c1",
            class3="a1",
            eclass1="Ba",
            eclass2="Ag",
            eclass3="Ba",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="b1",
            class2="a1",
            class3="c1",
            eclass1="Pb",
            eclass2="Ba",
            eclass3="Ag",
            angles=tuple(
                openmm.unit.Quantity(value=i, unit=openmm.unit.degrees)
                for i in range(90, 181, 5)
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="n1",
            class2="b1",
            class3="a1",
            eclass1="C",
            eclass2="Pb",
            eclass3="Ba",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="b1",
            class2="n1",
            class3="b1",
            eclass1="Pb",
            eclass2="C",
            eclass3="Pb",
            angles=(
                openmm.unit.Quantity(value=50, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=60, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=70, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=80, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=90, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=100, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=110, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=120, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_torsion_range(
        TargetTorsionRange(
            search_string=("b1", "a1", "c1", "a1", "b1"),
            search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
            measured_atom_ids=[0, 1, 3, 4],
            phi0s=(openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),),
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

    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="a",
            bead_element="Ba",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="c",
            bead_element="Ag",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="n",
            bead_element="C",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            bead_class="b",
            bead_element="Pb",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            force="custom-excl-vol",
        )
    )

    return forcefieldlibrary


def neighbour_2p3_library(ffnum: int) -> list[int]:
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
    new_nums = []
    # Change bmb angle.
    new_nums.append(ffnum - 1)
    new_nums.append(ffnum + 1)
    # Change bnb angle.
    new_nums.append(ffnum - 5)
    new_nums.append(ffnum + 5)
    return new_nums


def get_neighbour_library(ffnum: int, fftype: str) -> list[int]:
    if fftype == "2p3":
        return neighbour_2p3_library(ffnum)
    elif fftype == "2p4":
        return neighbour_2p4_library(ffnum)
    elif fftype == "3p4":
        return neighbour_3p4_library(ffnum)
    else:
        msg = f"{fftype} not known"
        raise ValueError(msg)
