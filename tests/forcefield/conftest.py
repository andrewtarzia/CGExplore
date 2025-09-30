import numpy as np
import pytest
import stk
from openmm import openmm

from cgexplore.forcefields import ForceFieldLibrary
from cgexplore.molecular import CgBead, FourC1Arm
from cgexplore.terms import (
    Angle,
    Bond,
    CosineAngle,
    Nonbonded,
    PyramidAngleRange,
    TargetAngleRange,
    TargetBondRange,
    TargetCosineAngleRange,
    TargetNonbondedRange,
    TargetTorsionRange,
    Torsion,
)

from .case_data import CaseData

ag_bead = CgBead(
    element_string="Ag",
    bead_type="c1",
    bead_class="c",
    coordination=2,
)
ba_bead = CgBead(
    element_string="Ba",
    bead_type="a1",
    bead_class="a",
    coordination=2,
)
pb_bead = CgBead(
    element_string="Pb",
    bead_type="b1",
    bead_class="b",
    coordination=2,
)
pd_bead = CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m1",
    coordination=4,
)
c_bead = CgBead(
    element_string="C",
    bead_type="c1",
    bead_class="c",
    coordination=2,
)
n_bead = CgBead(
    element_string="N",
    bead_type="n1",
    bead_class="n",
    coordination=2,
)
o_bead = CgBead(
    element_string="O",
    bead_type="o1",
    bead_class="o",
    coordination=2,
)

kjmol = openmm.unit.kilojoules_per_mole


@pytest.fixture(
    params=(
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[Pb][Ba][Ag][Ba][Pb]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [0.5, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                    )
                ),
            ),
            forcefield_library=ForceFieldLibrary(
                present_beads=(ag_bead, ba_bead, pb_bead),
                vdw_bond_cutoff=2,
                prefix="testff",
            ),
            bond_ranges=(
                TargetBondRange(
                    type1="a1",
                    type2="b1",
                    element1="Ba",
                    element2="Pb",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    bond_ks=(
                        openmm.unit.Quantity(
                            value=1e5,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                    ),
                ),
            ),
            angle_ranges=(
                TargetAngleRange(
                    type1="n1",
                    type2="b1",
                    type3="a1",
                    element1="C",
                    element2="Pb",
                    element3="Ba",
                    angles=(
                        openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                    ),
                    angle_ks=(
                        openmm.unit.Quantity(
                            value=1e2,
                            unit=kjmol / openmm.unit.radian**2,
                        ),
                    ),
                ),
            ),
            torsion_ranges=(
                TargetTorsionRange(
                    search_string=("b1", "a1", "c1", "a1", "b1"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0s=(
                        openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                    ),
                    torsion_ks=(
                        openmm.unit.Quantity(value=50, unit=kjmol),
                        openmm.unit.Quantity(value=0, unit=kjmol),
                    ),
                    torsion_ns=(1,),
                ),
            ),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    "c",
                    "Ag",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
                TargetNonbondedRange(
                    "b",
                    "Pb",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
            ),
            num_forcefields=2,
            present_bonds=(
                (
                    Bond(
                        atom_names=("Pb1", "Ba2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pb(0), stk.Ba(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Ba4", "Pb5"),
                        atom_ids=(3, 4),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Ba(3), stk.Pb(4)),
                        force="HarmonicBondForce",
                    ),
                ),
                (
                    Bond(
                        atom_names=("Pb1", "Ba2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pb(0), stk.Ba(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Ba4", "Pb5"),
                        atom_ids=(3, 4),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Ba(3), stk.Pb(4)),
                        force="HarmonicBondForce",
                    ),
                ),
            ),
            present_angles=((), ()),
            present_nonbondeds=(
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="Ag",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="Ag",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
            ),
            present_torsions=(
                (
                    Torsion(
                        atom_names=("Pb1", "Ba2", "Ba4", "Pb5"),
                        atom_ids=(0, 1, 3, 4),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                ),
                (
                    Torsion(
                        atom_names=("Pb1", "Ba2", "Ba4", "Pb5"),
                        atom_ids=(0, 1, 3, 4),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        torsion_k=openmm.unit.Quantity(value=0, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                ),
            ),
            library_string=(
                "ForceFieldLibrary(\n  present_beads=(CgBead(element_string='A"
                "g', bead_type='c1', bead_class='c', coordination=2), CgBead(e"
                "lement_string='Ba', bead_type='a1', bead_class='a', coordinat"
                "ion=2), CgBead(element_string='Pb', bead_type='b1', bead_clas"
                "s='b', coordination=2)),\n  bond_ranges=(TargetBondRange(type"
                "1='a1', type2='b1', element1='Ba', element2='Pb', bond_rs=(1."
                "0 A,), bond_ks=(100000.0 kJ/(nm**2 mol),)),),\n  angle_ranges"
                "=(TargetAngleRange(type1='n1', type2='b1', type3='a1', elemen"
                "t1='C', element2='Pb', element3='Ba', angles=(180 deg,), angl"
                "e_ks=(100.0 kJ/(mol rad**2),)),),\n  torsion_ranges=(TargetTo"
                "rsionRange(search_string=('b1', 'a1', 'c1', 'a1', 'b1'), sear"
                "ch_estring=('Pb', 'Ba', 'Ag', 'Ba', 'Pb'), measured_atom_ids="
                "[0, 1, 3, 4], phi0s=(180 deg,), torsion_ks=(50 kJ/mol, 0 kJ/m"
                "ol), torsion_ns=(1,)),),\n  nonbonded_ranges=(TargetNonbonded"
                "Range(bead_class='c', bead_element='Ag', sigmas=(1.0 A,), eps"
                "ilons=(10.0 kJ/mol,), force='custom-excl-vol'), TargetNonbond"
                "edRange(bead_class='b', bead_element='Pb', sigmas=(1.0 A,), e"
                "psilons=(10.0 kJ/mol,), force='custom-excl-vol'))\n)"
            ),
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][N][C][C].[C][C][N][C][C][C]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                        [1, 2, 2],
                        [1, 1, 2],
                        [1, 0, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    )
                ),
            ),
            forcefield_library=ForceFieldLibrary(
                present_beads=(c_bead, n_bead, o_bead),
                vdw_bond_cutoff=2,
                prefix="testff",
            ),
            bond_ranges=(
                TargetBondRange(
                    type1="n1",
                    type2="c1",
                    element1="C",
                    element2="N",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    bond_ks=(
                        openmm.unit.Quantity(
                            value=1e5, unit=kjmol / openmm.unit.nanometer**2
                        ),
                    ),
                ),
                TargetBondRange(
                    type1="n1",
                    type2="o1",
                    element1="C",
                    element2="O",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    bond_ks=(
                        openmm.unit.Quantity(
                            value=1e5, unit=kjmol / openmm.unit.nanometer**2
                        ),
                    ),
                ),
            ),
            angle_ranges=(
                TargetAngleRange(
                    type1="c1",
                    type2="n1",
                    type3="c1",
                    element1="C",
                    element2="N",
                    element3="C",
                    angles=(
                        openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degrees
                        ),
                    ),
                    angle_ks=(
                        openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                ),
            ),
            torsion_ranges=(
                TargetTorsionRange(
                    search_string=("c1", "n1", "c1", "c1"),
                    search_estring=("C", "N", "C", "C"),
                    measured_atom_ids=[0, 1, 2, 3],
                    phi0s=(
                        openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                    ),
                    torsion_ks=(openmm.unit.Quantity(value=50, unit=kjmol),),
                    torsion_ns=(1,),
                ),
            ),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    "c",
                    "C",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
                TargetNonbondedRange(
                    "n",
                    "N",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
                TargetNonbondedRange(
                    "o",
                    "O",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
            ),
            num_forcefields=4,
            present_bonds=(
                (
                    Bond(
                        atom_names=("C1", "N2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(0), stk.N(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N2", "C3"),
                        atom_ids=(1, 2),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(1), stk.C(2)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("C6", "N7"),
                        atom_ids=(5, 6),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(5), stk.N(6)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N7", "C8"),
                        atom_ids=(6, 7),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(6), stk.C(7)),
                        force="HarmonicBondForce",
                    ),
                ),
                (
                    Bond(
                        atom_names=("C1", "N2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(0), stk.N(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N2", "C3"),
                        atom_ids=(1, 2),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(1), stk.C(2)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("C6", "N7"),
                        atom_ids=(5, 6),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(5), stk.N(6)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N7", "C8"),
                        atom_ids=(6, 7),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(6), stk.C(7)),
                        force="HarmonicBondForce",
                    ),
                ),
                (
                    Bond(
                        atom_names=("C1", "N2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(0), stk.N(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N2", "C3"),
                        atom_ids=(1, 2),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(1), stk.C(2)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("C6", "N7"),
                        atom_ids=(5, 6),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(5), stk.N(6)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N7", "C8"),
                        atom_ids=(6, 7),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(6), stk.C(7)),
                        force="HarmonicBondForce",
                    ),
                ),
                (
                    Bond(
                        atom_names=("C1", "N2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(0), stk.N(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N2", "C3"),
                        atom_ids=(1, 2),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(1), stk.C(2)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("C6", "N7"),
                        atom_ids=(5, 6),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(5), stk.N(6)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N7", "C8"),
                        atom_ids=(6, 7),
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(6), stk.C(7)),
                        force="HarmonicBondForce",
                    ),
                ),
            ),
            present_angles=(
                (
                    Angle(
                        atom_names=("C1", "N2", "C3"),
                        atom_ids=(0, 1, 2),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(0), stk.N(1), stk.C(2)),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atom_names=("C6", "N7", "C8"),
                        atom_ids=(5, 6, 7),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(5), stk.N(6), stk.C(7)),
                        force="HarmonicAngleForce",
                    ),
                ),
                (
                    Angle(
                        atom_names=("C1", "N2", "C3"),
                        atom_ids=(0, 1, 2),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(0), stk.N(1), stk.C(2)),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atom_names=("C6", "N7", "C8"),
                        atom_ids=(5, 6, 7),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(5), stk.N(6), stk.C(7)),
                        force="HarmonicAngleForce",
                    ),
                ),
                (
                    Angle(
                        atom_names=("C1", "N2", "C3"),
                        atom_ids=(0, 1, 2),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(0), stk.N(1), stk.C(2)),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atom_names=("C6", "N7", "C8"),
                        atom_ids=(5, 6, 7),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(5), stk.N(6), stk.C(7)),
                        force="HarmonicAngleForce",
                    ),
                ),
                (
                    Angle(
                        atom_names=("C1", "N2", "C3"),
                        atom_ids=(0, 1, 2),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(0), stk.N(1), stk.C(2)),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atom_names=("C6", "N7", "C8"),
                        atom_ids=(5, 6, 7),
                        angle=openmm.unit.Quantity(
                            value=160, unit=openmm.unit.degree
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(5), stk.N(6), stk.C(7)),
                        force="HarmonicAngleForce",
                    ),
                ),
            ),
            present_nonbondeds=(
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=3,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=5,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=6,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=7,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=8,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=9,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=3,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=5,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=6,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=7,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=8,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=9,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=3,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=5,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=6,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=7,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=8,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=9,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=3,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=5,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=6,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=7,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=8,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=9,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
            ),
            present_torsions=(
                (
                    Torsion(
                        atom_names=("C1", "N2", "C3", "C4"),
                        atom_ids=(0, 1, 2, 3),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C5", "C6", "N7", "C8"),
                        atom_ids=(4, 5, 6, 7),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C6", "N7", "C8", "C9"),
                        atom_ids=(5, 6, 7, 8),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                ),
                (
                    Torsion(
                        atom_names=("C1", "N2", "C3", "C4"),
                        atom_ids=(0, 1, 2, 3),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C5", "C6", "N7", "C8"),
                        atom_ids=(4, 5, 6, 7),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C6", "N7", "C8", "C9"),
                        atom_ids=(5, 6, 7, 8),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                ),
                (
                    Torsion(
                        atom_names=("C1", "N2", "C3", "C4"),
                        atom_ids=(0, 1, 2, 3),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C5", "C6", "N7", "C8"),
                        atom_ids=(4, 5, 6, 7),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C6", "N7", "C8", "C9"),
                        atom_ids=(5, 6, 7, 8),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                ),
                (
                    Torsion(
                        atom_names=("C1", "N2", "C3", "C4"),
                        atom_ids=(0, 1, 2, 3),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C5", "C6", "N7", "C8"),
                        atom_ids=(4, 5, 6, 7),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                    Torsion(
                        atom_names=("C6", "N7", "C8", "C9"),
                        atom_ids=(5, 6, 7, 8),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degree
                        ),
                        torsion_k=openmm.unit.Quantity(value=50, unit=kjmol),
                        torsion_n=1,
                        force="PeriodicTorsionForce",
                    ),
                ),
            ),
            library_string=(
                "ForceFieldLibrary(\n  present_beads=(CgBead(element_string='C"
                "', bead_type='c1', bead_class='c', coordination=2), CgBead(el"
                "ement_string='N', bead_type='n1', bead_class='n', coordinatio"
                "n=2), CgBead(element_string='O', bead_type='o1', bead_class='"
                "o', coordination=2)),\n  bond_ranges=(TargetBondRange(type1='"
                "n1', type2='c1', element1='C', element2='N', bond_rs=(1.0 A, "
                "2.0 A), bond_ks=(100000.0 kJ/(nm**2 mol),)), TargetBondRange("
                "type1='n1', type2='o1', element1='C', element2='O', bond_rs=("
                "2.0 A,), bond_ks=(100000.0 kJ/(nm**2 mol),))),\n  angle_range"
                "s=(TargetAngleRange(type1='c1', type2='n1', type3='c1', eleme"
                "nt1='C', element2='N', element3='C', angles=(160 deg,), angle"
                "_ks=(100.0 kJ/(mol rad**2),)),),\n  torsion_ranges=(TargetTor"
                "sionRange(search_string=('c1', 'n1', 'c1', 'c1'), search_estr"
                "ing=('C', 'N', 'C', 'C'), measured_atom_ids=[0, 1, 2, 3], phi"
                "0s=(180 deg,), torsion_ks=(50 kJ/mol,), torsion_ns=(1,)),),\n"
                "  nonbonded_ranges=(TargetNonbondedRange(bead_class='c', bead"
                "_element='C', sigmas=(1.0 A, 2.0 A), epsilons=(10.0 kJ/mol,),"
                " force='custom-excl-vol'), TargetNonbondedRange(bead_class='n"
                "', bead_element='N', sigmas=(1.0 A,), epsilons=(10.0 kJ/mol,)"
                ", force='custom-excl-vol'), TargetNonbondedRange(bead_class='"
                "o', bead_element='O', sigmas=(1.0 A,), epsilons=(10.0 kJ/mol,"
                "), force='custom-excl-vol'))\n)"
            ),
            name=name,
        ),
        lambda name: CaseData(
            molecule=FourC1Arm(
                bead=pd_bead, abead1=pb_bead
            ).get_building_block(),
            forcefield_library=ForceFieldLibrary(
                present_beads=(pd_bead, pb_bead),
                vdw_bond_cutoff=2,
                prefix="testff",
            ),
            bond_ranges=(
                TargetBondRange(
                    type1="b1",
                    type2="m1",
                    element1="Pb",
                    element2="Pd",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                    ),
                    bond_ks=(
                        openmm.unit.Quantity(
                            value=1e5,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2,
                        ),
                    ),
                ),
            ),
            angle_ranges=(
                PyramidAngleRange(
                    type1="b1",
                    type2="m1",
                    type3="b1",
                    element1="Pd",
                    element2="Pb",
                    element3="Pd",
                    angles=(
                        openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                    ),
                    angle_ks=(
                        openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                ),
            ),
            torsion_ranges=(),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    bead_class="m",
                    bead_element="Pd",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
                TargetNonbondedRange(
                    bead_class="b",
                    bead_element="Pb",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
            ),
            num_forcefields=2,
            present_bonds=(
                (
                    Bond(
                        atom_names=("Pd1", "Pb2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Pd1", "Pb3"),
                        atom_ids=(0, 2),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(2)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Pd1", "Pb4"),
                        atom_ids=(0, 3),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(3)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Pd1", "Pb5"),
                        atom_ids=(0, 4),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(4)),
                        force="HarmonicBondForce",
                    ),
                ),
                (
                    Bond(
                        atom_names=("Pd1", "Pb2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Pd1", "Pb3"),
                        atom_ids=(0, 2),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(2)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Pd1", "Pb4"),
                        atom_ids=(0, 3),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(3)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("Pd1", "Pb5"),
                        atom_ids=(0, 4),
                        bond_r=openmm.unit.Quantity(
                            value=1.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.Pd(0), stk.Pb(4)),
                        force="HarmonicBondForce",
                    ),
                ),
            ),
            present_angles=(
                (
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb3"),
                        atom_ids=(1, 0, 2),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb4"),
                        atom_ids=(1, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=130.75, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb5"),
                        atom_ids=(1, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb4"),
                        atom_ids=(2, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb5"),
                        atom_ids=(2, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=130.75, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb4", "Pd1", "Pb5"),
                        atom_ids=(3, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                ),
                (
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb3"),
                        atom_ids=(1, 0, 2),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb4"),
                        atom_ids=(1, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb5"),
                        atom_ids=(1, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb4"),
                        atom_ids=(2, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb5"),
                        atom_ids=(2, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb4", "Pd1", "Pb5"),
                        atom_ids=(3, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2, unit=kjmol / openmm.unit.radian**2
                        ),
                        force="HarmonicAngleForce",
                    ),
                ),
            ),
            present_nonbondeds=(
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="m",
                        bead_element="Pd",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=3,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="m",
                        bead_element="Pd",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=3,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=4,
                        bead_class="b",
                        bead_element="Pb",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
            ),
            present_torsions=((), ()),
            library_string=(
                "ForceFieldLibrary(\n  present_beads=(CgBead(element_string='P"
                "d', bead_type='m1', bead_class='m', coordination=4), CgBead(e"
                "lement_string='Pb', bead_type='b1', bead_class='b', coordinat"
                "ion=2)),\n  bond_ranges=(TargetBondRange(type1='b1', type2='m"
                "1', element1='Pb', element2='Pd', bond_rs=(1.5 A,), bond_ks=("
                "100000.0 kJ/(nm**2 mol),)),),\n  angle_ranges=(PyramidAngleRa"
                "nge(type1='b1', type2='m1', type3='b1', element1='Pd', elemen"
                "t2='Pb', element3='Pd', angles=(80 deg, 90 deg), angle_ks=(10"
                "0.0 kJ/(mol rad**2),)),),\n  torsion_ranges=(),\n  nonbonded_"
                "ranges=(TargetNonbondedRange(bead_class='m', bead_element='Pd"
                "', sigmas=(1.0 A,), epsilons=(10.0 kJ/mol,), force='custom-ex"
                "cl-vol'), TargetNonbondedRange(bead_class='b', bead_element='"
                "Pb', sigmas=(1.0 A,), epsilons=(10.0 kJ/mol,), force='custom-"
                "excl-vol'))\n)"
            ),
            name=name,
        ),
        # This one should fail with bad units.
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[Pb][Ba][Ag][Ba][Pb]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [0.5, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                    )
                ),
            ),
            forcefield_library=ForceFieldLibrary(
                present_beads=(ag_bead, ba_bead, pb_bead),
                vdw_bond_cutoff=2,
                prefix="testff",
            ),
            bond_ranges=(
                TargetBondRange(
                    type1="a1",
                    type2="b1",
                    element1="Ba",
                    element2="Pb",
                    bond_rs=(1,),
                    bond_ks=(1e5,),
                ),
            ),
            angle_ranges=(
                TargetAngleRange(
                    type1="n1",
                    type2="b1",
                    type3="a1",
                    element1="C",
                    element2="Pb",
                    element3="Ba",
                    angles=(180,),
                    angle_ks=(1e2,),
                ),
            ),
            torsion_ranges=(
                TargetTorsionRange(
                    search_string=("b1", "a1", "c1", "a1"),
                    search_estring=("Pb", "Ba", "Ag", "Ba"),
                    measured_atom_ids=[0, 1, 2, 3],
                    phi0s=(180,),
                    torsion_ks=(50,),
                    torsion_ns=(1,),
                ),
                TargetTorsionRange(
                    search_string=("b1", "a1", "c1", "a1", "b1"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0s=(180,),
                    torsion_ks=(50,),
                    torsion_ns=(1,),
                ),
            ),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    "c",
                    "Ag",
                    epsilons=(10.0,),
                    sigmas=(1.0,),
                    force="custom-excl-vol",
                ),
                TargetNonbondedRange(
                    "b",
                    "Pb",
                    epsilons=(10.0,),
                    sigmas=(1.0,),
                    force="custom-excl-vol",
                ),
            ),
            num_forcefields=0,
            present_bonds=([]),
            present_angles=([]),
            present_nonbondeds=([]),
            present_torsions=([]),
            library_string="",
            name=name,
        ),
        # This one should fail with bad units for pyramid angles -
        # needed separate tests because pyramid angle error will show
        # up before all other exceptions, voiding their tests.
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[Pb][Ba][Ag][Ba][Pb]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [0.5, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                    )
                ),
            ),
            forcefield_library=ForceFieldLibrary(
                present_beads=(ag_bead, ba_bead, pb_bead),
                vdw_bond_cutoff=2,
                prefix="testff",
            ),
            bond_ranges=(),
            angle_ranges=(
                TargetAngleRange(
                    type1="n1",
                    type2="b1",
                    type3="a1",
                    element1="C",
                    element2="Pb",
                    element3="Ba",
                    angles=(180,),
                    angle_ks=(1e2,),
                ),
                PyramidAngleRange(
                    type1="n1",
                    type2="b1",
                    type3="a1",
                    element1="C",
                    element2="Pb",
                    element3="Ba",
                    angles=(180,),
                    angle_ks=(1e2,),
                ),
            ),
            torsion_ranges=(),
            nonbonded_ranges=(),
            num_forcefields=0,
            present_bonds=([]),
            present_angles=([]),
            present_nonbondeds=([]),
            present_torsions=([]),
            library_string="",
            name=name,
        ),
        # Working trigonal planar complex.
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][N][C]",
                position_matrix=np.array(([1, 0, 0], [0.5, 0, 0], [2, 0, 0])),
            ),
            forcefield_library=ForceFieldLibrary(
                present_beads=(c_bead, n_bead),
                vdw_bond_cutoff=2,
                prefix="testuff",
            ),
            bond_ranges=(
                TargetBondRange(
                    type1="c1",
                    type2="n1",
                    element1="Ba",
                    element2="Pb",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    bond_ks=(
                        openmm.unit.Quantity(
                            value=1e5,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                    ),
                ),
            ),
            angle_ranges=(
                TargetCosineAngleRange(
                    type1="c1",
                    type2="n1",
                    type3="c1",
                    element1="C",
                    element2="N",
                    element3="C",
                    ns=(3,),
                    bs=(-1, 1),
                    angle_ks=(
                        openmm.unit.Quantity(
                            value=1e2,
                            unit=kjmol / openmm.unit.radian**2,
                        ),
                    ),
                ),
            ),
            torsion_ranges=(),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    "c",
                    "C",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
                TargetNonbondedRange(
                    "n",
                    "N",
                    epsilons=(openmm.unit.Quantity(value=10.0, unit=kjmol),),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    force="custom-excl-vol",
                ),
            ),
            num_forcefields=2,
            present_bonds=(
                (
                    Bond(
                        atom_names=("C1", "N2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(0), stk.N(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N2", "C3"),
                        atom_ids=(1, 2),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(1), stk.C(2)),
                        force="HarmonicBondForce",
                    ),
                ),
                (
                    Bond(
                        atom_names=("C1", "N2"),
                        atom_ids=(0, 1),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.C(0), stk.N(1)),
                        force="HarmonicBondForce",
                    ),
                    Bond(
                        atom_names=("N2", "C3"),
                        atom_ids=(1, 2),
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=100000.0,
                            unit=kjmol / openmm.unit.nanometer**2,
                        ),
                        atoms=(stk.N(1), stk.C(2)),
                        force="HarmonicBondForce",
                    ),
                ),
            ),
            present_angles=(
                (
                    CosineAngle(
                        atom_names=("C1", "N2", "C3"),
                        atom_ids=(0, 1, 2),
                        n=3,
                        b=-1,
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(0), stk.N(1), stk.C(2)),
                        force="CosinePeriodicAngleForce",
                    ),
                ),
                (
                    CosineAngle(
                        atom_names=("C1", "N2", "C3"),
                        atom_ids=(0, 1, 2),
                        n=3,
                        b=1,
                        angle_k=openmm.unit.Quantity(
                            value=100.0, unit=kjmol / openmm.unit.radian**2
                        ),
                        atoms=(stk.C(0), stk.N(1), stk.C(2)),
                        force="CosinePeriodicAngleForce",
                    ),
                ),
            ),
            present_nonbondeds=(
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
                (
                    Nonbonded(
                        atom_id=0,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=1,
                        bead_class="n",
                        bead_element="N",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                    Nonbonded(
                        atom_id=2,
                        bead_class="c",
                        bead_element="C",
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        epsilon=openmm.unit.Quantity(value=10.0, unit=kjmol),
                        force="custom-excl-vol",
                    ),
                ),
            ),
            present_torsions=((), ()),
            library_string=(
                "ForceFieldLibrary(\n  present_beads=(CgBead(element_string='C"
                "', bead_type='c1', bead_class='c', coordination=2), CgBead(el"
                "ement_string='N', bead_type='n1', bead_class='n', coordinatio"
                "n=2)),\n  bond_ranges=(TargetBondRange(type1='c1', type2='n1'"
                ", element1='Ba', element2='Pb', bond_rs=(1.0 A,), bond_ks=(10"
                "0000.0 kJ/(nm**2 mol),)),),\n  angle_ranges=(TargetCosineAngl"
                "eRange(type1='c1', type2='n1', type3='c1', element1='C', elem"
                "ent2='N', element3='C', ns=(3,), bs=(-1, 1), angle_ks=(100.0 "
                "kJ/(mol rad**2),)),),\n  torsion_ranges=(),\n  nonbonded_rang"
                "es=(TargetNonbondedRange(bead_class='c', bead_element='C', si"
                "gmas=(1.0 A,), epsilons=(10.0 kJ/mol,), force='custom-excl-vo"
                "l'), TargetNonbondedRange(bead_class='n', bead_element='N', s"
                "igmas=(1.0 A,), epsilons=(10.0 kJ/mol,), force='custom-excl-v"
                "ol'))\n)"
            ),
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
