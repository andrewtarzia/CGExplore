import numpy as np
import pytest
import stk
from cgexplore.angles import Angle, PyramidAngleRange, TargetAngleRange
from cgexplore.beads import CgBead
from cgexplore.bonds import TargetBondRange
from cgexplore.forcefield import ForceFieldLibrary
from cgexplore.molecule_construction.topologies import FourC1Arm
from cgexplore.nonbonded import TargetNonbondedRange
from cgexplore.torsions import TargetTorsionRange, Torsion
from openmm import openmm

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
            force_field_library=ForceFieldLibrary(
                bead_library=(ag_bead, ba_bead, pb_bead),
                vdw_bond_cutoff=2,
            ),
            bond_ranges=(
                TargetBondRange(
                    class1="a",
                    class2="b",
                    eclass1="Ba",
                    eclass2="Pb",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
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
                TargetAngleRange(
                    class1="n",
                    class2="b",
                    class3="a",
                    eclass1="C",
                    eclass2="Pb",
                    eclass3="Ba",
                    angles=(
                        openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
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
                    search_string=("b", "a", "c", "a", "b"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0s=(
                        openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                    ),
                    torsion_ks=(
                        openmm.unit.Quantity(
                            value=50,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        openmm.unit.Quantity(
                            value=0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    torsion_ns=(1,),
                ),
            ),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    "c",
                    "Ag",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
                TargetNonbondedRange(
                    "b",
                    "Pb",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
            ),
            num_forcefields=2,
            present_bonds=(
                ('  <Bond class1="a" class2="b" length="0.1" k="100000.0"/>',),
                ('  <Bond class1="a" class2="b" length="0.1" k="100000.0"/>',),
            ),
            present_angles=(
                (
                    (
                        '  <Angle class1="n" class2="b" class3="a" angle='
                        '"3.141592653589793" k="100.0"/>'
                    ),
                ),
                (
                    (
                        '  <Angle class1="n" class2="b" class3="a" angle='
                        '"3.141592653589793" k="100.0"/>'
                    ),
                ),
            ),
            present_custom_angles=((), ()),
            present_nonbondeds=(
                (
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="b" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="b" sigma="0.1" epsilon="10.0"/>',
                ),
            ),
            present_torsions=((), ()),
            present_custom_torsions=(
                (
                    Torsion(
                        atom_names=("Pb1", "Ba2", "Ba4", "Pb5"),
                        atom_ids=(0, 1, 3, 4),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        torsion_k=openmm.unit.Quantity(
                            value=50,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        torsion_n=1,
                    ),
                ),
                (
                    Torsion(
                        atom_names=("Pb1", "Ba2", "Ba4", "Pb5"),
                        atom_ids=(0, 1, 3, 4),
                        phi0=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        torsion_k=openmm.unit.Quantity(
                            value=0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        torsion_n=1,
                    ),
                ),
            ),
            library_string=(
                "ForceFieldLibrary(\n"
                "  bead_library=(CgBead(element_string='Ag', bead_type="
                "'c1', bead_class='c', coordination=2), CgBead(element_"
                "string='Ba', bead_type='a1', bead_class='a', coordinat"
                "ion=2), CgBead(element_string='Pb', bead_type='b1', be"
                "ad_class='b', coordination=2)),\n"
                "  bond_ranges=(TargetBondRange(class1='a', class2='b',"
                " eclass1='Ba', eclass2='Pb', bond_rs=(Quantity(value=1"
                ".0, unit=angstrom),), bond_ks=(Quantity(value=100000.0"
                ", unit=kilojoule/(nanometer**2*mole)),)),),\n"
                "  angle_ranges=(TargetAngleRange(class1='n', class2='b"
                "', class3='a', eclass1='C', eclass2='Pb', eclass3='Ba'"
                ", angles=(Quantity(value=180, unit=degree),), angle_ks"
                "=(Quantity(value=100.0, unit=kilojoule/(mole*radian**2"
                ")),)),),\n"
                "  torsion_ranges=(TargetTorsionRange(search_string=('b"
                "', 'a', 'c', 'a', 'b'), search_estring=('Pb', 'Ba', 'A"
                "g', 'Ba', 'Pb'), measured_atom_ids=[0, 1, 3, 4], phi0s"
                "=(Quantity(value=180, unit=degree),), torsion_ks=(Quan"
                "tity(value=50, unit=kilojoule/mole), Quantity(value=0,"
                " unit=kilojoule/mole)), torsion_ns=(1,)),),\n"
                "  nonbonded_ranges=(TargetNonbondedRange(bead_class='c"
                "', bead_element='Ag', sigmas=(Quantity(value=1.0, unit"
                "=angstrom),), epsilons=(Quantity(value=10.0, unit=kilo"
                "joule/mole),)), TargetNonbondedRange(bead_class='b', b"
                "ead_element='Pb', sigmas=(Quantity(value=1.0, unit=ang"
                "strom),), epsilons=(Quantity(value=10.0, unit=kilojoul"
                "e/mole),)))\n"
                ")"
            ),
            xml_strings=(
                (
                    "<ForceField>\n"
                    "\n"
                    " <HarmonicBondForce>\n"
                    '  <Bond class1="a" class2="b" length="0.1" k="1000'
                    '00.0"/>\n'
                    " </HarmonicBondForce>\n"
                    "\n"
                    " <HarmonicAngleForce>\n"
                    '  <Angle class1="n" class2="b" class3="a" angle="3'
                    '.141592653589793" k="100.0"/>\n'
                    " </HarmonicAngleForce>\n"
                    "\n"
                    " <PeriodicTorsionForce>\n"
                    " </PeriodicTorsionForce>\n"
                    "\n"
                    ' <CustomNonbondedForce energy="sqrt(epsilon1*epsil'
                    'on2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                    '  <PerParticleParameter name="sigma"/>\n'
                    '  <PerParticleParameter name="epsilon"/>\n'
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="b" sigma="0.1" epsilon="10.0"/>\n'
                    " </CustomNonbondedForce>\n"
                    "\n"
                    "</ForceField>\n"
                ),
                (
                    "<ForceField>\n"
                    "\n"
                    " <HarmonicBondForce>\n"
                    '  <Bond class1="a" class2="b" length="0.1" k="1000'
                    '00.0"/>\n'
                    " </HarmonicBondForce>\n"
                    "\n"
                    " <HarmonicAngleForce>\n"
                    '  <Angle class1="n" class2="b" class3="a" angle="3'
                    '.141592653589793" k="100.0"/>\n'
                    " </HarmonicAngleForce>\n"
                    "\n"
                    " <PeriodicTorsionForce>\n"
                    " </PeriodicTorsionForce>\n"
                    "\n"
                    ' <CustomNonbondedForce energy="sqrt(epsilon1*epsil'
                    'on2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                    '  <PerParticleParameter name="sigma"/>\n'
                    '  <PerParticleParameter name="epsilon"/>\n'
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="b" sigma="0.1" epsilon="10.0"/>\n'
                    " </CustomNonbondedForce>\n"
                    "\n"
                    "</ForceField>\n"
                ),
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
            force_field_library=ForceFieldLibrary(
                bead_library=(c_bead, n_bead, o_bead),
                vdw_bond_cutoff=2,
            ),
            bond_ranges=(
                TargetBondRange(
                    class1="n",
                    class2="c",
                    eclass1="C",
                    eclass2="N",
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
                            value=1e5,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2,
                        ),
                    ),
                ),
                TargetBondRange(
                    class1="n",
                    class2="o",
                    eclass1="C",
                    eclass2="O",
                    bond_rs=(
                        openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
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
                TargetAngleRange(
                    class1="c",
                    class2="n",
                    class3="c",
                    eclass1="C",
                    eclass2="N",
                    eclass3="C",
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
                    search_string=("c", "n", "c", "c"),
                    search_estring=("C", "N", "C", "C"),
                    measured_atom_ids=[0, 1, 2, 3],
                    phi0s=(
                        openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                    ),
                    torsion_ks=(
                        openmm.unit.Quantity(
                            value=50,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    torsion_ns=(1,),
                ),
            ),
            nonbonded_ranges=(
                TargetNonbondedRange(
                    "c",
                    "C",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
                TargetNonbondedRange(
                    "n",
                    "N",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
                TargetNonbondedRange(
                    "o",
                    "O",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
            ),
            num_forcefields=4,
            present_bonds=(
                (
                    '  <Bond class1="n" class2="c" length="0.1" k="10000'
                    '0.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="10000'
                    '0.0"/>',
                ),
                (
                    '  <Bond class1="n" class2="c" length="0.1" k="10000'
                    '0.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="10000'
                    '0.0"/>',
                ),
                (
                    '  <Bond class1="n" class2="c" length="0.2" k="10000'
                    '0.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="10000'
                    '0.0"/>',
                ),
                (
                    '  <Bond class1="n" class2="c" length="0.2" k="10000'
                    '0.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="10000'
                    '0.0"/>',
                ),
            ),
            present_angles=(
                (
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>',
                ),
                (
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>',
                ),
                (
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>',
                ),
                (
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>',
                ),
            ),
            present_custom_angles=((), (), (), ()),
            present_nonbondeds=(
                (
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom class="c" sigma="0.2" epsilon="10.0"/>',
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom class="c" sigma="0.2" epsilon="10.0"/>',
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>',
                ),
            ),
            present_torsions=(
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>',
                ),
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>',
                ),
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>',
                ),
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>',
                ),
            ),
            present_custom_torsions=((), (), (), ()),
            library_string=(
                "ForceFieldLibrary(\n"
                "  bead_library=(CgBead(element_string='C', bead_type='"
                "c1', bead_class='c', coordination=2), CgBead(element_s"
                "tring='N', bead_type='n1', bead_class='n', coordinatio"
                "n=2), CgBead(element_string='O', bead_type='o1', bead_"
                "class='o', coordination=2)),\n"
                "  bond_ranges=(TargetBondRange(class1='n', class2='c',"
                " eclass1='C', eclass2='N', bond_rs=(Quantity(value=1.0"
                ", unit=angstrom), Quantity(value=2.0, unit=angstrom)),"
                " bond_ks=(Quantity(value=100000.0, unit=kilojoule/(nan"
                "ometer**2*mole)),)), TargetBondRange(class1='n', class"
                "2='o', eclass1='C', eclass2='O', bond_rs=(Quantity(val"
                "ue=2.0, unit=angstrom),), bond_ks=(Quantity(value=1000"
                "00.0, unit=kilojoule/(nanometer**2*mole)),))),\n"
                "  angle_ranges=(TargetAngleRange(class1='c', class2='n"
                "', class3='c', eclass1='C', eclass2='N', eclass3='C', "
                "angles=(Quantity(value=160, unit=degree),), angle_ks=("
                "Quantity(value=100.0, unit=kilojoule/(mole*radian**2))"
                ",)),),\n"
                "  torsion_ranges=(TargetTorsionRange(search_string=('c"
                "', 'n', 'c', 'c'), search_estring=('C', 'N', 'C', 'C')"
                ", measured_atom_ids=[0, 1, 2, 3], phi0s=(Quantity(valu"
                "e=180, unit=degree),), torsion_ks=(Quantity(value=50, "
                "unit=kilojoule/mole),), torsion_ns=(1,)),),\n"
                "  nonbonded_ranges=(TargetNonbondedRange(bead_class='c"
                "', bead_element='C', sigmas=(Quantity(value=1.0, unit="
                "angstrom), Quantity(value=2.0, unit=angstrom)), epsilo"
                "ns=(Quantity(value=10.0, unit=kilojoule/mole),)), Targ"
                "etNonbondedRange(bead_class='n', bead_element='N', sig"
                "mas=(Quantity(value=1.0, unit=angstrom),), epsilons=(Q"
                "uantity(value=10.0, unit=kilojoule/mole),)), TargetNon"
                "bondedRange(bead_class='o', bead_element='O', sigmas=("
                "Quantity(value=1.0, unit=angstrom),), epsilons=(Quanti"
                "ty(value=10.0, unit=kilojoule/mole),)))\n"
                ")"
            ),
            xml_strings=(
                (
                    "<ForceField>\n"
                    "\n"
                    " <HarmonicBondForce>\n"
                    '  <Bond class1="n" class2="c" length="0.1" k="1000'
                    '00.0"/>\n'
                    '  <Bond class1="n" class2="o" length="0.2" k="1000'
                    '00.0"/>\n'
                    " </HarmonicBondForce>\n"
                    "\n"
                    " <HarmonicAngleForce>\n"
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>\n'
                    " </HarmonicAngleForce>\n"
                    "\n"
                    " <PeriodicTorsionForce>\n"
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>\n'
                    " </PeriodicTorsionForce>\n"
                    "\n"
                    ' <CustomNonbondedForce energy="sqrt(epsilon1*epsil'
                    'on2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                    '  <PerParticleParameter name="sigma"/>\n'
                    '  <PerParticleParameter name="epsilon"/>\n'
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>\n'
                    " </CustomNonbondedForce>\n"
                    "\n"
                    "</ForceField>\n"
                ),
                (
                    "<ForceField>\n"
                    "\n"
                    " <HarmonicBondForce>\n"
                    '  <Bond class1="n" class2="c" length="0.1" k="1000'
                    '00.0"/>\n'
                    '  <Bond class1="n" class2="o" length="0.2" k="1000'
                    '00.0"/>\n'
                    " </HarmonicBondForce>\n"
                    "\n"
                    " <HarmonicAngleForce>\n"
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>\n'
                    " </HarmonicAngleForce>\n"
                    "\n"
                    " <PeriodicTorsionForce>\n"
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>\n'
                    " </PeriodicTorsionForce>\n"
                    "\n"
                    ' <CustomNonbondedForce energy="sqrt(epsilon1*epsil'
                    'on2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                    '  <PerParticleParameter name="sigma"/>\n'
                    '  <PerParticleParameter name="epsilon"/>\n'
                    '  <Atom class="c" sigma="0.2" epsilon="10.0"/>\n'
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>\n'
                    " </CustomNonbondedForce>\n"
                    "\n"
                    "</ForceField>\n"
                ),
                (
                    "<ForceField>\n"
                    "\n"
                    " <HarmonicBondForce>\n"
                    '  <Bond class1="n" class2="c" length="0.2" k="1000'
                    '00.0"/>\n'
                    '  <Bond class1="n" class2="o" length="0.2" k="1000'
                    '00.0"/>\n'
                    " </HarmonicBondForce>\n"
                    "\n"
                    " <HarmonicAngleForce>\n"
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>\n'
                    " </HarmonicAngleForce>\n"
                    "\n"
                    " <PeriodicTorsionForce>\n"
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>\n'
                    " </PeriodicTorsionForce>\n"
                    "\n"
                    ' <CustomNonbondedForce energy="sqrt(epsilon1*epsi'
                    'lon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                    '  <PerParticleParameter name="sigma"/>\n'
                    '  <PerParticleParameter name="epsilon"/>\n'
                    '  <Atom class="c" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>\n'
                    " </CustomNonbondedForce>\n"
                    "\n"
                    "</ForceField>\n"
                ),
                (
                    "<ForceField>\n"
                    "\n"
                    " <HarmonicBondForce>\n"
                    '  <Bond class1="n" class2="c" length="0.2" k="1000'
                    '00.0"/>\n'
                    '  <Bond class1="n" class2="o" length="0.2" k="1000'
                    '00.0"/>\n'
                    " </HarmonicBondForce>\n"
                    "\n"
                    " <HarmonicAngleForce>\n"
                    '  <Angle class1="c" class2="n" class3="c" angle="2'
                    '.792526803190927" k="100.0"/>\n'
                    " </HarmonicAngleForce>\n"
                    "\n"
                    " <PeriodicTorsionForce>\n"
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k'
                    '1="50"/>\n'
                    " </PeriodicTorsionForce>\n"
                    "\n"
                    ' <CustomNonbondedForce energy="sqrt(epsilon1*epsil'
                    'on2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                    '  <PerParticleParameter name="sigma"/>\n'
                    '  <PerParticleParameter name="epsilon"/>\n'
                    '  <Atom class="c" sigma="0.2" epsilon="10.0"/>\n'
                    '  <Atom class="n" sigma="0.1" epsilon="10.0"/>\n'
                    '  <Atom class="o" sigma="0.1" epsilon="10.0"/>\n'
                    " </CustomNonbondedForce>\n"
                    "\n"
                    "</ForceField>\n"
                ),
            ),
            name=name,
        ),
        lambda name: CaseData(
            molecule=FourC1Arm(
                bead=pd_bead, abead1=pb_bead
            ).get_building_block(),
            force_field_library=ForceFieldLibrary(
                bead_library=(pd_bead, pb_bead),
                vdw_bond_cutoff=2,
            ),
            bond_ranges=(
                TargetBondRange(
                    class1="b",
                    class2="m",
                    eclass1="Pb",
                    eclass2="Pd",
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
                    class1="b",
                    class2="m",
                    class3="b",
                    eclass1="Pd",
                    eclass2="Pb",
                    eclass3="Pd",
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
                    bead_element="Pb",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
                TargetNonbondedRange(
                    bead_class="b",
                    bead_element="Pb",
                    epsilons=(
                        openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                    ),
                    sigmas=(
                        openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
            ),
            num_forcefields=2,
            present_bonds=(
                (
                    '  <Bond class1="b" class2="m" length="0.1500000000'
                    '0000002" k="100000.0"/>',
                ),
                (
                    '  <Bond class1="b" class2="m" length="0.1500000000'
                    '0000002" k="100000.0"/>',
                ),
            ),
            present_angles=((), ()),
            present_custom_angles=(
                (
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb3"),
                        atom_ids=(1, 0, 2),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb4"),
                        atom_ids=(1, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=130.75, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb5"),
                        atom_ids=(1, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb4"),
                        atom_ids=(2, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb5"),
                        atom_ids=(2, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=130.75, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb4", "Pd1", "Pb5"),
                        atom_ids=(3, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=80, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
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
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb4"),
                        atom_ids=(1, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb2", "Pd1", "Pb5"),
                        atom_ids=(1, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb4"),
                        atom_ids=(2, 0, 3),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb3", "Pd1", "Pb5"),
                        atom_ids=(2, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=180, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                    Angle(
                        atoms=None,
                        atom_names=("Pb4", "Pd1", "Pb5"),
                        atom_ids=(3, 0, 4),
                        angle=openmm.unit.Quantity(
                            value=90, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                ),
            ),
            present_nonbondeds=(
                (
                    '  <Atom class="m" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="b" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom class="m" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom class="b" sigma="0.1" epsilon="10.0"/>',
                ),
            ),
            present_torsions=((), ()),
            present_custom_torsions=((), ()),
            library_string=(
                "ForceFieldLibrary(\n"
                "  bead_library=(CgBead(element_string='Pd', bead_type="
                "'m1', bead_class='m', coordination=4), CgBead(element_"
                "string='Pb', bead_type='b1', bead_class='b', coordinat"
                "ion=2)),\n"
                "  bond_ranges=(TargetBondRange(class1='b', class2='m',"
                " eclass1='Pb', eclass2='Pd', bond_rs=(Quantity(value=1"
                ".5, unit=angstrom),), bond_ks=(Quantity(value=100000.0"
                ", unit=kilojoule/(nanometer**2*mole)),)),),\n"
                "  angle_ranges=(PyramidAngleRange(class1='b', class2='"
                "m', class3='b', eclass1='Pd', eclass2='Pb', eclass3='P"
                "d', angles=(Quantity(value=80, unit=degree), Quantity("
                "value=90, unit=degree)), angle_ks=(Quantity(value=100."
                "0, unit=kilojoule/(mole*radian**2)),)),),\n"
                "  torsion_ranges=(),\n"
                "  nonbonded_ranges=(TargetNonbondedRange(bead_class='"
                "m', bead_element='Pb', sigmas=(Quantity(value=1.0, uni"
                "t=angstrom),), epsilons=(Quantity(value=10.0, unit=kil"
                "ojoule/mole),)), TargetNonbondedRange(bead_class='b', "
                "bead_element='Pb', sigmas=(Quantity(value=1.0, unit=an"
                "gstrom),), epsilons=(Quantity(value=10.0, unit=kilojou"
                "le/mole),)))\n"
                ")"
            ),
            xml_strings=(
                "<ForceField>\n"
                "\n"
                " <HarmonicBondForce>\n"
                '  <Bond class1="b" class2="m" length="0.15000000000000'
                '002" k="100000.0"/>\n'
                " </HarmonicBondForce>\n"
                "\n"
                " <HarmonicAngleForce>\n"
                " </HarmonicAngleForce>\n"
                "\n"
                " <PeriodicTorsionForce>\n"
                " </PeriodicTorsionForce>\n"
                "\n"
                ' <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)'
                '*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                '  <PerParticleParameter name="sigma"/>\n'
                '  <PerParticleParameter name="epsilon"/>\n'
                '  <Atom class="m" sigma="0.1" epsilon="10.0"/>\n'
                '  <Atom class="b" sigma="0.1" epsilon="10.0"/>\n'
                " </CustomNonbondedForce>\n"
                "\n"
                "</ForceField>\n",
                "<ForceField>\n"
                "\n"
                " <HarmonicBondForce>\n"
                '  <Bond class1="b" class2="m" length="0.15000000000000'
                '002" k="100000.0"/>\n'
                " </HarmonicBondForce>\n"
                "\n"
                " <HarmonicAngleForce>\n"
                " </HarmonicAngleForce>\n"
                "\n"
                " <PeriodicTorsionForce>\n"
                " </PeriodicTorsionForce>\n"
                "\n"
                ' <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)'
                '*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">\n'
                '  <PerParticleParameter name="sigma"/>\n'
                '  <PerParticleParameter name="epsilon"/>\n'
                '  <Atom class="m" sigma="0.1" epsilon="10.0"/>\n'
                '  <Atom class="b" sigma="0.1" epsilon="10.0"/>\n'
                " </CustomNonbondedForce>\n"
                "\n"
                "</ForceField>\n",
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
            force_field_library=ForceFieldLibrary(
                bead_library=(ag_bead, ba_bead, pb_bead),
                vdw_bond_cutoff=2,
            ),
            bond_ranges=(
                TargetBondRange(
                    class1="a",
                    class2="b",
                    eclass1="Ba",
                    eclass2="Pb",
                    bond_rs=(1,),
                    bond_ks=(1e5,),
                ),
            ),
            angle_ranges=(
                TargetAngleRange(
                    class1="n",
                    class2="b",
                    class3="a",
                    eclass1="C",
                    eclass2="Pb",
                    eclass3="Ba",
                    angles=(180,),
                    angle_ks=(1e2,),
                ),
            ),
            torsion_ranges=(
                TargetTorsionRange(
                    search_string=("b", "a", "c", "a"),
                    search_estring=("Pb", "Ba", "Ag", "Ba"),
                    measured_atom_ids=[0, 1, 2, 3],
                    phi0s=(180,),
                    torsion_ks=(50,),
                    torsion_ns=(1,),
                ),
                TargetTorsionRange(
                    search_string=("b", "a", "c", "a", "b"),
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
                ),
                TargetNonbondedRange(
                    "b",
                    "Pb",
                    epsilons=(10.0,),
                    sigmas=(1.0,),
                ),
            ),
            num_forcefields=0,
            present_bonds=([]),
            present_angles=([]),
            present_custom_angles=([]),
            present_nonbondeds=([]),
            present_torsions=([]),
            present_custom_torsions=([]),
            library_string="",
            xml_strings=(),
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
            force_field_library=ForceFieldLibrary(
                bead_library=(ag_bead, ba_bead, pb_bead),
                vdw_bond_cutoff=2,
            ),
            bond_ranges=(),
            angle_ranges=(
                TargetAngleRange(
                    class1="n",
                    class2="b",
                    class3="a",
                    eclass1="C",
                    eclass2="Pb",
                    eclass3="Ba",
                    angles=(180,),
                    angle_ks=(1e2,),
                ),
                PyramidAngleRange(
                    class1="n",
                    class2="b",
                    class3="a",
                    eclass1="C",
                    eclass2="Pb",
                    eclass3="Ba",
                    angles=(180,),
                    angle_ks=(1e2,),
                ),
            ),
            torsion_ranges=(),
            nonbonded_ranges=(),
            num_forcefields=0,
            present_bonds=([]),
            present_angles=([]),
            present_custom_angles=([]),
            present_nonbondeds=([]),
            present_torsions=([]),
            present_custom_torsions=([]),
            library_string="",
            xml_strings=(),
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
