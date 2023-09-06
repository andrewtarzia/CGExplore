import numpy as np
import pytest
import stk
from openmm import openmm
from cgexplore.beads import CgBead
from cgexplore.bonds import TargetBondRange
from cgexplore.angles import TargetAngleRange
from cgexplore.nonbonded import TargetNonbondedRange
from cgexplore.torsions import Torsion, TargetTorsionRange
from cgexplore.forcefield import ForceFieldLibrary

from .case_data import CaseData


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
                bead_library=(
                    CgBead(
                        element_string="Ag",
                        bead_type="c",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Ba",
                        bead_type="a",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Pb",
                        bead_type="b",
                        coordination=2,
                    ),
                ),
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
            present_nonbondeds=(
                (
                    '  <Atom type="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="b" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom type="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="b" sigma="0.1" epsilon="10.0"/>',
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
                """ForceFieldLibrary(
  bead_library=(CgBead(element_string='Ag', bead_type='c', coordination=2), CgBead(element_string='Ba', bead_type='a', coordination=2), CgBead(element_string='Pb', bead_type='b', coordination=2)),
  bond_ranges=(TargetBondRange(class1='a', class2='b', eclass1='Ba', eclass2='Pb', bond_rs=(Quantity(value=1.0, unit=angstrom),), bond_ks=(Quantity(value=100000.0, unit=kilojoule/(nanometer**2*mole)),)),),
  angle_ranges=(TargetAngleRange(class1='n', class2='b', class3='a', eclass1='C', eclass2='Pb', eclass3='Ba', angles=(Quantity(value=180, unit=degree),), angle_ks=(Quantity(value=100.0, unit=kilojoule/(mole*radian**2)),)),),
  torsion_ranges=(TargetTorsionRange(search_string=('b', 'a', 'c', 'a', 'b'), search_estring=('Pb', 'Ba', 'Ag', 'Ba', 'Pb'), measured_atom_ids=[0, 1, 3, 4], phi0s=(Quantity(value=180, unit=degree),), torsion_ks=(Quantity(value=50, unit=kilojoule/mole), Quantity(value=0, unit=kilojoule/mole)), torsion_ns=(1,)),),
  nonbonded_ranges=(TargetNonbondedRange(search_string='c', search_estring='Ag', sigmas=(Quantity(value=1.0, unit=angstrom),), epsilons=(Quantity(value=10.0, unit=kilojoule/mole),)), TargetNonbondedRange(search_string='b', search_estring='Pb', sigmas=(Quantity(value=1.0, unit=angstrom),), epsilons=(Quantity(value=10.0, unit=kilojoule/mole),)))
)"""
            ),
            xml_strings=(
                (
                    """<ForceField>

 <HarmonicBondForce>
  <Bond class1="a" class2="b" length="0.1" k="100000.0"/>
 </HarmonicBondForce>

 <HarmonicAngleForce>
  <Angle class1="n" class2="b" class3="a" angle="3.141592653589793" k="100.0"/>
 </HarmonicAngleForce>

 <PeriodicTorsionForce>
 </PeriodicTorsionForce>

 <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">
  <PerParticleParameter name="sigma"/>
  <PerParticleParameter name="epsilon"/>
  <Atom type="c" sigma="0.1" epsilon="10.0"/>
  <Atom type="b" sigma="0.1" epsilon="10.0"/>
 </CustomNonbondedForce>

</ForceField>
"""
                ),
                (
                    """<ForceField>

 <HarmonicBondForce>
  <Bond class1="a" class2="b" length="0.1" k="100000.0"/>
 </HarmonicBondForce>

 <HarmonicAngleForce>
  <Angle class1="n" class2="b" class3="a" angle="3.141592653589793" k="100.0"/>
 </HarmonicAngleForce>

 <PeriodicTorsionForce>
 </PeriodicTorsionForce>

 <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">
  <PerParticleParameter name="sigma"/>
  <PerParticleParameter name="epsilon"/>
  <Atom type="c" sigma="0.1" epsilon="10.0"/>
  <Atom type="b" sigma="0.1" epsilon="10.0"/>
 </CustomNonbondedForce>

</ForceField>
"""
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
                bead_library=(
                    CgBead(
                        element_string="C",
                        bead_type="c",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="N",
                        bead_type="n",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="O",
                        bead_type="o",
                        coordination=2,
                    ),
                ),
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
                    '  <Bond class1="n" class2="c" length="0.1" k="100000.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>',
                ),
                (
                    '  <Bond class1="n" class2="c" length="0.1" k="100000.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>',
                ),
                (
                    '  <Bond class1="n" class2="c" length="0.2" k="100000.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>',
                ),
                (
                    '  <Bond class1="n" class2="c" length="0.2" k="100000.0"/>',
                    '  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>',
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
            present_nonbondeds=(
                (
                    '  <Atom type="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="o" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom type="c" sigma="0.2" epsilon="10.0"/>',
                    '  <Atom type="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="o" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom type="c" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="o" sigma="0.1" epsilon="10.0"/>',
                ),
                (
                    '  <Atom type="c" sigma="0.2" epsilon="10.0"/>',
                    '  <Atom type="n" sigma="0.1" epsilon="10.0"/>',
                    '  <Atom type="o" sigma="0.1" epsilon="10.0"/>',
                ),
            ),
            present_torsions=(
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>',
                ),
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>',
                ),
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>',
                ),
                (
                    '  <Proper  class1="c" class2="n" class3="c" class4'
                    '="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>',
                ),
            ),
            present_custom_torsions=((), (), (), ()),
            library_string=(
                """ForceFieldLibrary(
  bead_library=(CgBead(element_string='C', bead_type='c', coordination=2), CgBead(element_string='N', bead_type='n', coordination=2), CgBead(element_string='O', bead_type='o', coordination=2)),
  bond_ranges=(TargetBondRange(class1='n', class2='c', eclass1='C', eclass2='N', bond_rs=(Quantity(value=1.0, unit=angstrom), Quantity(value=2.0, unit=angstrom)), bond_ks=(Quantity(value=100000.0, unit=kilojoule/(nanometer**2*mole)),)), TargetBondRange(class1='n', class2='o', eclass1='C', eclass2='O', bond_rs=(Quantity(value=2.0, unit=angstrom),), bond_ks=(Quantity(value=100000.0, unit=kilojoule/(nanometer**2*mole)),))),
  angle_ranges=(TargetAngleRange(class1='c', class2='n', class3='c', eclass1='C', eclass2='N', eclass3='C', angles=(Quantity(value=160, unit=degree),), angle_ks=(Quantity(value=100.0, unit=kilojoule/(mole*radian**2)),)),),
  torsion_ranges=(TargetTorsionRange(search_string=('c', 'n', 'c', 'c'), search_estring=('C', 'N', 'C', 'C'), measured_atom_ids=[0, 1, 2, 3], phi0s=(Quantity(value=180, unit=degree),), torsion_ks=(Quantity(value=50, unit=kilojoule/mole),), torsion_ns=(1,)),),
  nonbonded_ranges=(TargetNonbondedRange(search_string='c', search_estring='C', sigmas=(Quantity(value=1.0, unit=angstrom), Quantity(value=2.0, unit=angstrom)), epsilons=(Quantity(value=10.0, unit=kilojoule/mole),)), TargetNonbondedRange(search_string='n', search_estring='N', sigmas=(Quantity(value=1.0, unit=angstrom),), epsilons=(Quantity(value=10.0, unit=kilojoule/mole),)), TargetNonbondedRange(search_string='o', search_estring='O', sigmas=(Quantity(value=1.0, unit=angstrom),), epsilons=(Quantity(value=10.0, unit=kilojoule/mole),)))
)"""
            ),
            xml_strings=(
                (
                    """<ForceField>

 <HarmonicBondForce>
  <Bond class1="n" class2="c" length="0.1" k="100000.0"/>
  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>
 </HarmonicBondForce>

 <HarmonicAngleForce>
  <Angle class1="c" class2="n" class3="c" angle="2.792526803190927" k="100.0"/>
 </HarmonicAngleForce>

 <PeriodicTorsionForce>
  <Proper  class1="c" class2="n" class3="c" class4="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>
 </PeriodicTorsionForce>

 <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">
  <PerParticleParameter name="sigma"/>
  <PerParticleParameter name="epsilon"/>
  <Atom type="c" sigma="0.1" epsilon="10.0"/>
  <Atom type="n" sigma="0.1" epsilon="10.0"/>
  <Atom type="o" sigma="0.1" epsilon="10.0"/>
 </CustomNonbondedForce>

</ForceField>
"""
                ),
                (
                    """<ForceField>

 <HarmonicBondForce>
  <Bond class1="n" class2="c" length="0.1" k="100000.0"/>
  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>
 </HarmonicBondForce>

 <HarmonicAngleForce>
  <Angle class1="c" class2="n" class3="c" angle="2.792526803190927" k="100.0"/>
 </HarmonicAngleForce>

 <PeriodicTorsionForce>
  <Proper  class1="c" class2="n" class3="c" class4="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>
 </PeriodicTorsionForce>

 <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">
  <PerParticleParameter name="sigma"/>
  <PerParticleParameter name="epsilon"/>
  <Atom type="c" sigma="0.2" epsilon="10.0"/>
  <Atom type="n" sigma="0.1" epsilon="10.0"/>
  <Atom type="o" sigma="0.1" epsilon="10.0"/>
 </CustomNonbondedForce>

</ForceField>
"""
                ),
                (
                    """<ForceField>

 <HarmonicBondForce>
  <Bond class1="n" class2="c" length="0.2" k="100000.0"/>
  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>
 </HarmonicBondForce>

 <HarmonicAngleForce>
  <Angle class1="c" class2="n" class3="c" angle="2.792526803190927" k="100.0"/>
 </HarmonicAngleForce>

 <PeriodicTorsionForce>
  <Proper  class1="c" class2="n" class3="c" class4="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>
 </PeriodicTorsionForce>

 <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">
  <PerParticleParameter name="sigma"/>
  <PerParticleParameter name="epsilon"/>
  <Atom type="c" sigma="0.1" epsilon="10.0"/>
  <Atom type="n" sigma="0.1" epsilon="10.0"/>
  <Atom type="o" sigma="0.1" epsilon="10.0"/>
 </CustomNonbondedForce>

</ForceField>
"""
                ),
                (
                    """<ForceField>

 <HarmonicBondForce>
  <Bond class1="n" class2="c" length="0.2" k="100000.0"/>
  <Bond class1="n" class2="o" length="0.2" k="100000.0"/>
 </HarmonicBondForce>

 <HarmonicAngleForce>
  <Angle class1="c" class2="n" class3="c" angle="2.792526803190927" k="100.0"/>
 </HarmonicAngleForce>

 <PeriodicTorsionForce>
  <Proper  class1="c" class2="n" class3="c" class4="c" periodicity1="1" phase1="3.141592653589793" k1="50"/>
 </PeriodicTorsionForce>

 <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">
  <PerParticleParameter name="sigma"/>
  <PerParticleParameter name="epsilon"/>
  <Atom type="c" sigma="0.2" epsilon="10.0"/>
  <Atom type="n" sigma="0.1" epsilon="10.0"/>
  <Atom type="o" sigma="0.1" epsilon="10.0"/>
 </CustomNonbondedForce>

</ForceField>
"""
                ),
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
                bead_library=(
                    CgBead(
                        element_string="Ag",
                        bead_type="c",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Ba",
                        bead_type="a",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Pb",
                        bead_type="b",
                        coordination=2,
                    ),
                ),
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
