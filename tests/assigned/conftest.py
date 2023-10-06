import numpy as np
import pytest
import stk
from cgexplore.angles import TargetAngle
from cgexplore.beads import CgBead
from cgexplore.bonds import TargetBond
from cgexplore.forcefield import Forcefield
from cgexplore.nonbonded import TargetNonbonded
from cgexplore.torsions import TargetTorsion
from openmm import openmm

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
            force_field=Forcefield(
                identifier=name,
                prefix="assigned_tests",
                present_beads=(
                    CgBead(
                        element_string="Ag",
                        bead_type="c1",
                        bead_class="c",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Ba",
                        bead_type="a1",
                        bead_class="a",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Pb",
                        bead_type="b1",
                        bead_class="b",
                        coordination=2,
                    ),
                ),
                bond_targets=(),
                angle_targets=(),
                torsion_targets=(
                    TargetTorsion(
                        search_string=("b", "a", "c", "a", "b"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 3, 4],
                        phi0=openmm.unit.Quantity(
                            value=180,
                            unit=openmm.unit.degrees,
                        ),
                        torsion_k=openmm.unit.Quantity(
                            value=50,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_targets=(
                    TargetNonbonded(
                        "c",
                        "Ag",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    TargetNonbonded(
                        "a",
                        "Ba",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    TargetNonbonded(
                        "b",
                        "Pb",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                ),
                vdw_bond_cutoff=2,
            ),
            topology_xml_string=(
                """<ForceField>

 <AtomTypes>
  <Type name="b1" class="b" element="Pb" mass="10"/>
  <Type name="a1" class="a" element="Ba" mass="10"/>
  <Type name="c1" class="c" element="Ag" mass="10"/>
 </AtomTypes>

 <Residues>
  <Residue name="ALL">
   <Atom name="0" type="b1"/>
   <Atom name="1" type="a1"/>
   <Atom name="2" type="c1"/>
   <Atom name="3" type="a1"/>
   <Atom name="4" type="b1"/>
   <Bond atomName1="0" atomName2="1"/>
   <Bond atomName1="1" atomName2="2"/>
   <Bond atomName1="2" atomName2="3"/>
   <Bond atomName1="3" atomName2="4"/>
  </Residue>
 </Residues>

</ForceField>
"""
            ),
            name=name,
        ),
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
            force_field=Forcefield(
                identifier=name,
                prefix="assigned_tests",
                present_beads=(
                    CgBead(
                        element_string="Ag",
                        bead_type="c1",
                        bead_class="c",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Ba",
                        bead_type="a1",
                        bead_class="a",
                        coordination=2,
                    ),
                    CgBead(
                        element_string="Pb",
                        bead_type="b1",
                        bead_class="b",
                        coordination=2,
                    ),
                ),
                bond_targets=(
                    TargetBond(
                        class1="a",
                        class2="b",
                        eclass1="Ba",
                        eclass2="Pb",
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=1e5,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2,
                        ),
                    ),
                ),
                angle_targets=(
                    TargetAngle(
                        class1="b",
                        class2="n",
                        class3="b",
                        eclass1="Pb",
                        eclass2="C",
                        eclass3="Pb",
                        angle=openmm.unit.Quantity(
                            value=70, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    ),
                ),
                torsion_targets=(
                    TargetTorsion(
                        search_string=("b", "a", "c", "a", "b"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 3, 4],
                        phi0=openmm.unit.Quantity(
                            value=180,
                            unit=openmm.unit.degrees,
                        ),
                        torsion_k=openmm.unit.Quantity(
                            value=50,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_targets=(
                    TargetNonbonded(
                        "c",
                        "Ag",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    TargetNonbonded(
                        "a",
                        "Ba",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    TargetNonbonded(
                        "b",
                        "Pb",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoule / openmm.unit.mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                ),
                vdw_bond_cutoff=2,
            ),
            topology_xml_string=(
                """<ForceField>

 <AtomTypes>
  <Type name="b1" class="b" element="Pb" mass="10"/>
  <Type name="a1" class="a" element="Ba" mass="10"/>
  <Type name="c1" class="c" element="Ag" mass="10"/>
 </AtomTypes>

 <Residues>
  <Residue name="ALL">
   <Atom name="0" type="b1"/>
   <Atom name="1" type="a1"/>
   <Atom name="2" type="c1"/>
   <Atom name="3" type="a1"/>
   <Atom name="4" type="b1"/>
   <Bond atomName1="0" atomName2="1"/>
   <Bond atomName1="1" atomName2="2"/>
   <Bond atomName1="2" atomName2="3"/>
   <Bond atomName1="3" atomName2="4"/>
  </Residue>
 </Residues>

</ForceField>
"""
            ),
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
