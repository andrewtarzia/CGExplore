import numpy as np
import pytest
import stk
from cgexplore.angles import TargetAngle
from cgexplore.beads import CgBead
from cgexplore.bonds import TargetBond
from cgexplore.forcefield import ForceField
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
            forcefield=ForceField(
                identifier="testff",
                prefix="testffprefix",
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
                        phi0=180,
                        torsion_k=50,
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_targets=(
                    TargetNonbonded(
                        "c",
                        "Ag",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    TargetNonbonded(
                        "b",
                        "Pb",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
                vdw_bond_cutoff=2,
            ),
            xml_string=(
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
            forcefield=ForceField(
                identifier="testff",
                prefix="testffprefix",
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
                        type1="a",
                        type2="b",
                        element1="Ba",
                        element2="Pb",
                        bond_r=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=1e5,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2,
                        ),
                    )
                ),
                angle_targets=(
                    TargetAngle(
                        type1="b",
                        type2="n",
                        type3="b",
                        element1="Pb",
                        element2="C",
                        element3="Pb",
                        angle=openmm.unit.Quantity(
                            value=70, unit=openmm.unit.degrees
                        ),
                        angle_k=openmm.unit.Quantity(
                            value=1e2,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.radian**2,
                        ),
                    )
                ),
                torsion_targets=(
                    TargetTorsion(
                        search_string=("b", "a", "c", "a", "b"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 3, 4],
                        phi0=180,
                        torsion_k=50,
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_targets=(
                    TargetNonbonded(
                        "c",
                        "Ag",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                    ),
                    TargetNonbonded(
                        "b",
                        "Pb",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                    ),
                ),
                vdw_bond_cutoff=2,
            ),
            xml_string=(
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
