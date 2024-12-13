import pytest
import stk
from openmm import openmm

import cgexplore as cgx

from .case_data import CaseData

bead1 = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)
bead2 = cgx.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)
bead3 = cgx.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)
bead4 = cgx.molecular.CgBead(
    element_string="C",
    bead_class="n",
    bead_type="n",
    coordination=3,
)


@pytest.fixture(
    params=(
        lambda name: CaseData(
            molecule=stk.ConstructedMolecule(
                topology_graph=stk.polymer.Linear(
                    building_blocks=(
                        cgx.molecular.TwoC1Arm(
                            bead=bead1, abead1=bead2
                        ).get_building_block(),
                        cgx.molecular.TwoC1Arm(
                            bead=bead3, abead1=bead4
                        ).get_building_block(),
                    ),
                    num_repeating_units=1,
                    repeating_unit="AB",
                ),
            ),
            forcefield=cgx.forcefields.ForceField(
                identifier="test",
                prefix="opt_tests",
                present_beads=(bead1, bead2, bead3, bead4),
                bond_targets=(
                    cgx.terms.TargetBond(
                        type1="c",
                        type2="a",
                        element1="Ag",
                        element2="Ba",
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
                    cgx.terms.TargetBond(
                        type1="a",
                        type2="n",
                        element1="Ba",
                        element2="C",
                        bond_r=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=1e5,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2,
                        ),
                    ),
                    cgx.terms.TargetBond(
                        type1="b",
                        type2="n",
                        element1="Ba",
                        element2="C",
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
                angle_targets=(),
                torsion_targets=(),
                nonbonded_targets=(
                    cgx.terms.TargetNonbonded(
                        "c",
                        "Ag",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    cgx.terms.TargetNonbonded(
                        "a",
                        "Ba",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    cgx.terms.TargetNonbonded(
                        "b",
                        "Pb",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    cgx.terms.TargetNonbonded(
                        "n",
                        "C",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                ),
                vdw_bond_cutoff=2,
            ),
            known_decomposition={
                "(0, 'HarmonicBondForce')": (1.1249362269979693e-10, "kJ/mol"),
                "(1, 'CustomNonbondedForce')": (
                    0.00010877533350139856,
                    "kJ/mol",
                ),
                "total energy": (0.00010877544599502126, "kJ/mol"),
            },
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.ConstructedMolecule(
                topology_graph=stk.polymer.Linear(
                    building_blocks=(
                        cgx.molecular.TwoC1Arm(
                            bead=bead1, abead1=bead2
                        ).get_building_block(),
                        cgx.molecular.TwoC1Arm(
                            bead=bead3, abead1=bead4
                        ).get_building_block(),
                    ),
                    num_repeating_units=1,
                    repeating_unit="AB",
                ),
            ),
            forcefield=cgx.forcefields.ForceField(
                identifier="test",
                prefix="opt_tests",
                present_beads=(bead1, bead2, bead3, bead4),
                bond_targets=(
                    cgx.terms.TargetBond(
                        type1="c",
                        type2="a",
                        element1="Ag",
                        element2="Ba",
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
                    cgx.terms.TargetBond(
                        type1="a",
                        type2="n",
                        element1="Ba",
                        element2="C",
                        bond_r=openmm.unit.Quantity(
                            value=0.5, unit=openmm.unit.angstrom
                        ),
                        bond_k=openmm.unit.Quantity(
                            value=1e5,
                            unit=openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2,
                        ),
                    ),
                    cgx.terms.TargetBond(
                        type1="b",
                        type2="n",
                        element1="Ba",
                        element2="C",
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
                angle_targets=(),
                torsion_targets=(),
                nonbonded_targets=(
                    cgx.terms.TargetNonbonded(
                        "c",
                        "Ag",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    cgx.terms.TargetNonbonded(
                        "a",
                        "Ba",
                        epsilon=openmm.unit.Quantity(
                            value=10.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=1.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    cgx.terms.TargetNonbonded(
                        "b",
                        "Pb",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                    cgx.terms.TargetNonbonded(
                        "n",
                        "C",
                        epsilon=openmm.unit.Quantity(
                            value=2.0,
                            unit=openmm.unit.kilojoules_per_mole,
                        ),
                        sigma=openmm.unit.Quantity(
                            value=2.0, unit=openmm.unit.angstrom
                        ),
                        force="custom-excl-vol",
                    ),
                ),
                vdw_bond_cutoff=2,
            ),
            known_decomposition={
                "(0, 'HarmonicBondForce')": (2.1239746835015154e-05, "kJ/mol"),
                "(1, 'CustomNonbondedForce')": (0.02951405569911003, "kJ/mol"),
                "total energy": (0.029535295445945048, "kJ/mol"),
            },
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
