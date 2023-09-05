import numpy as np
import pytest
import stk
from cgexplore.beads import CgBead
from cgexplore.torsions import TargetTorsion, Torsion
from cgexplore.forcefield import Forcefield

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
                identifier="testff",
                output_dir=".",
                prefix="testffprefix",
                present_beads=(
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
                bond_terms=(),
                angle_terms=(),
                torsion_terms=(),
                custom_torsion_terms=(
                    TargetTorsion(
                        search_string=("b", "a", "c", "a", "b"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 3, 4],
                        phi0=180,
                        torsion_k=50,
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_terms=(),
                vdw_bond_cutoff=0,
            ),
            # custom_torsion_set={("b", "a", "c", "a", "b"): (180, 50)},
            present_torsions=(
                Torsion(
                    atom_names=("Pb1", "Ba2", "Ba4", "Pb5"),
                    atom_ids=(0, 1, 3, 4),
                    phi0=180,
                    torsion_k=50,
                    torsion_n=1,
                ),
                # ("Pb1", "Ba2", "Ba4", "Pb5", 0, 1, 3, 4, 50, 1, 180),
            ),
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[Pb][Ba][Ag][Ba][Pb].[Pb][Ba][Ag][Ba][Pb]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [0.5, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                        [1, 1, 2],
                        [1, 0, 2],
                        [0.5, 0, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    )
                ),
            ),
            # custom_torsion_definition=(180, 50),
            # custom_torsion_set={("b", "a", "c", "a", "b"): (180, 50)},
            # present_torsions=(
            #     ("Pb1", "Ba2", "Ba4", "Pb5", 0, 1, 3, 4, 50, 1, 180),
            #     ("Pb6", "Ba7", "Ba9", "Pb10", 5, 6, 8, 9, 50, 1, 180),
            # ),
            force_field=Forcefield(
                identifier="testff",
                output_dir=".",
                prefix="testffprefix",
                present_beads=(
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
                bond_terms=(),
                angle_terms=(),
                torsion_terms=(),
                custom_torsion_terms=(
                    TargetTorsion(
                        search_string=("b", "a", "c", "a", "b"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 3, 4],
                        phi0=180,
                        torsion_k=50,
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_terms=(),
                vdw_bond_cutoff=0,
            ),
            present_torsions=(
                Torsion(
                    atom_names=("Pb1", "Ba2", "Ba4", "Pb5"),
                    atom_ids=(0, 1, 3, 4),
                    phi0=180,
                    torsion_k=50,
                    torsion_n=1,
                ),
                Torsion(
                    atom_names=("Pb6", "Ba7", "Ba9", "Pb10"),
                    atom_ids=(5, 6, 8, 9),
                    phi0=180,
                    torsion_k=50,
                    torsion_n=1,
                ),
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
            # custom_torsion_definition=(180, 50),
            # custom_torsion_set={("b", "a", "d", "a", "b"): (180, 50)},
            # present_torsions=(),
            force_field=Forcefield(
                identifier="testff",
                output_dir=".",
                prefix="testffprefix",
                present_beads=(
                    CgBead(
                        element_string="Ag",
                        bead_type="d",
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
                bond_terms=(),
                angle_terms=(),
                torsion_terms=(),
                custom_torsion_terms=(
                    TargetTorsion(
                        search_string=("b", "a", "c", "a", "b"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 3, 4],
                        phi0=180,
                        torsion_k=50,
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_terms=(),
                vdw_bond_cutoff=0,
            ),
            # custom_torsion_set={("b", "a", "c", "a", "b"): (180, 50)},
            present_torsions=(),
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
            # custom_torsion_definition=(30, 20),
            # custom_torsion_set={("c", "n", "c", "c"): (30, 20)},
            # present_torsions=(
            #     ("C1", "N2", "C3", "C4", 0, 1, 2, 3, 20, 1, 30),
            #     ("C5", "N6", "C7", "C8", 4, 5, 6, 7, 20, 1, 30),
            # ),
            force_field=Forcefield(
                identifier="testff",
                output_dir=".",
                prefix="testffprefix",
                present_beads=(
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
                ),
                bond_terms=(),
                angle_terms=(),
                torsion_terms=(),
                custom_torsion_terms=(
                    TargetTorsion(
                        search_string=("c", "n", "c", "c"),
                        search_estring=(),
                        measured_atom_ids=[0, 1, 2, 3],
                        phi0=30,
                        torsion_k=10,
                        torsion_n=1.0,
                    ),
                ),
                nonbonded_terms=(),
                vdw_bond_cutoff=0,
            ),
            present_torsions=(
                Torsion(
                    atom_names=("C1", "N2", "C3", "C4"),
                    atom_ids=(0, 1, 2, 3),
                    phi0=30,
                    torsion_k=10,
                    torsion_n=1,
                ),
                Torsion(
                    atom_names=("C5", "C6", "N7", "C8"),
                    atom_ids=(4, 5, 6, 7),
                    phi0=30,
                    torsion_k=10,
                    torsion_n=1,
                ),
                Torsion(
                    atom_names=("C6", "N7", "C8", "C9"),
                    atom_ids=(5, 6, 7, 8),
                    phi0=30,
                    torsion_k=10,
                    torsion_n=1,
                ),
            ),
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
