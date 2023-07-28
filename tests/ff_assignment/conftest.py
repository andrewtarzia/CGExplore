import numpy as np
import pytest
import stk

import cgexplore

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
            custom_torsion_definition=(180, 50),
            custom_torsion_set={("b", "a", "c", "a", "b"): (180, 50)},
            present_torsions=(
                ("Pb1", "Ba2", "Ba4", "Pb5", 0, 1, 3, 4, 50, 1, 180),
            ),
            bead_set={
                "c": cgexplore.CgBead(
                    element_string="Ag",
                    bead_type="c",
                    bond_r=1,
                    bond_k=1,
                    angle_centered=1,
                    angle_k=1,
                    sigma=1,
                    epsilon=1,
                    coordination=2,
                ),
                "a": cgexplore.CgBead(
                    element_string="Ba",
                    bead_type="a",
                    bond_r=1,
                    bond_k=1,
                    angle_centered=1,
                    angle_k=1,
                    sigma=1,
                    epsilon=1,
                    coordination=2,
                ),
                "b": cgexplore.CgBead(
                    element_string="Pb",
                    bead_type="b",
                    bond_r=1,
                    bond_k=1,
                    angle_centered=1,
                    angle_k=1,
                    sigma=1,
                    epsilon=1,
                    coordination=2,
                ),
            },
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][N][C][C].[C][N][C][C][C]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                        [1, 1, 2],
                        [1, 0, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    )
                ),
            ),
            custom_torsion_definition=(30, 20),
            custom_torsion_set={("c", "n", "c", "c"): (30, 20)},
            present_torsions=(
                ("C1", "N2", "C3", "C4", 0, 1, 2, 3, 20, 1, 30),
                ("C5", "N6", "C7", "C8", 4, 5, 6, 7, 20, 1, 30),
            ),
            bead_set={
                "c": cgexplore.CgBead(
                    element_string="C",
                    bead_type="c",
                    bond_r=1,
                    bond_k=1,
                    angle_centered=1,
                    angle_k=1,
                    sigma=1,
                    epsilon=1,
                    coordination=2,
                ),
                "n": cgexplore.CgBead(
                    element_string="N",
                    bead_type="n",
                    bond_r=1,
                    bond_k=1,
                    angle_centered=1,
                    angle_k=1,
                    sigma=1,
                    epsilon=1,
                    coordination=2,
                ),
            },
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
