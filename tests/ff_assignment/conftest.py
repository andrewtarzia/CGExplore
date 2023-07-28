import numpy as np
import pytest
import stk

import cgexplore

from .case_data import CaseData


@pytest.fixture(
    params=(
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
            torsion_dict={"C_N_C_C": [0.0]},
            custom_torsion_definition=(180, 50),
            custom_torsion_set={("C", "N", "C", "C"): (180, 50)},
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
