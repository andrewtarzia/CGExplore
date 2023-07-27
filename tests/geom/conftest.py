import numpy as np
import pytest
import stk

import cgexplore

from .case_data import CaseData

# Three tests cases with four atoms with known bond lengths,
# angles and torsions.


@pytest.fixture(
    params=(
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][N][C][C]",
                position_matrix=np.array(
                    (
                        [0, 0, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [4, 0, 0],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(),
            length_dict={
                "C_C": [2.0],
                "C_N": [1.0, 1.0],
            },
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][N][C][C]",
                position_matrix=np.array(
                    (
                        [0, 0, 0],
                        [2, 0, 0],
                        [2, 0, 0],
                        [4, 0, 0],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(),
            length_dict={
                "C_C": [2.0],
                "C_N": [2.0, 0.0],
            },
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][O][C][C]",
                position_matrix=np.array(
                    (
                        [0, 0, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [4, 0, 0],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(),
            length_dict={
                "C_C": [2.0],
                "C_O": [1.0, 1.0],
            },
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
