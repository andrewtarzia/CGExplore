import cgexplore
import numpy as np
import pytest
import stk

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
                        [1, 1, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [2, 2, 0],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(
                torsion_set=("C", "N", "C", "C")
            ),
            length_dict={
                "C_C": [2.0],
                "C_N": [1.0, 1.0],
            },
            angle_dict={
                "C_N_C": [90.0],
                "N_C_C": [90.0],
            },
            torsion_dict={"C_N_C_C": [0.0]},
            radius_gyration=0.968,
            max_diam=2.236,
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][N][C][C]",
                position_matrix=np.array(
                    (
                        [0, 0, 0],
                        [1.5, 0, 0],
                        [2, 0, 0],
                        [4, 0, 0],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(),
            length_dict={
                "C_C": [2.0],
                "C_N": [1.5, 0.5],
            },
            angle_dict={
                "C_N_C": [180.0],
                "N_C_C": [180.0],
            },
            torsion_dict={},
            radius_gyration=1.4306,
            max_diam=4.0,
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][O][C][C]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [
                            2,
                            np.cos(np.radians(30)) * 2,
                            np.sin(np.radians(30)) * 2,
                        ],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(torsion_set="COCC"),
            length_dict={
                "C_C": [2.0],
                "C_O": [1.0, 1.0],
            },
            angle_dict={
                "C_O_C": [90.0],
                "O_C_C": [90.0],
            },
            torsion_dict={"C_N_C_C": [30.0]},
            radius_gyration=0.9854,
            max_diam=2.2361,
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][O][C][C]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [
                            2,
                            np.cos(-np.radians(30)) * 2,
                            np.sin(-np.radians(30)) * 2,
                        ],
                    )
                ),
            ),
            geommeasure=cgexplore.GeomMeasure(torsion_set="COCC"),
            length_dict={
                "C_C": [2.0],
                "C_O": [1.0, 1.0],
            },
            angle_dict={
                "C_O_C": [90.0],
                "O_C_C": [90.0],
            },
            torsion_dict={"C_N_C_C": [-30.0]},
            radius_gyration=0.9854,
            max_diam=2.2361,
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
