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
                smiles="C[Fe](C)(C)(C)(C)C",
                position_matrix=np.array(
                    (
                        [1, 0, 0],
                        [0, 0, 0],
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                        [0, 0, -1],
                    )
                ),
            ),
            shape_dict={
                "HP-6": 33.333,
                "PPY-6": 30.153,
                "OC-6": 0.0,
                "TPR-6": 16.737,
                "JPPY-6": 33.916,
            },
            expected_points=6,
            shape_string="[C].[C].[C].[C].[C].[C]",
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="C[Fe](C)(C)(C)(C)C",
                position_matrix=np.array(
                    (
                        [1, 0.2, 0],
                        [0, 0, 0],
                        [-1.2, 0, 0],
                        [0.3, 1, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                        [0, 0.5, -1],
                    )
                ),
            ),
            shape_dict={
                "HP-6": 25.184,
                "PPY-6": 21.006,
                "OC-6": 3.9,
                "TPR-6": 11.582,
                "JPPY-6": 23.9,
            },
            expected_points=6,
            shape_string="[C].[C].[C].[C].[C].[C]",
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="C[Fe](C)(C)(C)(C)C",
                position_matrix=np.array(
                    (
                        [1, 0, 0],
                        [0, 0, 0],
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                        [0, 0, -1],
                    )
                )
                * 4,
            ),
            shape_dict={
                "HP-6": 33.333,
                "PPY-6": 30.153,
                "OC-6": 0.0,
                "TPR-6": 16.737,
                "JPPY-6": 33.916,
            },
            expected_points=6,
            shape_string="[C].[C].[C].[C].[C].[C]",
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="C[Zn](C)(C)C",
                position_matrix=np.array(
                    (
                        [0.5541, 0.7996, 0.4965],
                        [0, 0, 0],
                        [0.6833, -0.8134, -0.2536],
                        [-0.7782, -0.3735, 0.6692],
                        [-0.4593, 0.3874, -0.9121],
                    )
                ),
            ),
            shape_dict={
                "SP-4": 33.332,
                "T-4": 0.0,
                "SS-4": 7.212,
                "vTBPY-4": 2.287,
            },
            expected_points=4,
            shape_string="[C].[C].[C].[C]",
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="C[Zn](C)(C)C",
                position_matrix=np.array(
                    (
                        [0.5541, 0.7996, 0.4965],
                        [0, 0, 0],
                        [0.6833, -0.2234, -0.2536],
                        [-0.7782, -0.4735, 0.6692],
                        [-0.4593, 0.3874, -0.9121],
                    )
                ),
            ),
            shape_dict={
                "SP-4": 25.71,
                "T-4": 3.947,
                "SS-4": 6.45,
                "vTBPY-4": 1.731,
            },
            expected_points=4,
            shape_string="[C].[C].[C].[C]",
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
