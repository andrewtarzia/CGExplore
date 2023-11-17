import numpy as np
import pytest
import stk
from cgexplore.geom import GeomMeasure
from cgexplore.torsions import TargetTorsion

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
            geommeasure=GeomMeasure(
                target_torsions=(
                    TargetTorsion(
                        search_string=(),
                        search_estring=("C", "N", "C", "C"),
                        measured_atom_ids=(0, 1, 2, 3),
                        phi0=0,
                        torsion_k=0,
                        torsion_n=0,
                    ),
                )
            ),
            length_dict={
                ("C", "C"): [2.0],
                ("C", "N"): [1.0, 1.0],
            },
            angle_dict={
                ("C", "N", "C"): [90.0],
                ("N", "C", "C"): [90.0],
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
            geommeasure=GeomMeasure(),
            length_dict={
                ("C", "C"): [2.0],
                ("C", "N"): [1.5, 0.5],
            },
            angle_dict={
                ("C", "N", "C"): [180.0],
                ("N", "C", "C"): [180.0],
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
            geommeasure=GeomMeasure(
                target_torsions=(
                    TargetTorsion(
                        search_string=(),
                        search_estring=("C", "O", "C", "C"),
                        measured_atom_ids=(0, 1, 2, 3),
                        phi0=0,
                        torsion_k=0,
                        torsion_n=0,
                    ),
                    # target_torsions="COCC"),
                )
            ),
            length_dict={
                ("C", "C"): [2.0],
                ("C", "O"): [1.0, 1.0],
            },
            angle_dict={
                ("C", "O", "C"): [90.0],
                ("O", "C", "C"): [90.0],
            },
            torsion_dict={"C_O_C_C": [30.0]},
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
            geommeasure=GeomMeasure(
                target_torsions=(
                    TargetTorsion(
                        search_string=(),
                        search_estring=("C", "O", "C", "C"),
                        measured_atom_ids=(0, 1, 2, 3),
                        phi0=0,
                        torsion_k=0,
                        torsion_n=0,
                    ),
                    # target_torsions="COCC"),
                )
            ),
            length_dict={
                ("C", "C"): [2.0],
                ("C", "O"): [1.0, 1.0],
            },
            angle_dict={
                ("C", "O", "C"): [90.0],
                ("O", "C", "C"): [90.0],
            },
            torsion_dict={"C_O_C_C": [-30.0]},
            radius_gyration=0.9854,
            max_diam=2.2361,
            name=name,
        ),
        lambda name: CaseData(
            molecule=stk.BuildingBlock(
                smiles="[C][C][O][C][C]",
                position_matrix=np.array(
                    (
                        [1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [-1, 0, 0],
                        [-1, 1, 0],
                    )
                ),
            ),
            geommeasure=GeomMeasure(
                target_torsions=(
                    TargetTorsion(
                        search_string=(),
                        search_estring=("C", "O", "C", "C"),
                        measured_atom_ids=(0, 1, 2, 3),
                        phi0=0,
                        torsion_k=0,
                        torsion_n=0,
                    ),
                )
            ),
            length_dict={
                ("C", "C"): [1.0, 1.0],
                ("C", "O"): [1.0, 1.0],
            },
            angle_dict={
                ("C", "O", "C"): [180.0],
                ("C", "C", "O"): [90.0, 90.0],
            },
            torsion_dict={"C_C_O_C": [0.0, 0.0]},
            radius_gyration=1.0198,
            max_diam=2.2361,
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
