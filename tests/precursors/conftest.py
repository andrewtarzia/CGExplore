import numpy as np
import pytest
from cgexplore.beads import CgBead
from cgexplore.molecule_construction import (
    FourC0Arm,
    FourC1Arm,
    ThreeC0Arm,
    ThreeC1Arm,
    ThreeC2Arm,
    TwoC0Arm,
    TwoC1Arm,
    TwoC2Arm,
    TwoC3Arm,
)

from .case_data import CaseData

# Three tests cases with four atoms with known bond lengths,
# angles and torsions.


@pytest.fixture(
    params=(
        lambda name: CaseData(
            precursor=FourC0Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=4,
                ),
            ),
            precursor_name="4C0c",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=4,
                )
            },
            smiles="Br[Ag](Br)(Br)Br",
            position_matrix=np.array(
                [
                    [-2.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -2.0, -1.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            num_fgs=4,
            name=name,
        ),
        lambda name: CaseData(
            precursor=FourC1Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=4,
                ),
                abead1=CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
            ),
            precursor_name="4C1cp",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=4,
                ),
                "p": CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
            },
            smiles="[P][Ag]([P])([P])[P]",
            position_matrix=np.array(
                [
                    [0, 0, 1.6],
                    [6, 0, 0],
                    [0, 6, 0],
                    [-6, 0, 0],
                    [0, -6, 0],
                ]
            ),
            num_fgs=4,
            name=name,
        ),
        lambda name: CaseData(
            precursor=ThreeC0Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=3,
                ),
            ),
            precursor_name="3C0c",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=3,
                ),
            },
            smiles="Br[Ag](Br)Br",
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
            num_fgs=3,
            name=name,
        ),
        lambda name: CaseData(
            precursor=ThreeC1Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=3,
                ),
                abead1=CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
            ),
            precursor_name="3C1cp",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=3,
                ),
                "p": CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
            },
            smiles="[P][Ag]([P])[P]",
            position_matrix=np.array(
                [
                    [1.160028, 0.0, 0.0704211],
                    [3.0, 0.0, 0.0],
                    [-1.5, 2.59807621, 0.0],
                    [-1.5, -2.59807621, 0.0],
                ]
            ),
            num_fgs=3,
            name=name,
        ),
        lambda name: CaseData(
            precursor=ThreeC2Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=3,
                ),
                abead1=CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
                abead2=CgBead(
                    element_string="N",
                    bead_type="n",
                    coordination=2,
                ),
            ),
            precursor_name="3C2cpn",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=3,
                ),
                "p": CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
                "n": CgBead(
                    element_string="N",
                    bead_type="n",
                    coordination=2,
                ),
            },
            smiles="[N][P][Ag]([P][N])[P][N]",
            position_matrix=np.array(
                [
                    [1.1, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                    [-0.75, 1.29903811, 0.0],
                    [-2.25, 3.89711432, 0.0],
                    [-0.75, -1.29903811, 0.0],
                    [-2.25, -3.89711432, 0.0],
                ]
            ),
            num_fgs=3,
            name=name,
        ),
        lambda name: CaseData(
            precursor=TwoC0Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
            ),
            precursor_name="2C0c",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
            },
            smiles="Br[Ag]Br",
            position_matrix=np.array(
                [
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                ]
            ),
            num_fgs=2,
            name=name,
        ),
        lambda name: CaseData(
            precursor=TwoC1Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
                abead1=CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
            ),
            precursor_name="2C1cp",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
                "p": CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
            },
            smiles="[P][Ag][P]",
            position_matrix=np.array(
                [
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                ]
            ),
            num_fgs=2,
            name=name,
        ),
        lambda name: CaseData(
            precursor=TwoC2Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
                abead1=CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
                abead2=CgBead(
                    element_string="N",
                    bead_type="n",
                    coordination=2,
                ),
            ),
            precursor_name="2C2cpn",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
                "p": CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
                "n": CgBead(
                    element_string="N",
                    bead_type="n",
                    coordination=2,
                ),
            },
            smiles="[N][P][Ag][P][N]",
            position_matrix=np.array(
                [
                    [-8, 0, 0],
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                    [8, 0, 0],
                ]
            ),
            num_fgs=2,
            name=name,
        ),
        lambda name: CaseData(
            precursor=TwoC3Arm(
                bead=CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
                abead1=CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
                abead2=CgBead(
                    element_string="N",
                    bead_type="n",
                    coordination=2,
                ),
                abead3=CgBead(
                    element_string="C",
                    bead_type="a",
                    coordination=2,
                ),
            ),
            precursor_name="2C3cpna",
            bead_set={
                "c": CgBead(
                    element_string="Ag",
                    bead_type="c",
                    coordination=2,
                ),
                "p": CgBead(
                    element_string="P",
                    bead_type="p",
                    coordination=2,
                ),
                "n": CgBead(
                    element_string="N",
                    bead_type="n",
                    coordination=2,
                ),
                "a": CgBead(
                    element_string="C",
                    bead_type="a",
                    coordination=2,
                ),
            },
            smiles="[C][N][P][Ag][P][N][C]",
            position_matrix=np.array(
                [
                    [-12, 0, 0],
                    [-8, 0, 0],
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                    [8, 0, 0],
                    [12, 0, 0],
                ]
            ),
            num_fgs=2,
            name=name,
        ),
    )
)
def precursor(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
