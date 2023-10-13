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

ag_bead = CgBead(
    element_string="Ag",
    bead_type="c1",
    bead_class="c",
    coordination=4,
)
p_bead = CgBead(
    element_string="P",
    bead_type="p1",
    bead_class="p",
    coordination=2,
)
n_bead = CgBead(
    element_string="N",
    bead_class="n",
    bead_type="n1",
    coordination=2,
)
a_bead = CgBead(
    element_string="C",
    bead_class="a",
    bead_type="a1",
    coordination=2,
)


@pytest.fixture(
    params=(
        lambda name: CaseData(
            precursor=FourC0Arm(bead=ag_bead),
            precursor_name="4C0c1",
            bead_set={"c1": ag_bead},
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
            precursor=FourC1Arm(bead=ag_bead, abead1=p_bead),
            precursor_name="4C1c1p1",
            bead_set={"c1": ag_bead, "p1": p_bead},
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
            precursor=ThreeC0Arm(bead=ag_bead),
            precursor_name="3C0c1",
            bead_set={"c1": ag_bead},
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
            precursor=ThreeC1Arm(bead=ag_bead, abead1=p_bead),
            precursor_name="3C1c1p1",
            bead_set={"c1": ag_bead, "p1": p_bead},
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
            precursor=ThreeC2Arm(bead=ag_bead, abead1=p_bead, abead2=n_bead),
            precursor_name="3C2c1p1n1",
            bead_set={"c1": ag_bead, "p1": p_bead, "n1": n_bead},
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
            precursor=TwoC0Arm(bead=ag_bead),
            precursor_name="2C0c1",
            bead_set={"c1": ag_bead},
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
            precursor=TwoC1Arm(bead=ag_bead, abead1=p_bead),
            precursor_name="2C1c1p1",
            bead_set={"c1": ag_bead, "p1": p_bead},
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
            precursor=TwoC2Arm(bead=ag_bead, abead1=p_bead, abead2=n_bead),
            precursor_name="2C2c1p1n1",
            bead_set={"c1": ag_bead, "p1": p_bead, "n1": n_bead},
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
                bead=ag_bead, abead1=p_bead, abead2=n_bead, abead3=a_bead
            ),
            precursor_name="2C3c1p1n1a1",
            bead_set={"c1": ag_bead, "p1": p_bead, "n1": n_bead, "a1": a_bead},
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
