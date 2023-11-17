import pytest
import stk

from .case_data import CaseData

# Three tests cases with four atoms with known bond lengths,
# angles and torsions.


@pytest.fixture(
    params=(
        lambda name: CaseData(
            molecules=[
                stk.BuildingBlock(smiles="CC"),
                stk.BuildingBlock(smiles="CNC"),
                stk.BuildingBlock(smiles="CNNC"),
                stk.BuildingBlock(smiles="CNC"),
            ],
            property_dicts=[
                {"1": 2},
                {"1": 2},
                {"1": 2},
                {"1": 3, "2": {"h": "w"}},
            ],
            expected_count=3,
            name=name,
        ),
    )
)
def molecule(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
