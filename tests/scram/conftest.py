import pathlib

import pytest
import stk

from .case_data import CaseData

bbs = {
    1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
    2: stk.BuildingBlock("BrCNCBr", (stk.BromoFactory(),)),
    3: stk.BuildingBlock("BrC(Br)BCBr", (stk.BromoFactory(),)),
    4: stk.BuildingBlock("BrCC(Br)COC(Br)CCBr", (stk.BromoFactory(),)),
}


@pytest.fixture(
    params=(
        lambda name: CaseData(
            building_block_counts={bbs[4]: 2, bbs[2]: 4},
            graph_type="2-4FG_4-2FG",
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=1e4,
            graph_filename="rxx_2-4FG_4-2FG.json",
            num_graphs=2,
            name=name,
            doubles={0: True, 1: True},
            parallels={0: True, 1: False},
        ),
        lambda name: CaseData(
            building_block_counts={bbs[3]: 4, bbs[2]: 6},
            graph_type="4-3FG_6-2FG",
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=1e4,
            graph_filename="rxx_4-3FG_6-2FG.json",
            num_graphs=2,
            name=name,
            doubles={0: True, 1: True, 2: True, 3: False, 4: False},
            parallels={0: True, 1: False, 2: True, 3: True, 4: False},
        ),
        lambda name: CaseData(
            building_block_counts={bbs[4]: 5},
            graph_type="5-4FG",
            graph_set="rxx",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=1e4,
            graph_filename="rxx_5-4FG.json",
            num_graphs=2,
            name=name,
            doubles={0: True, 1: False, 2: True, 3: True, 4: True, 5: False},
            parallels={0: True, 1: True, 2: True, 3: True, 4: False, 5: True},
        ),
        # tri tri tri
        # lambda name: CaseData(
        #     building_block_counts={bbs[4]: 2, bbs[2]: 4},
        #     graph_type="2-4FG_4-2FG",
        #     graph_set="rxx",
        #     graph_directory=pathlib.Path(__file__).resolve().parent
        #     / "temp_graphs",
        #     max_samples=1e4,
        #     graph_filename="rxx_2-4FG_4-2FG.json",
        #     num_graphs=2,
        #     name=name,
        # ),
    )
)
def graph_data(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )


@pytest.fixture(params=["kamada", "spring", "spectral"])
def graph_type(request: pytest.FixtureRequest) -> CaseData:
    return request.param


@pytest.fixture(params=[1, 5])
def scale(request: pytest.FixtureRequest) -> CaseData:
    return request.param
