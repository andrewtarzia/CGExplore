import pathlib

import pytest
import stk

from .case_data import CaseData

bbs = {
    1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
    2: stk.BuildingBlock("BrCNCBr", (stk.BromoFactory(),)),
    3: stk.BuildingBlock("BrC(Br)BCBr", (stk.BromoFactory(),)),
    4: stk.BuildingBlock("BrCC(Br)COC(Br)CCBr", (stk.BromoFactory(),)),
    "2x": stk.BuildingBlock("BrCCSCCBr", (stk.BromoFactory(),)),
    "3x": stk.BuildingBlock("BrC(Br)SCSBCBr", (stk.BromoFactory(),)),
}


@pytest.fixture(
    params=(
        lambda name: CaseData(
            building_block_counts={bbs[4]: 2, bbs[2]: 4},
            graph_type="2-4FG_4-2FG",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            num_graphs=2,
            num_configs=0,
            name=name,
            iso_pass=(),
        ),
        lambda name: CaseData(
            building_block_counts={bbs[4]: 3, bbs[2]: 3, bbs["2x"]: 3},
            graph_type="3-4FG_6-2FG",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            num_graphs=4,
            num_configs=20,
            name=name,
            iso_pass=(
                (0, 0),
                (1, 0),
                (2, 0),
                (4, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (4, 1),
                (0, 2),
                (1, 2),
                (4, 2),
                (4, 4),
                (0, 5),
                (1, 5),
                (4, 6),
                (0, 7),
                (1, 9),
                (1, 10),
                (4, 17),
            ),
        ),
        lambda name: CaseData(
            building_block_counts={bbs[3]: 4, bbs[2]: 6},
            graph_type="4-3FG_6-2FG",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            num_graphs=5,
            num_configs=0,
            name=name,
            iso_pass=(),
        ),
        lambda name: CaseData(
            building_block_counts={bbs[4]: 5},
            graph_type="5-4FG",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            num_graphs=6,
            num_configs=0,
            name=name,
            iso_pass=(),
        ),
        lambda name: CaseData(
            building_block_counts={bbs[3]: 2, bbs[2]: 2, bbs[1]: 2},
            graph_type="2-3FG_2-2FG_2-1FG",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            num_graphs=2,
            num_configs=0,
            name=name,
            iso_pass=(),
        ),
        lambda name: CaseData(
            building_block_counts={bbs[4]: 3, bbs[3]: 3, bbs["3x"]: 1},
            graph_type="3-4FG_4-3FG",
            graph_directory=pathlib.Path(__file__).resolve().parent
            / "temp_graphs",
            max_samples=int(1e4),
            num_graphs=9,
            num_configs=4,
            name=name,
            iso_pass=(
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (8, 1),
                (1, 2),
                (4, 2),
                (6, 2),
                (3, 3),
                (4, 3),
            ),
        ),
    )
)
def graph_data(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore[attr-defined]
    )


@pytest.fixture(params=["kamada", "spring", "spectral"])
def layout_type(request: pytest.FixtureRequest) -> CaseData:
    return request.param


@pytest.fixture(params=[1, 5])
def scale(request: pytest.FixtureRequest) -> CaseData:
    return request.param
