import pathlib
from dataclasses import dataclass

import stk


@dataclass(slots=True, frozen=True)
class CaseData:
    """A test case."""

    building_block_counts: dict[stk.BuildingBlock, int]
    graph_type: str
    graph_set: str
    graph_directory: pathlib.Path
    graph_filename: str
    num_graphs: int
    max_samples: int
    doubles: dict[int, bool]
    parallels: dict[int, bool]
    name: str
