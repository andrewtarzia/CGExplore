import pathlib

import cgexplore as cgx

from .case_data import CaseData


def test_enumerate(graph_data: CaseData) -> None:
    """Test topology code enumeration.

    This should produce a graph and check the string against known ones, then
    delete them.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "temp_graphs"
    known_graph_directory = (
        pathlib.Path(__file__).resolve().parent / "test_graphs"
    )
    graph_directory.mkdir(exist_ok=False)

    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        graph_type=graph_data.graph_type,
        graph_set=graph_data.graph_set,
        max_samples=graph_data.max_samples,
        # Remake graphs.
        graph_directory=graph_directory,
    )

    for idx, tc in enumerate(iterator.yield_graphs()):
        # Look at previous string.
        str_file = (
            known_graph_directory / f"str_{graph_data.graph_type}_{idx}.txt"
        )
        if not str_file.exists():
            raise AssertionError

        with str_file.open("r") as f:
            lines = f.readlines()

        assert lines[0] == tc.get_as_string()

    # Delete them.
    filename = (
        graph_directory
        / f"{graph_data.graph_set}_{graph_data.graph_type}.json"
    )
    filename.unlink()
    graph_directory.rmdir()
