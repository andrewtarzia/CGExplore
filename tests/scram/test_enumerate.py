import pathlib

import cgexplore as cgx

from .case_data import CaseData


def test_enumerate(graph_data: CaseData) -> None:
    """Test topology code enumeration.

    This should produce a graph and check the string against known ones, then
    delete them.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "temp_graphs"
    known_graph_directory = (
        pathlib.Path(__file__).resolve().parent / "test_graphs"
    )

    # Delete them.
    filename = graph_directory / f"rxx_{graph_data.graph_type}.json.gz"
    if graph_directory.exists():
        for filen in graph_directory.iterdir():
            filen.unlink()
        graph_directory.rmdir()

    graph_directory.mkdir(exist_ok=False)

    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Remake graphs.
        graph_directory=graph_directory,
    )
    assert iterator.count_graphs() == graph_data.num_graphs

    for tc in iterator.yield_graphs():
        # Look at previous string.
        str_file = (
            known_graph_directory / f"str_{iterator.graph_type}_{tc.idx}.txt"
        )
        if not str_file.exists():
            msg = f"{str_file} not found"
            with str_file.open("w") as f:
                f.write(tc.get_as_string())
            raise AssertionError(msg)

        with str_file.open("r") as f:
            lines = f.readlines()

        assert lines[0] == tc.get_as_string()

    # Delete them.
    filename = graph_directory / f"rxx_{iterator.graph_type}.json.gz"
    filename.unlink()
    graph_directory.rmdir()
