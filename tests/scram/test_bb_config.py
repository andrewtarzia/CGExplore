import pathlib

import pytest

import cgexplore as cgx

from .case_data import CaseData


def test_building_block_configuration(graph_data: CaseData) -> None:
    """Test graph layout processes.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    config_directory = pathlib.Path(__file__).resolve().parent / "test_configs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        graph_type=graph_data.graph_type,
        graph_set=graph_data.graph_set,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )
    if graph_data.num_configs == 0:
        with pytest.raises(RuntimeError) as e_info:
            cgx.scram.get_custom_bb_configurations(iterator=iterator)
        print(str(e_info.value))
        assert str(e_info.value) == (
            "modifiable_types is len 0. If 0, then you have no need to "
            "screen building block configurations. If greater than 2, then "
            "this code cannot handle this yet. Sorry!"
        )
    else:
        possible_bbdicts = cgx.scram.get_custom_bb_configurations(
            iterator=iterator
        )
        assert len(possible_bbdicts) == graph_data.num_configs

        run_topology_codes = []
        for bb_config in possible_bbdicts:
            # Check for iso checks, iterating over topology codes as well.
            for idx, topology_code in enumerate(iterator.yield_graphs()):
                if cgx.scram.passes_graph_bb_iso(
                    topology_code=topology_code,
                    bb_config=bb_config,
                    run_topology_codes=run_topology_codes,
                ):
                    assert (idx, bb_config.idx) in graph_data.iso_pass
                    run_topology_codes.append((topology_code, bb_config))

            bc_name = (
                config_directory
                / f"bc_{graph_data.graph_type}_{idx}_{bb_config.idx}.txt"
            )

            if not bc_name.exists():
                with bc_name.open("w") as f:
                    f.write(str(bb_config.get_hashable_bbidx_dict()))

            with bc_name.open("r") as f:
                lines = f.readlines()
            test = lines[0]
            assert str(bb_config.get_hashable_bbidx_dict()) == test
