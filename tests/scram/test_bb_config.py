import pathlib

import agx
import pytest

import cgexplore as cgx

from .case_data import CaseData


def test_building_block_configuration(graph_data: CaseData) -> None:
    """Test graph layout processes."""
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    config_directory = pathlib.Path(__file__).resolve().parent / "test_configs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )
    if graph_data.num_configs == 0:
        with pytest.raises(RuntimeError) as e_info:
            iterator.get_configurations()
        print(str(e_info.value))
        assert str(e_info.value) == (
            "modifiable_types is len 0. If 0, then you have no need to "
            "screen building block configurations. If greater than 2, then "
            "this code cannot handle this yet. Sorry!"
        )
    else:
        possible_bbdicts = iterator.get_configurations()
        assert len(possible_bbdicts) == graph_data.num_configs

        run_topology_codes: list[agx.ConfiguredCode] = []
        for bb_config in possible_bbdicts:
            # Check for iso checks, iterating over topology codes as well.
            for topology_code in iterator.yield_graphs():
                configured = agx.ConfiguredCode(topology_code, bb_config)
                if agx.utilities.is_configured_code_isomoprhic(
                    test_code=configured,
                    run_topology_codes=run_topology_codes,
                ):
                    assert (
                        topology_code.idx,
                        bb_config.idx,
                    ) in graph_data.iso_pass
                    assert graph_data.iso_pass[len(run_topology_codes)] == (
                        topology_code.idx,
                        bb_config.idx,
                    )
                    run_topology_codes.append(configured)

            bc_name = (
                config_directory
                / f"bc_{graph_data.graph_type}_{topology_code.idx}"
                f"_{bb_config.idx}.txt"
            )

            if not bc_name.exists():
                with bc_name.open("w") as f:
                    f.write(str(bb_config.get_hashable_idx_dict()))

            with bc_name.open("r") as f:
                lines = f.readlines()
            test = lines[0]
            assert str(bb_config.get_hashable_idx_dict()) == test
