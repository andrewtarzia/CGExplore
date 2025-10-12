import pathlib

import numpy as np
import stk

import cgexplore as cgx

from .case_data import CaseData


def test_layouts(
    graph_data: CaseData,
    graph_type: str,
    scale: int,
) -> None:
    """Test graph layout processes.

    Parameters:

        graph_data:
            The graph data.

    """
    known_mols = pathlib.Path(__file__).resolve().parent / "test_molecules"

    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        graph_type=graph_data.graph_type,
        graph_set=graph_data.graph_set,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=pathlib.Path(__file__).resolve().parent
        / "test_graphs",
    )

    for idx, tc in enumerate(iterator.yield_graphs()):
        vs_name = (
            known_mols
            / f"vs_{graph_data.graph_type}_{idx}_{graph_type}_{scale}.mol"
        )

        if not vs_name.exists():
            # Build it.
            vertex_set = cgx.scram.get_vertexset_molecule(
                graph_type=graph_type,
                scale=scale,
                topology_code=tc,
                iterator=iterator,
                bb_config=None,
            )
            vertex_set.write(vs_name)

        known_molecule = stk.BuildingBlock.init_from_file(vs_name)
        vertex_set = cgx.scram.get_vertexset_molecule(
            graph_type=graph_type,
            scale=scale,
            topology_code=tc,
            iterator=iterator,
            bb_config=None,
        )

        # Actual coordinates are not constant, so test other measures.
        assert np.allclose(
            known_molecule.get_centroid(),
            vertex_set.get_centroid(),
            atol=1e-3,
        )
        # Currently, we are using graph layouts that are not deterministic,
        # unsure why. One day this will be fixed, for now, we expect failure.
        # Although it seems spectral is consistent!
        # Actually no, not across machines. So no longer checking position
        # matrices.

        if graph_type != "spectral":
            rg_name = (
                known_mols
                / f"rg_{graph_data.graph_type}_{idx}_{graph_type}_{scale}.mol"
            )

            if not rg_name.exists():
                regraphed = cgx.scram.get_regraphed_molecule(
                    graph_type=graph_type,
                    scale=scale,
                    topology_code=tc,
                    iterator=iterator,
                    bb_config=None,
                )
                regraphed.write(rg_name)

            known_molecule = stk.BuildingBlock.init_from_file(rg_name)
            regraphed = cgx.scram.get_regraphed_molecule(
                graph_type=graph_type,
                scale=scale,
                topology_code=tc,
                iterator=iterator,
                bb_config=None,
            )

            # Actual coordinates are not constant, so test other measures.
            assert np.allclose(
                known_molecule.get_centroid(),
                regraphed.get_centroid(),
                atol=1e-3,
            )
