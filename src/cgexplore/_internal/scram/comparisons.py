"""Script to generate and optimise CG models."""

import logging
from collections import abc

import rustworkx as rx

from .building_block_enum import BuildingBlockConfiguration
from .topology_code import TopologyCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_bb_topology_code_graph(
    topology_code: TopologyCode,
    bb_config: BuildingBlockConfiguration,
) -> rx.PyGraph:
    """Convert TopologyCode and BBConfig to rx graph."""
    graph: rx.PyGraph = rx.PyGraph(multigraph=True)

    vertices = {}
    for vi in sorted({i for j in topology_code.vertex_map for i in j}):
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if vi in vert_ids
        )

        vertices[f"{vi}-{bb_id}"] = graph.add_node(f"{vi}-{bb_id}")

    for vert in topology_code.vertex_map:
        v1 = vert[0]
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if v1 in vert_ids
        )
        v1str = f"{v1}-{bb_id}"
        v2 = vert[1]
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if v2 in vert_ids
        )
        v2str = f"{v2}-{bb_id}"
        nodeaidx = vertices[v1str]
        nodebidx = vertices[v2str]
        graph.add_edge(nodeaidx, nodebidx, None)

    return graph


def passes_graph_bb_iso(
    topology_code: TopologyCode,
    bb_config: BuildingBlockConfiguration,
    run_topology_codes: abc.Sequence[
        tuple[TopologyCode, BuildingBlockConfiguration]
    ],
) -> bool:
    """Check if a graph and bb config passes isomorphism check."""
    # Testing bb-config aware graph check.
    # Convert TopologyCode to a graph.
    current_graph = get_bb_topology_code_graph(
        topology_code=topology_code,
        bb_config=bb_config,
    )

    # Check that graph for isomorphism with others graphs.
    passed_iso = True
    for tc, bc in run_topology_codes:
        test_graph = get_bb_topology_code_graph(topology_code=tc, bb_config=bc)

        if rx.is_isomorphic(
            current_graph,
            test_graph,
            node_matcher=lambda x, y: x.split("-")[1] == y.split("-")[1],
        ):
            passed_iso = False
            break
    return passed_iso
