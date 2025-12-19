"""Script to generate and optimise CG models."""

import logging
from collections import abc

import rustworkx as rx
from agx import Configuration, TopologyCode

from .building_block_enum import VertexAlignment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_bb_va_topology_code_graph(
    topology_code: TopologyCode,
    bb_config: Configuration,
    vertex_alignment: VertexAlignment,
) -> rx.PyGraph:
    """Convert TopologyCode and BBConfig and vertex alignments to rx graph."""
    graph: rx.PyGraph = rx.PyGraph(multigraph=True)

    vertices = {}
    for vi in sorted({i for j in topology_code.vertex_map for i in j}):
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if vi in vert_ids
        )
        va = vertex_alignment.vertex_dict[vi]

        vertices[f"{vi}-{bb_id}-{va}"] = graph.add_node(f"{vi}-{bb_id}-{va}")

    for vert in topology_code.vertex_map:
        v1 = vert[0]
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if v1 in vert_ids
        )
        va1 = vertex_alignment.vertex_dict[v1]
        v1str = f"{v1}-{bb_id}-{va1}"
        v2 = vert[1]
        bb_id = next(
            i
            for i, vert_ids in bb_config.building_block_idx_dict.items()
            if v2 in vert_ids
        )
        va2 = vertex_alignment.vertex_dict[v2]
        v2str = f"{v2}-{bb_id}-{va2}"
        nodeaidx = vertices[v1str]
        nodebidx = vertices[v2str]
        graph.add_edge(nodeaidx, nodebidx, None)

    return graph


def passes_graph_bb_va_iso(
    topology_code: TopologyCode,
    bb_config: Configuration,
    vertex_alignment: VertexAlignment,
    run_topology_codes: abc.Sequence[
        tuple[TopologyCode, Configuration, VertexAlignment]
    ],
) -> bool:
    """Check if a graph and bb config passes isomorphism check."""
    msg = (
        "I do not know if this is technically correct, so I do not recommmend"
    )
    raise NotImplementedError(msg)

    # Testing bb-config aware graph check.
    # Convert TopologyCode to a graph.
    current_graph = get_bb_va_topology_code_graph(
        topology_code=topology_code,
        bb_config=bb_config,
        vertex_alignment=vertex_alignment,
    )

    # Check that graph for isomorphism with others graphs.
    passed_iso = True
    for tc, bc, va in run_topology_codes:
        test_graph = get_bb_va_topology_code_graph(
            topology_code=tc,
            bb_config=bc,
            vertex_alignment=va,
        )

        if rx.is_isomorphic(
            current_graph,
            test_graph,
            node_matcher=lambda x, y: (x.split("-")[1] == y.split("-")[1])
            and (x.split("-")[2] == y.split("-")[2]),
        ):
            passed_iso = False
            break
    return passed_iso
