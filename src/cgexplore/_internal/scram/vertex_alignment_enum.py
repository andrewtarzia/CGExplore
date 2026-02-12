"""Script to generate and optimise CG models."""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

import numpy as np
import rustworkx as rx
import stk
from agx import Configuration, TopologyCode

from cgexplore._internal.scram.enumeration import TopologyIterator
from cgexplore._internal.topologies.custom_topology import CustomTopology

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VertexAlignment:
    """Naming convention for vertex alignments."""

    idx: int
    vertex_dict: dict[int, int]

    def __str__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return (
            f"{self.__class__.__name__}(idx={self.idx}, "
            f"vertex_dict={self.vertex_dict})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return str(self)


def get_vertex_alignments(
    iterator: TopologyIterator,
    allow_rotation: abc.Sequence[int],
) -> abc.Sequence[VertexAlignment]:
    """Get potential vertex alignment dictionaries.

    Parameters:
        iterator:
            The graph iterator.

        allow_rotation:
            Which building blocks to allow rotation based on the number of
            functional groups.

    """
    allow_rotation = tuple(allow_rotation)

    # Get the associated vertex ids.
    modifiable_vertices = {
        vertex: range(fg_count) if fg_count in allow_rotation else [0]
        for fg_count in iterator.vertex_types_by_fg
        for vertex in iterator.vertex_types_by_fg[fg_count]
        if fg_count > 1
    }

    if len(modifiable_vertices) == 0:
        msg = "There are no modifiable types"
        raise RuntimeError(msg)

    iteration = it.product(*modifiable_vertices.values())
    possible_dicts = []
    for idx, item in enumerate(iteration):
        vmap = dict(zip(modifiable_vertices.keys(), item, strict=True))
        possible_dicts.append(VertexAlignment(idx=idx, vertex_dict=vmap))

    return tuple(possible_dicts)


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
            for i, vert_ids in bb_config.node_idx_dict.items()
            if vi in vert_ids
        )
        va = vertex_alignment.vertex_dict[vi]

        vertices[f"{vi}-{bb_id}-{va}"] = graph.add_node(f"{vi}-{bb_id}-{va}")

    for vert in topology_code.vertex_map:
        v1 = vert[0]
        bb_id = next(
            i
            for i, vert_ids in bb_config.node_idx_dict.items()
            if v1 in vert_ids
        )
        va1 = vertex_alignment.vertex_dict[v1]
        v1str = f"{v1}-{bb_id}-{va1}"
        v2 = vert[1]
        bb_id = next(
            i
            for i, vert_ids in bb_config.node_idx_dict.items()
            if v2 in vert_ids
        )
        va2 = vertex_alignment.vertex_dict[v2]
        v2str = f"{v2}-{bb_id}-{va2}"
        nodeaidx = vertices[v1str]
        nodebidx = vertices[v2str]
        graph.add_edge(nodeaidx, nodebidx, None)

    return graph


def aligned_construction(  # noqa: PLR0913
    iterator: TopologyIterator,
    topology_code: TopologyCode,
    scale_multiplier: float | None = None,
    building_block_configuration: Configuration | None = None,
    vertex_positions: dict[int, np.ndarray] | None = None,
    vertex_alignment: VertexAlignment | None = None,
    reaction_factory: stk.ReactionFactory = stk.GenericReactionFactory(),  # noqa: B008
    optimizer: stk.Optimizer = stk.NullOptimizer(),  # noqa: B008
) -> stk.ConstructedMolecule:
    """Try construction with alignment, then without."""
    msg = "To be implemented with other construction once vas done."
    raise NotImplementedError(msg)
    if building_block_configuration is None:  # type: ignore[unreachable]
        bbs = iterator.building_blocks
    else:
        bbs = building_block_configuration.get_building_block_dictionary()

    if vertex_alignment is None:
        vertex_alignments = None
    else:
        vertex_alignments = vertex_alignment.vertex_dict

    new_vertices = [
        i
        if i.__class__.__name__ == "NonLinearVertex"
        else stk.cage.LinearVertex(
            id=i.get_id(),
            position=i.get_position(),
            use_neighbor_placement=False,
        )
        for i in iterator.get_vertex_prototypes(unaligning=False)
    ]

    # Try with aligning vertices.
    return stk.ConstructedMolecule(
        CustomTopology(  # type: ignore[arg-type]
            building_blocks=bbs,
            vertex_prototypes=new_vertices,
            # Convert to edge prototypes.
            edge_prototypes=topology_code.edges_from_connection(new_vertices),
            vertex_alignments=vertex_alignments,
            vertex_positions=vertex_positions,
            scale_multiplier=iterator.scale_multiplier
            if scale_multiplier is None
            else scale_multiplier,
            reaction_factory=reaction_factory,
            optimizer=optimizer,
        )
    )


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
    current_graph = get_bb_va_topology_code_graph(  # type: ignore[unreachable]
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
            node_matcher=lambda x, y: (
                (x.split("-")[1] == y.split("-")[1])
                and (x.split("-")[2] == y.split("-")[2])
            ),
        ):
            passed_iso = False
            break
    return passed_iso
