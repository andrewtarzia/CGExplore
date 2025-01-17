"""Definition of a custom topology graph."""

from collections import abc

import numpy as np
import stk


class CustomTopology:
    """Container for a custom topology graph."""

    def __init__(  # noqa: PLR0913
        self,
        building_blocks: (
            abc.Iterable[stk.BuildingBlock]
            | dict[stk.BuildingBlock, tuple[int, ...]]
        ),
        vertex_prototypes: abc.Sequence[stk.Vertex],
        edge_prototypes: abc.Sequence[stk.Edge],
        vertex_alignments: dict[int, int] | None = None,
        vertex_positions: dict[int, np.ndarray] | None = None,
        reaction_factory: stk.ReactionFactory = stk.GenericReactionFactory(),  # noqa: B008
        num_processes: int = 1,
        optimizer: stk.Optimizer = stk.NullOptimizer(),  # noqa: B008
        scale_multiplier: float = 1.0,
    ) -> None:
        """Initialize."""

        class InternalTopology(stk.cage.Cage):
            _vertex_prototypes = vertex_prototypes  # type: ignore[assignment]
            _edge_prototypes = edge_prototypes  # type: ignore[assignment]

        self._topology_graph = InternalTopology(
            building_blocks=building_blocks,
            vertex_alignments=vertex_alignments,
            vertex_positions=vertex_positions,
            reaction_factory=reaction_factory,
            num_processes=num_processes,
            scale_multiplier=scale_multiplier,
            optimizer=optimizer,
        )

    def construct(self) -> stk.ConstructionResult:
        """Construct topology."""
        return self._topology_graph.construct()
