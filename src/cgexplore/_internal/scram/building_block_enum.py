"""Script to generate and optimise CG models."""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

from cgexplore._internal.scram.enumeration import TopologyIterator

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
