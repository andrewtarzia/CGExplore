"""Define classes for enumeration of graphs."""

import logging
import pathlib
from collections import abc, defaultdict
from dataclasses import dataclass, field

import agx
import stk

from .utilities import points_on_sphere

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TopologyIterator:
    """Iterate over topology graphs.

    This is the latest version, but without good symmetry and graph checks,
    this can over produce structures.

    This class now contains :class:`agx.TopologyIterator`.

    .. important::

      **Warning**: Currently, the order of ``building_block_counts`` has to
      have the building block with the most FGs first! This ordering is defined
      by the order used when defining the graphs. If you are defining your own
      graph library (i.e., setting ``graph_directory`` or using a new
      ``graph_type``), then the order is defined by the order in
      ``building_block_counts`` when generating the json.

    .. important::

      To reproduce the ``no_doubles`` dataset, you must filter the topology
      codes after generation using the :class:`cgexplore.scram.TopologyCode`
      methods (this is now the recommended approach).

    Parameters:
        building_block_counts:
            Dictionary of :class:`stk.BuildingBlock` and their count in the
            proposed structures. Always put the building blocks with more
            functional groups first (this is a current bug). Additionally, only
            mixtures of three distinct building block functional group counts
            is implemented, and in the case of three components, all building
            blocks bind to the building block with the most functional groups.

        graph_type:
            Name of the graph. Current name convention is long, but complete,
            capturing the count of each building block with certain functional
            group count included. Following this name convention will allow you
            to use saved graphs, if not, you can make your own. Although it can
            be time consuming.

        graph_set:
            Set of graphs to use based on different algorithms or papers.
            Can be custom, as above. Note that the code to generation ``nx``
            graphs is no longer present in ``cgexplore`` because the
            :mod:`networkx` algorithms were slow.

        scale_multiplier:
            Scale multiplier to use in construction.

        allowed_num_components:
            Allowed number of disconnected graph components. Usually ``1`` to
            generate complete graphs only.

        max_samples:
            When constructing graphs, there is some randomness in their order,
            although that order should be consistent, and only up-to
            ``max_samples`` are sampled. For very large numbers of building
            blocks there is not guarantee all possible graphs will be explored.

        graph_directory:
            Directory to check for and save graph jsons.

        verbose:
            Whether to log outcomes.

    """

    building_block_counts: dict[stk.BuildingBlock, int]
    graph_type: str | None = None
    graph_set: str = "rxx"
    scale_multiplier = 5
    allowed_num_components: int = 1
    max_samples: int | None = None
    graph_directory: pathlib.Path | None = None
    verbose: bool = True
    node_to_bb_map: dict[agx.NodeType, stk.BuildingBlock] = field(init=False)
    node_counts: dict[agx.NodeType, int] = field(init=False)
    iterator: agx.TopologyIterator = field(init=False)
    building_blocks: dict[stk.BuildingBlock, abc.Sequence[int]] = field(
        init=False
    )
    vertex_types_by_fg: dict[int, abc.Sequence[int]] = field(init=False)
    vertex_prototypes: list[stk.Vertex] = field(init=False)
    unaligned_vertex_prototypes: list[stk.Vertex] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize."""
        if (
            self.graph_directory is not None
            and not self.graph_directory.exists()
        ):
            msg = (
                f"User specified graph directory does not "
                f"exist ({self.graph_directory})"
            )
            raise RuntimeError(msg)

        if self.graph_set != "rxx":
            msg = f"{self.graph_set} not defined"  # type:ignore[unreachable]
            raise NotImplementedError(msg)

        self.node_to_bb_map = {
            agx.NodeType(
                type_id=idx, num_connections=key.get_num_functional_groups()
            ): key
            for idx, key in enumerate(self.building_block_counts)
        }
        self.node_counts = {
            agx.NodeType(
                type_id=idx, num_connections=key.get_num_functional_groups()
            ): value
            for idx, (key, value) in enumerate(
                self.building_block_counts.items()
            )
        }

        self.iterator = agx.TopologyIterator(
            node_counts=self.node_counts,
            graph_type=self.graph_type,
            graph_set=self.graph_set,
            graph_directory=self.graph_directory,
            allowed_num_components=self.allowed_num_components,
            max_samples=self.max_samples,
            verbose=self.verbose,
        )

        # Use an angle rotation of points on a sphere for each building block
        # type to avoid overlap of distinct building block spheres with the
        # same number of instances.
        angle_rotations = range(
            0, 360, int(360 / len(self.building_block_counts))
        )

        # Write vertex prototypes as a function of number of functional groups
        # and position them on spheres.
        vertex_prototypes: list[stk.Vertex] = []
        unaligned_vertex_prototypes: list[stk.Vertex] = []
        vertex_types_by_fg = defaultdict(list)
        building_block_dict: dict[stk.BuildingBlock, list[int]] = {}
        for building_block, angle_rotation in zip(
            self.building_block_counts,
            angle_rotations,
            strict=True,
        ):
            building_block_dict[building_block] = []

            num_functional_groups = building_block.get_num_functional_groups()
            num_instances = self.building_block_counts[building_block]

            type_positions = points_on_sphere(
                sphere_radius=1,
                num_points=num_instances,
                angle_rotation=angle_rotation,
            )
            for _, position in zip(
                range(num_instances),
                type_positions,
                strict=True,
            ):
                vertex_id = len(vertex_prototypes)
                vertex_types_by_fg[num_functional_groups].append(vertex_id)
                building_block_dict[building_block].append(vertex_id)

                if num_functional_groups == 1:
                    vertex_prototypes.append(
                        stk.cage.UnaligningVertex(
                            id=vertex_id,
                            position=position,
                            use_neighbor_placement=False,
                        )
                    )

                elif num_functional_groups == 2:  # noqa: PLR2004
                    vertex_prototypes.append(
                        stk.cage.AngledVertex(
                            id=vertex_id,
                            position=position,
                            use_neighbor_placement=False,
                        )
                    )

                elif num_functional_groups >= 3:  # noqa: PLR2004
                    vertex_prototypes.append(
                        stk.cage.NonLinearVertex(
                            id=vertex_id,
                            position=position,
                            use_neighbor_placement=False,
                        )
                    )

                else:
                    msg = "Building blocks need at least 1 FG."
                    raise RuntimeError(msg)

                unaligned_vertex_prototypes.append(
                    stk.cage.UnaligningVertex(
                        id=vertex_id,
                        position=position,
                        use_neighbor_placement=False,
                    )
                )

        self.building_blocks: dict[stk.BuildingBlock, abc.Sequence[int]] = {
            i: tuple(building_block_dict[i]) for i in building_block_dict
        }
        self.vertex_types_by_fg = {
            i: tuple(vertex_types_by_fg[i]) for i in vertex_types_by_fg
        }
        self.vertex_prototypes = vertex_prototypes
        self.unaligned_vertex_prototypes = unaligned_vertex_prototypes
        self.graph_type = self.iterator.graph_type

    def get_num_building_blocks(self) -> int:
        """Get number of building blocks."""
        return len(self.vertex_prototypes)

    def get_vertex_prototypes(
        self,
        unaligning: bool,
    ) -> abc.Sequence[stk.Vertex]:
        """Get vertex prototypes."""
        if unaligning:
            return self.unaligned_vertex_prototypes
        return self.vertex_prototypes

    def get_edges_from_topology_code(
        self,
        topology_code: agx.TopologyCode,
        unaligning: bool = False,
    ) -> list[stk.Edge]:
        """Get stk Edges from topology code."""
        vertex_prototypes = self.get_vertex_prototypes(unaligning=unaligning)

        return [
            stk.Edge(
                id=i,
                vertex1=vertex_prototypes[pair[0]],
                vertex2=vertex_prototypes[pair[1]],
            )
            for i, pair in enumerate(topology_code.vertex_map)
        ]

    def count_graphs(self) -> int:
        """Count completely connected graphs in iteration."""
        return self.iterator.count_graphs()

    def graph_exists(self) -> bool:
        """Checks if the graphs have been defined."""
        return self.iterator.graph_path.exists()

    def yield_graphs(self) -> abc.Generator[agx.TopologyCode]:
        """Get constructed molecules from iteration.

        Yields only graphs with the allowed number of components.
        """
        yield from self.iterator.yield_graphs()

    def get_configurations(self) -> abc.Sequence[agx.Configuration]:
        """Get potential node configurations."""
        return self.iterator.get_configurations()

    def yield_configured_codes(self) -> abc.Iterator[agx.ConfiguredCode]:
        """Get potential node configurations."""
        yield from self.iterator.yield_configured_codes()
