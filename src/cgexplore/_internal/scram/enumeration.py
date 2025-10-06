"""Define classes for enumeration of graphs."""

import json
import logging
import pathlib
from collections import Counter, abc, defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rustworkx as rx
import stk

from .topology_code import TopologyCode
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

    .. important::

      **Warning**: Currently, the order of ``building_block_counts`` has to
      have the building block with the most FGs first!

    .. important::

      To reproduce the ``no_doubles'' dataset, you must use
      ``graph_set=rx_nodoubles``, or filter the topology codes after
      generation (this is now the recommended approach).

    Parameters:
        building_block_counts:
            Dictionary of :class:`stk.BuildingBlock` and their count in the
            proposed structures. Always put the building blocks with more
            functional groups first (this is a current bug).

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

    """

    building_block_counts: dict[stk.BuildingBlock, int]
    graph_type: str
    graph_set: Literal["rxx", "rx", "nx", "rx_nodoubles"] = "rxx"
    scale_multiplier = 5
    allowed_num_components: int = 1
    max_samples: int | None = None
    graph_directory: pathlib.Path | None = None

    def __post_init__(self) -> None:  # noqa: PLR0915, PLR0912, C901
        """Initialize."""
        if self.graph_directory is None:
            self.graph_directory = (
                pathlib.Path(__file__).resolve().parent / "known_graphs"
            )

        if not self.graph_directory.exists():
            msg = f"graph directory does not exist ({self.graph_directory})"
            raise RuntimeError(msg)

        match self.graph_set:
            case "rxx":
                self.graph_path = (
                    self.graph_directory / f"rxx_{self.graph_type}.json"
                )
                if self.max_samples is None:
                    self.used_samples = int(1e7)

            case "rx":
                self.graph_path = (
                    self.graph_directory / f"rx_{self.graph_type}.json"
                )
                if self.max_samples is None:
                    self.used_samples = int(1e4)

            case "rx_nodoubles":
                self.graph_path = (
                    self.graph_directory / f"rxnd_{self.graph_type}.json"
                )
                if self.max_samples is None:
                    self.used_samples = int(1e5)

            case "nx":
                self.graph_path = (
                    self.graph_directory / f"g_{self.graph_type}.json"
                )
                self.used_samples = 0

        # Use an angle rotation of points on a sphere for each building block
        # type to avoid overlap of distinct building block spheres with the
        # same number of instances.
        angle_rotations = range(
            0, 360, int(360 / len(self.building_block_counts))
        )

        # Write vertex prototypes as a function of number of functional groups
        # and position them on spheres.
        vertex_map = {}
        vertex_prototypes: list[stk.Vertex] = []
        unaligned_vertex_prototypes = []
        reactable_vertex_ids = []
        num_edges = 0
        vertex_counts = {}
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

                vertex_map[vertex_id] = building_block

                num_edges += num_functional_groups
                reactable_vertex_ids.extend(
                    [vertex_id] * num_functional_groups
                )
                building_block_dict[building_block].append(vertex_id)
                vertex_counts[vertex_id] = num_functional_groups
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
                    msg = "wrong number of functional groups"
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
        self.vertex_map = vertex_map
        self.vertex_counts = vertex_counts
        self.vertex_types_by_fg = {
            i: tuple(vertex_types_by_fg[i]) for i in vertex_types_by_fg
        }
        self.reactable_vertex_ids = reactable_vertex_ids
        self.vertex_prototypes = vertex_prototypes
        self.unaligned_vertex_prototypes = unaligned_vertex_prototypes

    def get_num_building_blocks(self) -> int:
        """Get number of building blocks."""
        return len(self.vertex_prototypes)

    def get_vertex_prototypes(
        self, unaligning: bool
    ) -> abc.Sequence[stk.Vertex]:
        """Get vertex prototypes."""
        if unaligning:
            return self.unaligned_vertex_prototypes
        return self.vertex_prototypes

    def _two_type_algorithm(self) -> None:
        combinations_tested = set()
        run_topology_codes: list[TopologyCode] = []

        type1, type2 = sorted(self.vertex_types_by_fg.keys(), reverse=True)

        itera1 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_fg[type1]
        ]

        rng = np.random.default_rng(seed=100)
        options = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_fg[type2]
        ]

        to_save = []
        for _ in range(self.used_samples):
            rng.shuffle(options)
            # Build an edge selection.
            combination: abc.Sequence[tuple[int, int]] = [
                tuple(sorted((i, j)))  # type:ignore[misc]
                for i, j in zip(itera1, options, strict=True)
            ]

            # Need to check for nonsensical ones here.
            # Check the number of egdes per vertex is correct.
            counter = Counter([i for j in combination for i in j])
            if counter != self.vertex_counts:
                continue

            # If are any self-reactions.
            if any(abs(i - j) == 0 for i, j in combination):
                continue

            topology_code = TopologyCode(
                vertex_map=combination,
                as_string=vmap_to_str(combination),
            )

            if (
                self.graph_set == "rx_nodoubles"
                and topology_code.contains_doubles()
            ):
                continue

            # Check for string done.
            if topology_code.get_as_string() in combinations_tested:
                continue

            combinations_tested.add(topology_code.get_as_string())

            # Convert TopologyCode to a graph.
            current_graph = topology_code.get_graph()

            # Check that graph for isomorphism with others graphs.
            passed_iso = True
            for tc in run_topology_codes:
                test_graph = tc.get_graph()

                if rx.is_isomorphic(current_graph, test_graph):
                    passed_iso = False
                    break

            if not passed_iso:
                continue

            run_topology_codes.append(topology_code)
            to_save.append(combination)
            logger.info("found one at %s", _)

        with self.graphs_path.open("w") as f:
            json.dump(to_save, f)

    def _three_type_algorithm(self) -> None:
        combinations_tested = set()
        run_topology_codes: list[TopologyCode] = []

        type1, type2, type3 = sorted(
            self.vertex_types_by_fg.keys(), reverse=True
        )

        itera1 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_fg[type1]
        ]

        rng = np.random.default_rng(seed=100)
        options1 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_fg[type2]
        ]
        options2 = [
            i
            for i in self.reactable_vertex_ids
            if i in self.vertex_types_by_fg[type3]
        ]

        to_save = []
        for _ in range(self.used_samples):
            # Merging options1 and options2 because they both bind to itera.
            mixed_options = options1 + options2
            rng.shuffle(mixed_options)

            # Build an edge selection.
            combination: abc.Sequence[tuple[int, int]] = [
                tuple(sorted((i, j)))  # type:ignore[misc]
                for i, j in zip(itera1, mixed_options, strict=True)
            ]

            # Need to check for nonsensical ones here.
            # Check the number of egdes per vertex is correct.
            counter = Counter([i for j in combination for i in j])
            if counter != self.vertex_counts:
                continue

            # If are any self-reactions.
            if any(abs(i - j) == 0 for i, j in combination):
                continue

            topology_code = TopologyCode(
                vertex_map=combination,
                as_string=vmap_to_str(combination),
            )

            if (
                self.graph_set == "rx_nodoubles"
                and topology_code.contains_doubles()
            ):
                continue

            # Check for string done.
            if topology_code.get_as_string() in combinations_tested:
                continue

            combinations_tested.add(topology_code.get_as_string())

            # Convert TopologyCode to a graph.
            current_graph = topology_code.get_graph()

            # Check that graph for isomorphism with others graphs.
            passed_iso = True
            for tc in run_topology_codes:
                test_graph = tc.get_graph()

                if rx.is_isomorphic(current_graph, test_graph):
                    passed_iso = False
                    break

            if not passed_iso:
                continue

            run_topology_codes.append(topology_code)
            to_save.append(combination)
            logger.info("found one at %s", _)

        with self.graphs_path.open("w") as f:
            json.dump(to_save, f)

    def _define_all_graphs(self) -> None:
        num_types = len(self.vertex_types_by_fg.keys())
        if num_types == 2:  # noqa: PLR2004
            self._two_type_algorithm()
        elif num_types == 3:  # noqa: PLR2004
            self._three_type_algorithm()
        else:
            msg = "not implemented for other types yet"
            raise RuntimeError(msg)

    def count_graphs(self) -> int:
        """Count completely connected graphs in iteration."""
        if not self.graphs_path.exists():
            self._define_all_graphs()

        with self.graphs_path.open("r") as f:
            all_graphs = json.load(f)

        count = 0
        for combination in all_graphs:
            topology_code = TopologyCode(combination)

            num_components = topology_code.get_number_connected_components()
            if num_components == self.allowed_num_components:
                count += 1

        return count

    def yield_graphs(self) -> abc.Generator[TopologyCode]:
        """Get constructed molecules from iteration.

        Yields only completely connected graphs.
        """
        if not self.graphs_path.exists():
            self._define_all_graphs()

        with self.graphs_path.open("r") as f:
            all_graphs = json.load(f)

        for combination in all_graphs:
            topology_code = TopologyCode(combination)

            num_components = topology_code.get_number_connected_components()
            if num_components == self.allowed_num_components:
                yield topology_code
