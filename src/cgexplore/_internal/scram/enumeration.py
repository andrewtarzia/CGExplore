"""Define classes for enumeration of graphs."""

import json
import logging
import pathlib
from collections import Counter, abc, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rustworkx as rx
import stk

from cgexplore._internal.topologies.custom_topology import CustomTopology
from cgexplore._internal.topologies.graphs import (
    CGM4L8,
    CGM12L24,
    UnalignedM1L2,
)

from .topology_code import Constructed, TopologyCode
from .utilities import points_on_sphere, vmap_to_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class Scrambler:
    """Iterate over topology graphs.

    This is an old version of this code, which I do not recommend using over
    the `TopologyIterator`.

    TODO: Clean-up and remove this class.

    """

    def __init__(
        self,
        tetra_bb: stk.BuildingBlock,
        converging_bb: stk.BuildingBlock,
        diverging_bb: stk.BuildingBlock,
        multiplier: int,
        stoichiometry: tuple[int, int, int],
    ) -> None:
        """Initialize."""
        self._building_blocks: dict[stk.BuildingBlock, abc.Sequence[int]]
        self._underlying_topology: type[stk.cage.Cage]

        if stoichiometry == (1, 1, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0,),
                    converging_bb: (1,),
                    diverging_bb: (2,),
                }
                self._underlying_topology = UnalignedM1L2
                self._scale_multiplier = 2
                self._skip_initial = True

            elif multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    converging_bb: (2, 3),
                    diverging_bb: (4, 5),
                }
                self._underlying_topology = stk.cage.M2L4Lantern
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 3:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5),
                    diverging_bb: (6, 7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    converging_bb: (4, 5, 6, 7),
                    diverging_bb: (8, 9, 10, 11),
                }
                self._underlying_topology = CGM4L8
                self._scale_multiplier = 2
                self._skip_initial = False

        if stoichiometry == (4, 2, 3):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5, 6),
                    diverging_bb: (7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3, 4, 5),
                    converging_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                    diverging_bb: (14, 15, 16, 17),
                }
                self._underlying_topology = stk.cage.M6L12Cube
                self._scale_multiplier = 5
                self._skip_initial = False

            elif multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(12),
                    converging_bb: range(12, 28),
                    diverging_bb: range(28, 36),
                }
                self._underlying_topology = CGM12L24
                self._scale_multiplier = 5
                self._skip_initial = False

        self._init_vertex_prototypes = deepcopy(
            self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._init_edge_prototypes = deepcopy(
            self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._vertices = tuple(
            stk.cage.UnaligningVertex(
                id=i.get_id(),
                position=i.get_position(),
                aligner_edge=i.get_aligner_edge(),
                use_neighbor_placement=i.use_neighbor_placement(),
            )
            for i in self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._edges = tuple(
            stk.Edge(
                id=i.get_id(),
                vertex1=self._vertices[i.get_vertex1_id()],
                vertex2=self._vertices[i.get_vertex2_id()],
            )
            for i in self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._num_scrambles = 200
        self._num_mashes = 2

        self._define_underlying()
        self._beta = 10

    def _define_underlying(self) -> None:
        self._vertex_connections: dict[int, int] = {}
        for edge in self._init_edge_prototypes:
            if edge.get_vertex1_id() not in self._vertex_connections:
                self._vertex_connections[edge.get_vertex1_id()] = 0
            self._vertex_connections[edge.get_vertex1_id()] += 1

            if edge.get_vertex2_id() not in self._vertex_connections:
                self._vertex_connections[edge.get_vertex2_id()] = 0
            self._vertex_connections[edge.get_vertex2_id()] += 1

        self._type1 = [
            i
            for i in self._vertex_connections
            if self._vertex_connections[i] == 4  # noqa: PLR2004
        ]
        self._type2 = [
            i
            for i in self._vertex_connections
            if self._vertex_connections[i] == 2  # noqa: PLR2004
        ]

        combination = [
            tuple(sorted((i.get_vertex1_id(), i.get_vertex2_id())))
            for i in self._init_edge_prototypes
        ]
        self._initial_topology_code = TopologyCode(
            vertex_map=combination,
            as_string=vmap_to_str(combination),
        )

    def get_num_building_blocks(self) -> int:
        """Get number of building blocks."""
        return len(self._init_vertex_prototypes)

    def get_num_scrambles(self) -> int:
        """Get num. scrambles algorithm."""
        return self._num_scrambles

    def get_num_mashes(self) -> int:
        """Get num. mashes algorithm."""
        return self._num_mashes

    def get_constructed_molecules(self) -> abc.Generator[Constructed]:  # noqa: C901, PLR0912, PLR0915
        """Get constructed molecules from iteration."""
        combinations_tested = set()
        rng = np.random.default_rng(seed=100)
        count = 0

        if not self._skip_initial:
            try:
                constructed = stk.ConstructedMolecule(
                    self._underlying_topology(
                        building_blocks=self._building_blocks,
                        vertex_positions=None,
                    )
                )

                yield Constructed(
                    constructed_molecule=constructed,
                    idx=0,
                    topology_code=self._initial_topology_code,
                )
            except ValueError:
                pass
            combinations_tested.add(self._initial_topology_code.as_string)

            # Scramble the vertex positions.
            for _ in range(self._num_mashes):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10
                    for j, i in enumerate(self._vertices)
                }

                count += 1
                try:
                    # Try with aligning vertices.
                    constructed = stk.ConstructedMolecule(
                        self._underlying_topology(
                            building_blocks=self._building_blocks,
                            vertex_positions=None,
                        )
                    )
                    yield Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=self._initial_topology_code,
                    )
                except ValueError:
                    # Try with unaligning.
                    try:
                        constructed = stk.ConstructedMolecule(
                            self._underlying_topology(
                                building_blocks=self._building_blocks,
                                vertex_positions=None,
                            )
                        )
                        yield Constructed(
                            constructed_molecule=constructed,
                            idx=count,
                            topology_code=self._initial_topology_code,
                        )
                    except ValueError:
                        pass

        for _ in range(self._num_scrambles):
            # Scramble the edges.
            remaining_connections = deepcopy(self._vertex_connections)
            available_type1s = deepcopy(self._type1)
            available_type2s = deepcopy(self._type2)

            new_edges: list[stk.Edge] = []
            combination = []
            for _ in range(len(self._init_edge_prototypes)):
                try:
                    vertex1 = rng.choice(available_type1s)
                    vertex2 = rng.choice(available_type2s)
                except ValueError:
                    if len(remaining_connections) == 1:
                        vertex1 = next(iter(remaining_connections.keys()))
                        vertex2 = next(iter(remaining_connections.keys()))

                new_edge = stk.Edge(
                    id=len(new_edges),
                    vertex1=self._vertices[vertex1],
                    vertex2=self._vertices[vertex2],
                )
                new_edges.append(new_edge)

                remaining_connections[vertex1] += -1
                remaining_connections[vertex2] += -1

                remaining_connections = {
                    i: remaining_connections[i]
                    for i in remaining_connections
                    if remaining_connections[i] != 0
                }

                available_type1s = [
                    i for i in self._type1 if i in remaining_connections
                ]
                available_type2s = [
                    i for i in self._type2 if i in remaining_connections
                ]
                combination.append(tuple(sorted((vertex1, vertex2))))

            topology_code = TopologyCode(
                vertex_map=combination,
                as_string=vmap_to_str(combination),
            )

            # If you broke early, do not try to build.
            if len(new_edges) != len(self._edges):
                continue

            if topology_code.as_string in combinations_tested:
                continue

            combinations_tested.add(topology_code.as_string)

            count += 1
            try:
                # Try with aligning vertices.
                constructed = stk.ConstructedMolecule(
                    CustomTopology(  # type:ignore[arg-type]
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._init_vertex_prototypes,
                        edge_prototypes=new_edges,
                        vertex_alignments=None,
                        vertex_positions=None,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                yield Constructed(
                    constructed_molecule=constructed,
                    idx=count,
                    topology_code=topology_code,
                )
            except ValueError:
                # Try with unaligning.
                try:
                    constructed = stk.ConstructedMolecule(
                        CustomTopology(  # type:ignore[arg-type]
                            building_blocks=self._building_blocks,
                            vertex_prototypes=self._vertices,
                            edge_prototypes=new_edges,
                            vertex_alignments=None,
                            vertex_positions=None,
                            scale_multiplier=self._scale_multiplier,
                        )
                    )
                    yield Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=topology_code,
                    )
                except ValueError:
                    pass

            # Scramble the vertex positions.
            for _ in range(self._num_mashes):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10
                    for j, i in enumerate(self._vertices)
                }

                count += 1
                try:
                    # Try with aligning vertices.
                    constructed = stk.ConstructedMolecule(
                        CustomTopology(  # type:ignore[arg-type]
                            building_blocks=self._building_blocks,
                            vertex_prototypes=self._init_vertex_prototypes,
                            edge_prototypes=new_edges,
                            vertex_alignments=None,
                            vertex_positions=new_vertex_positions,
                            scale_multiplier=self._scale_multiplier,
                        )
                    )
                    yield Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=topology_code,
                    )
                except ValueError:
                    # Try with unaligning.
                    try:
                        constructed = stk.ConstructedMolecule(
                            CustomTopology(  # type:ignore[arg-type]
                                building_blocks=self._building_blocks,
                                vertex_prototypes=self._vertices,
                                edge_prototypes=new_edges,
                                vertex_alignments=None,
                                vertex_positions=new_vertex_positions,
                                scale_multiplier=self._scale_multiplier,
                            )
                        )
                        yield Constructed(
                            constructed_molecule=constructed,
                            idx=count,
                            topology_code=topology_code,
                        )
                    except ValueError:
                        pass

    def _get_random_topology_code(
        self, generator: np.random.Generator
    ) -> TopologyCode:
        remaining_connections = deepcopy(self._vertex_connections)
        available_type1s = deepcopy(self._type1)
        available_type2s = deepcopy(self._type2)

        vertex_map = []
        for _ in range(len(self._init_edge_prototypes)):
            try:
                vertex1 = generator.choice(available_type1s)
                vertex2 = generator.choice(available_type2s)
            except ValueError:
                if len(remaining_connections) == 1:
                    vertex1 = next(iter(remaining_connections.keys()))
                    vertex2 = next(iter(remaining_connections.keys()))

            vertex_map.append(tuple(sorted((vertex1, vertex2))))

            remaining_connections[vertex1] += -1
            remaining_connections[vertex2] += -1
            remaining_connections = {
                i: remaining_connections[i]
                for i in remaining_connections
                if remaining_connections[i] != 0
            }
            available_type1s = [
                i for i in self._type1 if i in remaining_connections
            ]
            available_type2s = [
                i for i in self._type2 if i in remaining_connections
            ]

        return TopologyCode(
            vertex_map=vertex_map, as_string=vmap_to_str(vertex_map)
        )

    def _shuffle_topology_code(
        self,
        topology_code: TopologyCode,
        generator: np.random.Generator,
    ) -> TopologyCode:
        old_vertex_map = topology_code.vertex_map

        size = (
            generator.integers(
                low=1, high=int(len(old_vertex_map) / 2), size=1
            )
            * 2
        )

        swaps = list(
            generator.choice(
                range(len(old_vertex_map)),
                size=int(size[0]),
                replace=False,
            )
        )

        new_vertex_map = []
        already_done = set()
        for vmap_idx in range(len(old_vertex_map)):
            if vmap_idx in already_done:
                continue
            if vmap_idx in swaps:
                possible_ids = [i for i in swaps if i != vmap_idx]
                other_idx = generator.choice(possible_ids, size=1)[0]

                # Swap connections.
                old1 = old_vertex_map[vmap_idx]
                old2 = old_vertex_map[other_idx]

                new1 = (old1[0], old2[1])
                new2 = (old2[0], old1[1])

                new_vertex_map.append(new1)
                new_vertex_map.append(new2)
                swaps = [i for i in swaps if i not in (vmap_idx, other_idx)]

                already_done.add(other_idx)
            else:
                new_vertex_map.append(old_vertex_map[vmap_idx])

        return TopologyCode(
            vertex_map=new_vertex_map, as_string=vmap_to_str(new_vertex_map)
        )

    def get_topology(
        self,
        input_topology_code: TopologyCode | None,
        generator: np.random.Generator,
    ) -> Constructed | None:
        """Get a topology."""
        if input_topology_code is None:
            topology_code = self._get_random_topology_code(generator=generator)
        else:
            topology_code = self._shuffle_topology_code(
                topology_code=input_topology_code,
                generator=generator,
            )

        try:
            # Try with aligning vertices.
            constructed = stk.ConstructedMolecule(
                CustomTopology(  # type:ignore[arg-type]
                    building_blocks=self._building_blocks,
                    vertex_prototypes=self._init_vertex_prototypes,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=self._init_vertex_prototypes[vmap[0]],
                            vertex2=self._init_vertex_prototypes[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=None,
                    scale_multiplier=self._scale_multiplier,
                )
            )
            return Constructed(
                constructed_molecule=constructed,
                idx=None,
                topology_code=topology_code,
            )
        except ValueError:
            # Try with unaligning.
            try:
                constructed = stk.ConstructedMolecule(
                    CustomTopology(  # type:ignore[arg-type]
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._vertices,
                        edge_prototypes=tuple(
                            stk.Edge(
                                id=i,
                                vertex1=self._vertices[vmap[0]],
                                vertex2=self._vertices[vmap[1]],
                            )
                            for i, vmap in enumerate(topology_code.vertex_map)
                        ),
                        vertex_alignments=None,
                        vertex_positions=None,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                return Constructed(
                    constructed_molecule=constructed,
                    idx=None,
                    topology_code=topology_code,
                )
            except ValueError:
                return None

    def get_mashed_topology(
        self,
        topology_code: TopologyCode,
        generator: np.random.Generator,
    ) -> Constructed | None:
        """Get a mashed topology, where vertex coordinates are changed."""
        coordinates = generator.random(size=(len(self._vertices), 3))
        new_vertex_positions = {
            j: coordinates[j] * 10 for j, i in enumerate(self._vertices)
        }

        try:
            # Try with aligning vertices.
            constructed = stk.ConstructedMolecule(
                CustomTopology(  # type:ignore[arg-type]
                    building_blocks=self._building_blocks,
                    vertex_prototypes=self._init_vertex_prototypes,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=self._init_vertex_prototypes[vmap[0]],
                            vertex2=self._init_vertex_prototypes[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=new_vertex_positions,
                    scale_multiplier=self._scale_multiplier,
                )
            )
            return Constructed(
                constructed_molecule=constructed,
                idx=None,
                topology_code=topology_code,
            )
        except ValueError:
            # Try with unaligning.
            try:
                constructed = stk.ConstructedMolecule(
                    CustomTopology(  # type:ignore[arg-type]
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._vertices,
                        edge_prototypes=tuple(
                            stk.Edge(
                                id=i,
                                vertex1=self._vertices[vmap[0]],
                                vertex2=self._vertices[vmap[1]],
                            )
                            for i, vmap in enumerate(topology_code.vertex_map)
                        ),
                        vertex_alignments=None,
                        vertex_positions=new_vertex_positions,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                return Constructed(
                    constructed_molecule=constructed,
                    idx=None,
                    topology_code=topology_code,
                )
            except ValueError:
                return None


@dataclass
class TopologyIterator:
    """Iterate over topology graphs.

    This is the latest version, but without good symmetry and graph checks,
    this can over produce structures.

    """

    building_block_counts: dict[stk.BuildingBlock, int]
    graph_type: str
    graph_set: Literal["rx", "nx", "rx_nodoubles"] = "rx"
    scale_multiplier = 5
    allowed_num_components: int = 1
    max_samples: int | None = None

    def __post_init__(self) -> None:  # noqa: PLR0915, PLR0912, C901
        """Initialize."""
        match self.graph_set:
            case "rx":
                self.graphs_path = (
                    pathlib.Path(__file__).resolve().parent
                    / "known_graphs"
                    / f"rx_{self.graph_type}.json"
                )
                if self.max_samples is None:
                    self.used_samples = int(1e4)

            case "rx_nodoubles":
                self.graphs_path = (
                    pathlib.Path(__file__).resolve().parent
                    / "known_graphs"
                    / f"rxnd_{self.graph_type}.json"
                )
                if self.max_samples is None:
                    self.used_samples = int(1e5)

            case "nx":
                self.graphs_path = (
                    pathlib.Path(__file__).resolve().parent
                    / "known_graphs"
                    / f"g_{self.graph_type}.json"
                )
                if not self.graphs_path.exists():
                    msg = "building graphs with nx no longer available"
                    raise RuntimeError(msg)

            case _:
                raise RuntimeError

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
            if topology_code.as_string in combinations_tested:
                continue

            combinations_tested.add(topology_code.as_string)

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
            logging.info("found one at %s", _)

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
            if topology_code.as_string in combinations_tested:
                continue

            combinations_tested.add(topology_code.as_string)

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
            logging.info("found one at %s", _)

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
            topology_code = TopologyCode(
                vertex_map=combination,
                as_string=vmap_to_str(combination),
            )

            num_components = rx.number_connected_components(
                topology_code.get_graph()
            )

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
            topology_code = TopologyCode(
                vertex_map=combination,
                as_string=vmap_to_str(combination),
            )

            num_components = rx.number_connected_components(
                topology_code.get_graph()
            )
            if num_components == self.allowed_num_components:
                yield topology_code
