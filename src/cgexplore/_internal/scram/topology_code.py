"""Define the topology code class."""

import logging
from collections import Counter, abc
from dataclasses import dataclass

import networkx as nx
import rustworkx as rx
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class TopologyCode:
    """Naming convention for topology graphs."""

    vertex_map: abc.Sequence[tuple[int, int]]
    as_string: str

    def get_nx_graph(self) -> nx.Graph:
        """Convert TopologyCode to a networkx graph."""
        graph = nx.MultiGraph()

        for vert in self.vertex_map:
            graph.add_edge(vert[0], vert[1])

        return graph

    def get_graph(self) -> rx.PyGraph:
        """Convert TopologyCode to a graph."""
        graph: rx.PyGraph = rx.PyGraph(multigraph=True)

        vertices = {
            vi: graph.add_node(vi)
            for vi in sorted({i for j in self.vertex_map for i in j})
        }

        for vert in self.vertex_map:
            nodea = graph[vertices[vert[0]]]
            nodeb = graph[vertices[vert[1]]]
            graph.add_edge(nodea, nodeb, None)

        return graph

    def get_weighted_graph(self) -> rx.PyGraph:
        """Convert TopologyCode to a graph."""
        graph: rx.PyGraph = rx.PyGraph(multigraph=False)

        vertices = {
            vi: graph.add_node(vi)
            for vi in sorted({i for j in self.vertex_map for i in j})
        }

        for vert in self.vertex_map:
            nodea = graph[vertices[vert[0]]]
            nodeb = graph[vertices[vert[1]]]
            if not graph.has_edge(nodea, nodeb):
                graph.add_edge(nodea, nodeb, 1)
            else:
                graph.add_edge(
                    nodea, nodeb, graph.get_edge_data(nodea, nodeb) + 1
                )

        return graph

    def edges_from_connection(
        self,
        vertex_prototypes: abc.Sequence[stk.Vertex],
    ) -> list[stk.Edge]:
        """Get stk Edges from topology code."""
        return [
            stk.Edge(
                id=i,
                vertex1=vertex_prototypes[pair[0]],
                vertex2=vertex_prototypes[pair[1]],
            )
            for i, pair in enumerate(self.vertex_map)
        ]

    def contains_doubles(self) -> bool:
        """True if the graph contains "double-walls"."""
        weighted_graph = self.get_weighted_graph()
        num_parallel_edges = len(
            [
                i
                for i in weighted_graph.edges()
                if i == 2  # noqa: PLR2004
            ]
        )

        filtered_paths = set()
        for node in weighted_graph.nodes():
            paths = list(
                rx.graph_all_simple_paths(
                    weighted_graph,
                    origin=node,  # type: ignore[call-arg]
                    to=node,  # type: ignore[call-arg]
                    cutoff=12,
                    min_depth=4,
                )
            )

            for path in paths:
                if (
                    tuple(path) not in filtered_paths
                    and tuple(path[::-1]) not in filtered_paths
                ):
                    filtered_paths.add(tuple(path))

        path_lengths = [len(i) - 1 for i in filtered_paths]
        counter = Counter(path_lengths)

        return num_parallel_edges != 0 or counter[4] != 0


@dataclass
class Constructed:
    """Container for constructed molecule and topology graph."""

    constructed_molecule: stk.ConstructedMolecule
    idx: int | None
    topology_code: TopologyCode
    mash_idx: int | None = None
