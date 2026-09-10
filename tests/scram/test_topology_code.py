import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx

import cgexplore as cgx

from .case_data import CaseData


def test_topology_code_nxgraph(graph_data: CaseData) -> None:
    """Test topology code methods."""
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )

    for tc in iterator.yield_graphs():
        nxgml_file = (
            graph_directory / f"nx_{graph_data.graph_type}_{tc.idx}.gml.gz"
        )
        if not nxgml_file.exists():
            nx.write_gml(tc.get_nx_graph(), nxgml_file)

        test = nx.read_gml(nxgml_file)
        new_nodes = [int(i) for i in test.nodes]
        assert new_nodes == list(tc.get_nx_graph().nodes)
        assert test.graph == tc.get_nx_graph().graph
        new_adj = {
            int(i): {int(k): dict(x) for k, x in j.items()}
            for i, j in test.adj.items()
        }
        assert new_adj == dict(tc.get_nx_graph().adj)


def test_topology_code_rxgraph(graph_data: CaseData) -> None:
    """Test topology code methods.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )

    for tc in iterator.yield_graphs():
        rxgml_file = (
            graph_directory / f"rx_{graph_data.graph_type}_{tc.idx}.gml"
        )
        if not rxgml_file.exists():
            rx.write_graphml(tc.get_graph(), str(rxgml_file))

        test: rx.PyGraph = rx.read_graphml(str(rxgml_file))[0]  # type:ignore[assignment]

        # Odd formatting of nodes, caution.
        assert [
            int(i["id"].strip("n")) for i in test.nodes()
        ] == tc.get_graph().nodes()
        assert np.allclose(
            rx.graph_adjacency_matrix(test),
            rx.graph_adjacency_matrix(tc.get_graph()),
        )

        rxgml_file = (
            graph_directory / f"rxw_{graph_data.graph_type}_{tc.idx}.gml"
        )
        if not rxgml_file.exists():
            rx.write_graphml(tc.get_weighted_graph(), str(rxgml_file))

        test = rx.read_graphml(str(rxgml_file))[0]  # type:ignore[assignment]

        assert [
            int(i["id"].strip("n")) for i in test.nodes()
        ] == tc.get_weighted_graph().nodes()
        assert np.allclose(
            rx.graph_adjacency_matrix(test),
            rx.graph_adjacency_matrix(tc.get_weighted_graph()),
        )


def summarise_topology_code(
    topology_code: cgx.scram.TopologyCode,
    name: str,
    figure_dir: pathlib.Path,
) -> None:
    """Use networkx to layout and summarise a graph."""
    g = topology_code.get_nx_graph()
    degree_sequence = sorted((d for n, d in g.degree()), reverse=True)

    fig = plt.figure(figsize=(8, 5))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(1, 2)

    ax0 = fig.add_subplot(axgrid[:, :1])
    gcc = g.subgraph(max(nx.connected_components(g), key=len))
    pos = nx.spring_layout(gcc, seed=10396953)
    nx.draw_networkx_nodes(gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(gcc, pos, ax=ax0, alpha=0.4)
    ax0.tick_params(axis="both", which="major", labelsize=16)
    ax0.set_title(f"Connected components of {name}", fontsize=16)
    ax0.set_axis_off()

    ax2 = fig.add_subplot(axgrid[:, 1:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_title("Degree histogram", fontsize=16)
    ax2.set_xlabel("Degree", fontsize=16)
    ax2.set_ylabel("# of Nodes", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / f"g_{name}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def test_topology_code_as_string(graph_data: CaseData) -> None:
    """Test topology code methods.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )

    for tc in iterator.yield_graphs():
        str_file = (
            graph_directory / f"str_{graph_data.graph_type}_{tc.idx}.txt"
        )
        if not str_file.exists():
            with str_file.open("w") as f:
                f.write(tc.get_as_string())

        with str_file.open("r") as f:
            lines = f.readlines()

        assert lines[0] == tc.get_as_string()

        summarise_topology_code(
            topology_code=tc,
            name=f"str_{graph_data.graph_type}_{tc.idx}",
            figure_dir=graph_directory,
        )


def test_topology_code_components(graph_data: CaseData) -> None:
    """Test topology code methods.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )

    for tc in iterator.yield_graphs():
        assert tc.get_number_connected_components() == 1


def test_topology_code_doubles(graph_data: CaseData) -> None:
    """Test topology code methods.

    Parameters:

        graph_data:
            The graph data.

    """
    graph_directory = pathlib.Path(__file__).resolve().parent / "test_graphs"
    iterator = cgx.scram.TopologyIterator(
        building_block_counts=graph_data.building_block_counts,
        max_samples=graph_data.max_samples,
        # Use known graphs.
        graph_directory=graph_directory,
    )

    for tc in iterator.yield_graphs():
        print(
            graph_data.graph_type,
            tc.idx,
            tc.contains_doubles(),
            tc.contains_parallels(),
        )
        assert tc.contains_doubles() is graph_data.doubles[tc.idx]
        assert tc.contains_parallels() is graph_data.parallels[tc.idx]
