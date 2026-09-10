"""Copiable code from Recipe #5."""  # noqa: INP001

import logging
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stk

import cgexplore as cgx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


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
    gcc = g.subgraph(
        sorted(nx.connected_components(g), key=len, reverse=True)[0]
    )
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


def main() -> None:
    """Run script."""
    # Define working directories.
    wd = (
        pathlib.Path(__file__).resolve().parent
        / ".."
        / ".."
        / "recipes"
        / "recipe_8_output"
    )
    figure_dir = wd / "figures"
    cgx.utilities.check_directory(figure_dir)
    bbs = {
        1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
        2: stk.BuildingBlock("BrCCCCBr", (stk.BromoFactory(),)),
        3: stk.BuildingBlock("BrCC(Br)CCBr", (stk.BromoFactory(),)),
    }

    system = {bbs[3]: 3, bbs[2]: 4, bbs[1]: 1}

    iterator = cgx.scram.TopologyIterator(building_block_counts=system)

    for tc in iterator.yield_graphs():
        summarise_topology_code(
            topology_code=tc,
            name=f"graph_{tc.idx}",
            figure_dir=figure_dir,
        )


if __name__ == "__main__":
    main()
