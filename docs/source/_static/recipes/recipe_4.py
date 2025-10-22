"""Copiable code from Recipe #4."""  # noqa: INP001

import argparse
import logging
import pathlib

import stk

import cgexplore as cgx

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def main() -> None:
    """Run script."""
    args = _parse_args()
    # Define a working directory.
    wd = (
        pathlib.Path(__file__).resolve().parent
        / ".."
        / ".."
        / "recipes"
        / "recipe_4_output"
    )
    wd.mkdir(exist_ok=True)

    tritopic_building_block = stk.BuildingBlock(
        smiles="C1=C(C=C(C=C1C=O)C=O)C=O",
        functional_groups=[stk.AldehydeFactory()],
    )
    ditopic_building_block = stk.BuildingBlock(
        smiles="NC1CCCCC1N",
        functional_groups=[stk.PrimaryAminoFactory()],
    )

    # Currently, this definition is up to the user, but we will make this
    # uniform soon.
    building_block_library = {
        "ditopic": ditopic_building_block,
        "tritopic": tritopic_building_block,
    }

    # Currently, this definition is up to the user, but we will make this
    # uniform soon.
    systems = {
        "s1": {
            # Always order with the most functional groups first.
            "stoichiometry_map": {"tritopic": 2, "ditopic": 3},
            "multipliers": (1, 2),
        },
    }

    syst_d = systems["s1"]
    iterators = {}
    if args.run:
        for multiplier in syst_d["multipliers"]:
            # Automate the graph type naming.
            graph_type = cgx.scram.generate_graph_type(
                stoichiometry_map=syst_d["stoichiometry_map"],  # type:ignore[arg-type]
                multiplier=int(multiplier),
                bb_library=building_block_library,
            )

            # Define the iterator.
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    building_block_library[name]: stoich * multiplier
                    for name, stoich in syst_d["stoichiometry_map"].items()  # type:ignore[attr-defined]
                },
                graph_type=graph_type,
                # Use a known graph set.
                graph_set="rxx",
            )
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
            iterators[multiplier] = iterator

        for multiplier in syst_d["multipliers"]:
            iterator = iterators[multiplier]
            for idx, topology_code in enumerate(iterator.yield_graphs()):
                # Filter graphs for 1-loops.
                if topology_code.contains_parallels():
                    continue

                name = f"s1_{multiplier}_{idx}"

                # Use vertex set regraphing.
                constructed_molecule = cgx.scram.get_vertexset_molecule(
                    graph_type="kamada",
                    scale=5,
                    topology_code=topology_code,
                    iterator=iterator,
                    bb_config=None,
                )
                if not (wd / f"{name}_unopt.mol").exists():
                    constructed_molecule.write(wd / f"{name}_unopt.mol")

                # Implement optimisation workflows!

                # And then do some analysis!


if __name__ == "__main__":
    main()
