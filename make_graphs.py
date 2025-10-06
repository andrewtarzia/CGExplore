import itertools as it
import logging

import stk

import cgexplore as cgx

logger = logging.getLogger(__name__)


def main() -> None:
    """Run script."""
    bbs = {
        1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
        2: stk.BuildingBlock("BrCCCCBr", (stk.BromoFactory(),)),
        3: stk.BuildingBlock("BrCC(Br)CCBr", (stk.BromoFactory(),)),
        4: stk.BuildingBlock("BrCC(Br)CCC(Br)CCBr", (stk.BromoFactory(),)),
    }
    multipliers = range(1, 13)

    # One typers.
    for midx, fgnum in it.product(multipliers, bbs):
        print(f"doing {midx}, {fgnum}")
        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum]: midx,
                },
                graph_type=f"{midx}-{fgnum}FG",
                graph_set="rxx",
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            print(f" {midx}, {fgnum} failed")

    # Two typers.
    two__type_stoichiometries = ((1, 2), (2, 3), (3, 4))
    for midx, (fgnum1, fgnum2), stoich in it.product(
        multipliers, bbs, bbs, two__type_stoichiometries
    ):
        if fgnum1 == fgnum2:
            continue
        fgnum1_, fgnum2_ = sorted(fgnum1, fgnum2)
        print(f"doing {midx}, {(fgnum1_, fgnum2_)}, {stoich}")
        print((stoich[0] * midx, stoich[1] * midx))
        continue
        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum1_]: midx * stoich[0],
                    bbs[fgnum2_]: midx * stoich[1],
                },
                graph_type=f"{midx * stoich[0]}-{fgnum}FG_"
                f"{midx * stoich[1]}-{fgnum}FG",
                graph_set="rxx",
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            print(f" {midx}, {(fgnum1, fgnum2)}, {stoich} failed")

    raise SystemExit("delete me!")


if __name__ == "__main__":
    main()
