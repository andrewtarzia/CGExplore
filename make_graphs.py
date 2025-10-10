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
    multipliers = range(1, 11)

    # One typers.
    for midx, fgnum in it.product(multipliers, bbs):
        print(f"doing m={midx}, bb={fgnum}")
        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum]: midx,
                },
                graph_type=f"{midx}-{fgnum}FG",
                graph_set="rxx",
            )
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            print(f"      m={midx}, bb={fgnum} failed")

    # Two typers.
    multipliers = range(1, 13)
    two_type_stoichiometries = ((1, 2), (2, 3), (3, 4))
    for midx, fgnum1, fgnum2, stoich in it.product(
        multipliers, bbs, bbs, two_type_stoichiometries
    ):
        if fgnum1 == fgnum2:
            continue
        fgnum1_, fgnum2_ = sorted((fgnum1, fgnum2), reverse=True)
        print(
            f"doing m={midx}, bbs={(fgnum1_, fgnum2_)}, s={stoich} "
            f"-> m*s={(stoich[0] * midx, stoich[1] * midx)}"
        )

        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum1_]: midx * stoich[0],
                    bbs[fgnum2_]: midx * stoich[1],
                },
                graph_type=f"{midx * stoich[0]}-{fgnum1_}FG_"
                f"{midx * stoich[1]}-{fgnum2_}FG",
                graph_set="rxx",
            )
            logger.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            print(
                f"      m={midx}, bbs={(fgnum1_, fgnum2_)}, s={stoich} failed"
            )

    raise SystemExit("delete me!")


if __name__ == "__main__":
    main()
