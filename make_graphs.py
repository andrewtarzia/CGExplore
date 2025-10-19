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

    # Three typers.
    multipliers = (1, 2)
    three_type_stoichiometries = tuple(
        (i, j, k) for i, j, k in it.product((1, 2, 3, 4), repeat=3)
    )

    for midx, fgnum1, fgnum2, fgnum3, stoich in it.product(
        multipliers, bbs, bbs, bbs, three_type_stoichiometries
    ):
        if fgnum1 in (fgnum2, fgnum3) or fgnum2 == fgnum3:
            continue
        fgnum1_, fgnum2_, fgnum3_ = sorted(
            (fgnum1, fgnum2, fgnum3), reverse=True
        )
        print(
            f"doing m={midx}, bbs={(fgnum1_, fgnum2_, fgnum3_)}, s={stoich} "
            f"-> m*s={(stoich[0] * midx, stoich[1] * midx, stoich[2] * midx)}"
        )

        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum1_]: midx * stoich[0],
                    bbs[fgnum2_]: midx * stoich[1],
                    bbs[fgnum3_]: midx * stoich[2],
                },
                graph_type=f"{midx * stoich[0]}-{fgnum1_}FG_"
                f"{midx * stoich[1]}-{fgnum2_}FG_"
                f"{midx * stoich[2]}-{fgnum3_}FG",
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
