"""Copiable code from Recipe #5."""  # noqa: INP001

import logging

import rustworkx as rx
import stk

import cgexplore as cgx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run script."""
    bbs = {
        1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
        2: stk.BuildingBlock("BrCCCCBr", (stk.BromoFactory(),)),
        3: stk.BuildingBlock("BrCC(Br)CCBr", (stk.BromoFactory(),)),
        4: stk.BuildingBlock("BrCC(Br)CCC(Br)CCBr", (stk.BromoFactory(),)),
        5: stk.BuildingBlock("BrC1C(Br)C(Br)C(Br)C1Br", (stk.BromoFactory(),)),
    }

    knowns = {
        "6-2FG": (stk.cage.M3L3Triangle, {bbs[2]: 6}),
        "8-2FG": (stk.cage.M4L4Square, {bbs[2]: 8}),
        "2-3FG": (stk.cage.OnePlusOne, {bbs[3]: 2}),
        "4-3FG": (stk.cage.TwoPlusTwo, {bbs[3]: 4}),
        "x4-3FG": (stk.cage.M4L6Tetrahedron, {bbs[3]: 4}),
        "8-3FG": (stk.cage.FourPlusFour, {bbs[3]: 8}),
        "x8-3FG": (stk.cage.M4L4Tetrahedron, {bbs[3]: 8}),
        "2-3FG_3-2FG": (stk.cage.TwoPlusThree, {bbs[3]: 2, bbs[2]: 3}),
        "4-3FG_6-2FG": (stk.cage.FourPlusSix, {bbs[3]: 4, bbs[2]: 6}),
        "x4-3FG_6-2FG": (
            stk.cage.M4L6TetrahedronSpacer,
            {bbs[3]: 4, bbs[2]: 6},
        ),
        "xx4-3FG_6-2FG": (stk.cage.FourPlusSix2, {bbs[3]: 4, bbs[2]: 6}),
        "6-3FG_9-2FG": (stk.cage.SixPlusNine, {bbs[3]: 6, bbs[2]: 9}),
        "8-3FG_12-2FG": (stk.cage.EightPlusTwelve, {bbs[3]: 8, bbs[2]: 12}),
        "20-3FG_30-2FG": (stk.cage.TwentyPlusThirty, {bbs[3]: 20, bbs[2]: 30}),
        "1-4FG_2-2FG": (cgx.topologies.UnalignedM1L2, {bbs[4]: 1, bbs[2]: 2}),
        "2-4FG_4-2FG": (stk.cage.M2L4Lantern, {bbs[4]: 2, bbs[2]: 4}),
        "x2-4FG_4-2FG": (stk.cage.TwoPlusFour, {bbs[4]: 2, bbs[2]: 4}),
        "3-4FG_6-2FG": (stk.cage.M3L6, {bbs[4]: 3, bbs[2]: 6}),
        "x3-4FG_6-2FG": (stk.cage.ThreePlusSix, {bbs[4]: 3, bbs[2]: 6}),
        "4-4FG_8-2FG": (cgx.topologies.CGM4L8, {bbs[4]: 4, bbs[2]: 8}),
        "x4-4FG_8-2FG": (cgx.topologies.M4L82, {bbs[4]: 4, bbs[2]: 8}),
        "xx4-4FG_8-2FG": (stk.cage.FourPlusEight, {bbs[4]: 4, bbs[2]: 8}),
        "xxx4-4FG_8-2FG": (stk.cage.M4L8, {bbs[4]: 4, bbs[2]: 8}),
        "5-4FG_10-2FG": (stk.cage.FivePlusTen, {bbs[4]: 5, bbs[2]: 10}),
        "6-4FG_12-2FG": (stk.cage.M6L12Cube, {bbs[4]: 6, bbs[2]: 12}),
        "x6-4FG_12-2FG": (cgx.topologies.M6L122, {bbs[4]: 6, bbs[2]: 12}),
        "xx6-4FG_12-2FG": (stk.cage.SixPlusTwelve, {bbs[4]: 6, bbs[2]: 12}),
        "8-4FG_16-2FG": (stk.cage.EightPlusSixteen, {bbs[4]: 8, bbs[2]: 16}),
        "x8-4FG_16-2FG": (cgx.topologies.M8L162, {bbs[4]: 8, bbs[2]: 16}),
        "10-4FG_20-2FG": (stk.cage.TenPlusTwenty, {bbs[4]: 10, bbs[2]: 20}),
        "12-4FG_24-2FG": (cgx.topologies.CGM12L24, {bbs[4]: 12, bbs[2]: 24}),
        "x12-4FG_24-2FG": (stk.cage.M12L24, {bbs[4]: 12, bbs[2]: 24}),
        "24-4FG_48-2FG": (stk.cage.M24L48, {bbs[4]: 24, bbs[2]: 48}),
        "6-4FG_8-3FG": (stk.cage.SixPlusEight, {bbs[4]: 6, bbs[3]: 8}),
        "x6-4FG_8-3FG": (stk.cage.M8L6Cube, {bbs[4]: 6, bbs[3]: 8}),
        "3-4FG_8-3FG": (stk.cage.M6L2L3Prism, {bbs[4]: 3, bbs[3]: 8}),
        "12-5FG_30-2FG": (stk.cage.TwelvePlusThirty, {bbs[5]: 12, bbs[2]: 30}),
    }

    known_failures = (
        "3-4FG_8-3FG",
        "20-3FG_30-2FG",
        "24-4FG_48-2FG",
        "12-5FG_30-2FG",
    )

    for known_, (tfun, sele_bbs) in knowns.items():
        found = False
        stk_topology_code, _ = cgx.scram.get_stk_topology_code(tfun)
        iterator = cgx.scram.TopologyIterator(building_block_counts=sele_bbs)
        if not iterator.graph_exists():
            logger.info("%s graph not built yet", known_)
            if known_ in known_failures:
                logger.info("----> it is ok though, we knew about this one!")
            continue

        for tc in iterator.yield_graphs():
            if rx.is_isomorphic(stk_topology_code.get_graph(), tc.get_graph()):
                found = True
                break

        if found:
            logger.info(
                "found stk graph for %s in the %s graphs",
                known_,
                iterator.count_graphs(),
            )

        else:
            logger.info(
                "not found stk graph for %s in the %s graphs",
                known_,
                iterator.count_graphs(),
            )
            if known_ in known_failures:
                logger.info("----> it is ok though, we knew about this one!")


if __name__ == "__main__":
    main()
