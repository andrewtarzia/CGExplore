# Distributed under the terms of the MIT License.

"""Classes of topologies of cages."""

from typing import assert_never

import stk
from cgexplore.topologies import CGM4L8, CGM12L24, M4L82


def cage_topology_options(fg_set: str) -> dict[str, stk.TopologyGraph]:
    """Cage topology options for this project."""
    match fg_set:
        case "2p3":
            topologies = {
                "2P3": stk.cage.TwoPlusThree,
                "4P6": stk.cage.FourPlusSix,
                "4P62": stk.cage.FourPlusSix2,
                "6P9": stk.cage.SixPlusNine,
                "8P12": stk.cage.EightPlusTwelve,
            }

        case "2p4":
            topologies = {
                "2P4": stk.cage.M2L4Lantern,
                "3P6": stk.cage.M3L6,
                "4P8": CGM4L8,
                "4P82": M4L82,
                "6P12": stk.cage.M6L12Cube,
                "8P16": stk.cage.EightPlusSixteen,
                "12P24": CGM12L24,
            }
        case "3p3":
            topologies = {
                "2P2": stk.cage.TwoPlusTwo,
                "4P4": stk.cage.FourPlusFour,
            }

        case "3p4":
            topologies = {"6P8": stk.cage.SixPlusEight}

        case _ as unreachable:
            assert_never(unreachable)

    return topologies
