#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of cages.

Author: Andrew Tarzia

"""

import stk


def cage_topology_options(fg_set):
    if fg_set == "2p3":
        topologies = {
            "TwoPlusThree": stk.cage.TwoPlusThree,
            "FourPlusSix": stk.cage.FourPlusSix,
            "FourPlusSix2": stk.cage.FourPlusSix2,
            "SixPlusNine": stk.cage.SixPlusNine,
            "EightPlusTwelve": stk.cage.EightPlusTwelve,
            # "TwentyPlusThirty": stk.cage.TwentyPlusThirty,
        }
    if fg_set == "2p4":
        topologies = {
            "M2L4": stk.cage.M2L4Lantern,
            "M3L6": stk.cage.M3L6,
            "M4L8": stk.cage.M4L8,
            "M6L12": stk.cage.M6L12Cube,
        }

    return topologies


class CGM12L24(stk.cage.M12L24):
    def _get_scale(self, building_block_vertices):
        return 15

    def get_vertex_alignments(self):
        return self._vertex_alignments


def unsymm_topology_options():
    topologies = {
        "CGM12L24": CGM12L24,
    }

    return topologies
