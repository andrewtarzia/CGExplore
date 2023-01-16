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
            "2P3": stk.cage.TwoPlusThree,
            "4P6": stk.cage.FourPlusSix,
            "4P62": stk.cage.FourPlusSix2,
            "6P9": stk.cage.SixPlusNine,
            "8P12": stk.cage.EightPlusTwelve,
        }
    if fg_set == "2p4":
        topologies = {
            "2P4": stk.cage.M2L4Lantern,
            "3P6": stk.cage.M3L6,
            "4P8": stk.cage.M4L8,
            "6P12": stk.cage.M6L12Cube,
            "8P6": stk.cage.EightPlusSixteen,
            "12PL24": CGM12L24,
        }
    if fg_set == "3p3":
        topologies = {
            "2P2": stk.cage.TwoPlusTwo,
            "4P4": stk.cage.FourPlusFour,
        }
    if fg_set == "3p4":
        topologies = {
            "6P8": stk.cage.SixPlusEight,
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
