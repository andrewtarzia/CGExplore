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
        }
    if fg_set == "2p4":
        topologies = {
            "TwoPlusFour": stk.cage.TwoPlusFour,
            "ThreePlusSix": stk.cage.ThreePlusSix,
            "FourPlusEight": stk.cage.FourPlusEight,
            "SixPlusTwelve": stk.cage.SixPlusTwelve,
        }
    if fg_set == "3p3":
        topologies = {
            "TwoPlusTwo": stk.cage.TwoPlusTwo,
            "FourPlusFour": stk.cage.FourPlusFour,
        }
    if fg_set == "3p4":
        topologies = {
            "SixPlusEight": stk.cage.SixPlusEight,
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
