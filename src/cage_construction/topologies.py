#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of cages.

Author: Andrew Tarzia

"""

import stk


def cage_topology_options():
    topologies = {
        "TwoPlusThree": stk.cage.TwoPlusThree,
        "FourPlusSix": stk.cage.FourPlusSix,
        "FourPlusSix2": stk.cage.FourPlusSix2,
        "SixPlusNine": stk.cage.SixPlusNine,
        "EightPlusTwelve": stk.cage.EightPlusTwelve,
        # "TwentyPlusThirty": stk.cage.TwentyPlusThirty,
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
