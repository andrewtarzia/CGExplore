#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of cages.

Author: Andrew Tarzia

"""

import stk


class CGFourPlusSix(stk.cage.FourPlusSix):
    def _get_scale(self, building_block_vertices):
        return 5


class CGFourPlusSix2(stk.cage.FourPlusSix2):
    def _get_scale(self, building_block_vertices):
        return 5


def cage_topologies(threec_bb, twoc_bb):
    topologies = {
        "FourPlusSix": CGFourPlusSix((threec_bb, twoc_bb)),
        "FourPlusSix2": CGFourPlusSix2((threec_bb, twoc_bb)),
    }

    return topologies


def cage_topology_options():
    topologies = {
        "FourPlusSix": stk.cage.FourPlusSix,
        "FourPlusSix2": stk.cage.FourPlusSix2,
        # "SixPlusNine": stk.cage.SixPlusNine,
        # "EightPlusTwelve": stk.cage.EightPlusTwelve,
        # "TwentyPlusThirty": stk.cage.TwentyPlusThirty,
        # "TwoPlusThree": stk.cage.TwoPlusThree,
    }

    return topologies
