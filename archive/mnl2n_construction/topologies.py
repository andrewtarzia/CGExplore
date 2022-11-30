#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of cages.

Author: Andrew Tarzia

"""

import stk


class CGM2L4Lantern(stk.cage.M2L4Lantern):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM3L6(stk.cage.M3L6):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM4L8(stk.cage.M4L8):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM6L12Cube(stk.cage.M6L12Cube):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM12L24(stk.cage.M12L24):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM24L48(stk.cage.M24L48):
    def _get_scale(self, building_block_vertices):
        return 10


def cage_topologies(fourc_bb, twoc_bb):
    topologies = {
        "m2l4": CGM2L4Lantern(
            building_blocks=(fourc_bb, twoc_bb),
            optimizer=stk.Collapser(distance_threshold=3),
        ),
        "m3l6": CGM3L6(
            building_blocks=(fourc_bb, twoc_bb),
            optimizer=stk.Collapser(distance_threshold=3),
        ),
        "m4l8": CGM4L8(
            building_blocks=(fourc_bb, twoc_bb),
            optimizer=stk.Collapser(distance_threshold=3),
        ),
        "m6l12": CGM6L12Cube(
            building_blocks=(fourc_bb, twoc_bb),
            optimizer=stk.Collapser(distance_threshold=3),
        ),
        "m12l24": CGM12L24(
            building_blocks=(fourc_bb, twoc_bb),
        ),
        "m24l48": CGM24L48(
            building_blocks=(fourc_bb, twoc_bb),
        ),
    }

    return topologies
