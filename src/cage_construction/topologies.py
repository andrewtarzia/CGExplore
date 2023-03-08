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
            "2P3": CGTwoPlusThree,
            "4P6": CGFourPlusSix,
            "4P62": CGFourPlusSix2,
            "6P9": CGSixPlusNine,
            "8P12": CGEightPlusTwelve,
        }
    if fg_set == "2p4":
        topologies = {
            "2P4": CGM2L4Lantern,
            "3P6": CGM3L6,
            "4P8": CGM6L12Cube,
            "6P12": CGM6L12Cube,
            "8P16": stk.cage.EightPlusSixteen,
            "12P24": CGM12L24,
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


class CGTwoPlusThree(stk.cage.TwoPlusThree):
    def _get_scale(self, building_block_vertices):
        return 5


class CGFourPlusSix(stk.cage.FourPlusSix):
    def _get_scale(self, building_block_vertices):
        return 10


class CGFourPlusSix2(stk.cage.FourPlusSix2):
    def _get_scale(self, building_block_vertices):
        return 10


class CGSixPlusNine(stk.cage.SixPlusNine):
    def _get_scale(self, building_block_vertices):
        return 15


class CGEightPlusTwelve(stk.cage.EightPlusTwelve):
    def _get_scale(self, building_block_vertices):
        return 15


class CGM2L4Lantern(stk.cage.M2L4Lantern):
    def _get_scale(self, building_block_vertices):
        return 5


class CGM3L6(stk.cage.M3L6):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM4L8(stk.cage.M4L8):
    def _get_scale(self, building_block_vertices):
        return 10


class CGM6L12Cube(stk.cage.M6L12Cube):
    def _get_scale(self, building_block_vertices):
        return 15


class CGM12L24(stk.cage.M12L24):
    def _get_scale(self, building_block_vertices):
        return 15

    def get_vertex_alignments(self):
        return self._vertex_alignments
