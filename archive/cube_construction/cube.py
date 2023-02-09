#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG cube.

Author: Andrew Tarzia

"""

import stk


class CGM8L6Cube(stk.cage.M8L6Cube):
    def _get_scale(self, building_block_vertices):
        return 10
