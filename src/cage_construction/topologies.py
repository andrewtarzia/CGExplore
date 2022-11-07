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
