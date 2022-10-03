#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate CG models of MnL2n systems.

Author: Andrew Tarzia

"""

import sys
import stk
import numpy as np
import os
import json
import logging

from env_set import (
    mnl2n_figures,
    mnl2n_structures,
)
from utilities import (
    get_distances,
    get_angles,
)
from gulp_optimizer import CGGulpOptimizer


class MNL2NOptimizer(CGGulpOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        biteangle,
    ):
        self._biteangle = float(biteangle)
        super().__init__(fileprefix, output_dir)

    def define_bond_potentials(self):
        bond_ks_ = {
            ("N", "Pd"): 10,
            ("C", "N"): 10,
            ("B", "C"): 10,
        }
        bond_rs_ = {
            ("N", "Pd"): 2,
            ("C", "N"): 2,
            ("B", "C"): 3,
        }
        return bond_ks_, bond_rs_

    def define_angle_potentials(self):
        angle_ks_ = {
            ("B", "C", "C"): 20,
            ("N", "N", "Pd"): 20,
            ("B", "C", "N"): 20,
            ("C", "N", "Pd"): 20,
        }
        angle_thetas_ = {
            ("B", "C", "C"): 180,
            ("N", "N", "Pd"): (
                "check",
                {"cut": 130, "min": 90, "max": 180},
            ),
            ("B", "C", "N"): (self._biteangle / 2) + 90,
            ("C", "N", "Pd"): 180,
        }

        return angle_ks_, angle_thetas_


