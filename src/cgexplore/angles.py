#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling angles.

Author: Andrew Tarzia

"""

import itertools
import logging
from dataclasses import dataclass
from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class TargetAngle:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity


@dataclass
class TargetAngleRange:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
        for angle, k in itertools.product(self.angles, self.angle_ks):
            yield TargetAngle(
                class1=self.class1,
                class2=self.class2,
                class3=self.class3,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                eclass3=self.eclass3,
                angle=angle,
                angle_k=k,
            )