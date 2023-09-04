#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling nobonded interactions.

Author: Andrew Tarzia

"""

import itertools
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class TargetNonbonded:
    search_string: str
    search_estring: str
    sigma: float
    epsilon: float


@dataclass
class TargetNonbondedRange:
    search_string: str
    search_estring: str
    sigmas: tuple[float]
    epsilons: tuple[float]

    def yield_nonbondeds(self):
        for sigma, epsilon in itertools.product(self.sigmas, self.epsilons):
            yield TargetNonbonded(
                search_string=self.search_string,
                search_estring=self.search_estring,
                epsilon=epsilon,
                sigma=sigma,
            )
