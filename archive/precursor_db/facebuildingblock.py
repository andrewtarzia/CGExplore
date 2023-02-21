#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module defining FaceBuildingBlock class.

Author: Andrew Tarzia

Date Created: 22 Mar 2021

"""

from itertools import combinations

import stk

from utilities import get_atom_distance


class FaceBuildingBlock(stk.BuildingBlock):
    def get_bromines(self):

        bromine_atom_ids = [
            tuple(fg.get_deleter_ids())[0]
            for fg in self.get_functional_groups()
        ]

        return bromine_atom_ids

    def get_long_axis(self):

        atom_ids = self.get_bromines()
        if len(atom_ids) != 4:
            raise ValueError(f"{self} has too many deleter bromines.")

        atom_dists = sorted(
            [
                (idx1, idx2, get_atom_distance(self, idx1, idx2))
                for idx1, idx2 in combinations(atom_ids, r=2)
            ],
            key=lambda a: a[2],
        )

        # Can assume the ordering of the binder atom distances:
        # 0, 1: short vectors
        # 2, 3: long vectors
        # 4, 5: diagonal vectors
        # This fails when the molecule is not sufficiently anisotropic,
        # at which point it will not matter.
        short_vector_fg_1 = (atom_dists[0][0], atom_dists[0][1])
        short_vector_fg_2 = (
            (atom_dists[1][0], atom_dists[1][1])
            if (
                atom_dists[1][0] not in short_vector_fg_1
                and atom_dists[1][1] not in short_vector_fg_1
            )
            else (atom_dists[2][0], atom_dists[2][1])
        )

        short_pair_1_centroid = self.get_centroid(
            atom_ids=(i for i in short_vector_fg_1),
        )
        short_pair_2_centroid = self.get_centroid(
            atom_ids=(i for i in short_vector_fg_2),
        )

        long_axis = short_pair_2_centroid - short_pair_1_centroid

        return long_axis