#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module containing available precursors.

Author: Andrew Tarzia

"""

import os
import math
import stk

from .facebuildingblock import FaceBuildingBlock
from .utilities import reorient_linker
from .topologies import (
    ThreeC0Arm,
    TwoC0Arm,
    TwoC1Arm,
    TwoC2Arm,
    TwoC3Arm,
)


def precursor_dir():
    return os.path.dirname(os.path.realpath(__file__))


def delta_bb():
    return stk.BuildingBlock.init_from_file(
        path=os.path.join(precursor_dir(), "corner_delta.mol"),
        functional_groups=(stk.BromoFactory(),),
    )


def lambda_bb():
    return stk.BuildingBlock.init_from_file(
        path=os.path.join(precursor_dir(), "corner_lambda.mol"),
        functional_groups=(stk.BromoFactory(),),
    )


def threec_bb():

    top_vertex = (0, math.sqrt(3) / 4)
    lef_vertex = (-1 / 2, -math.sqrt(3) / 4)
    rig_vertex = (1 / 2, -math.sqrt(3) / 4)
    bb = stk.BuildingBlock(
        smiles="[S]12([Fe]3[C]1([Br])[P]23[Br])[Br]",
        functional_groups=(stk.BromoFactory(),),
        position_matrix=[
            [3 * top_vertex[0], 3 * top_vertex[1], 0],
            [0, 0, 0],
            [3 * lef_vertex[0], 3 * lef_vertex[1], 0],
            [6 * lef_vertex[0], 6 * lef_vertex[1], 0],
            [3 * rig_vertex[0], 3 * rig_vertex[1], 0],
            [6 * rig_vertex[0], 6 * rig_vertex[1], 0],
            [6 * top_vertex[0], 6 * top_vertex[1], 0],
            # [3 * top_vertex[0], 3 * top_vertex[1], 0],
            # [4 * top_vertex[0], 4 * top_vertex[1], 0],
            # [3 * lef_vertex[0], 3 * lef_vertex[1], 0],
            # [4 * lef_vertex[0], 4 * lef_vertex[1], 0],
            # [3 * rig_vertex[0], 3 * rig_vertex[1], 0],
            # [4 * rig_vertex[0], 4 * rig_vertex[1], 0],
        ],
    )
    bb.write(os.path.join(precursor_dir(), "threec_bb.mol"))
    return bb


def plane_bb():

    bb = FaceBuildingBlock.init_from_file(
        path=os.path.join(precursor_dir(), "plane.mol"),
        functional_groups=(stk.BromoFactory(),),
    )

    temp_plane = reorient_linker(bb)

    # Set functional group ordering based on long axis.
    fg_centroids = tuple(
        temp_plane.get_centroid(
            atom_ids=fg.get_placer_ids(),
        )
        for fg in temp_plane.get_functional_groups()
    )
    plus_minus_fg_id = tuple(
        i
        for i, cent in enumerate(fg_centroids)
        if cent[0] > 0 and cent[1] < 0
    )[0]
    fg1_id = plus_minus_fg_id
    fg2_id, fg3_id, fg4_id = tuple(
        i
        for i in range(temp_plane.get_num_functional_groups())
        if i != fg1_id
    )
    new_fgs = tuple(temp_plane.get_functional_groups())
    bb = temp_plane.with_functional_groups(
        functional_groups=(
            new_fgs[fg1_id],
            new_fgs[fg2_id],
            new_fgs[fg3_id],
            new_fgs[fg4_id],
        )
    )

    return bb


def three_precursor_topology_options():
    topologies = {
        "3c-0": ThreeC0Arm,
    }

    return topologies


def two_precursor_topology_options():
    topologies = {
        "2c-0": TwoC0Arm,
        "2c-1": TwoC1Arm,
        "2c-2": TwoC2Arm,
        "2c-3": TwoC3Arm,
    }

    return topologies
