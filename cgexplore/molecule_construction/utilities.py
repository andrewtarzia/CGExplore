#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Utilities module for precursors.

Author: Andrew Tarzia

"""

import numpy as np

try:
    from stk._internal.utilities.utilities import (
        get_acute_vector,
        get_plane_normal,
    )
except ImportError:
    from stk import (
        get_acute_vector,
        get_plane_normal,
    )


def reorient_linker(molecule):
    target_coords = (
        np.array([1, 1, 0]),
        np.array([1, -1, 0]),
        np.array([-1, -1, 0]),
        np.array([-1, 1, 0]),
    )
    centroid_pos = np.array([0, 0, 0])
    molecule = molecule.with_centroid(
        position=centroid_pos,
        atom_ids=molecule.get_placer_ids(),
    )

    edge_centroid = sum(target_coords) / len(target_coords)
    edge_normal = get_acute_vector(
        reference=edge_centroid,
        vector=get_plane_normal(
            points=np.array(target_coords),
        ),
    )

    fg_bonder_centroid = molecule.get_centroid(
        atom_ids=next(molecule.get_functional_groups()).get_placer_ids(),
    )
    edge_position = target_coords[0]
    molecule = molecule.with_rotation_to_minimize_angle(
        start=fg_bonder_centroid - centroid_pos,
        target=edge_position - edge_centroid,
        axis=edge_normal,
        origin=centroid_pos,
    )

    # Flatten wrt to xy plane.
    core_centroid = molecule.get_centroid(
        atom_ids=molecule.get_core_atom_ids(),
    )
    normal = molecule.get_plane_normal(
        atom_ids=molecule.get_placer_ids(),
    )
    normal = get_acute_vector(
        reference=core_centroid - centroid_pos,
        vector=normal,
    )
    molecule = molecule.with_rotation_between_vectors(
        start=normal,
        target=[0, 0, 1],
        origin=centroid_pos,
    )

    # Align long axis of molecule (defined by deleter atoms) with
    # y axis.
    long_axis_vector = molecule.get_long_axis()
    edge_centroid = sum(target_coords) / len(target_coords)
    edge_normal = get_acute_vector(
        reference=edge_centroid,
        vector=get_plane_normal(
            points=np.array(target_coords),
        ),
    )
    molecule = molecule.with_rotation_to_minimize_angle(
        start=long_axis_vector,
        target=[1, 0, 0],
        axis=edge_normal,
        origin=centroid_pos,
    )
    return molecule
