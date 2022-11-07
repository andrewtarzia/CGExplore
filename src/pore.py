#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for pore analysis.

Author: Andrew Tarzia

"""

import os
import json
import pore_mapper as pm


class PoreMeasure:
    """
    Uses PoreMapper [1]_ to calculare pore properties.

    References
    ----------
    .. [1] https://andrewtarzia.github.io/posts/2021/11/poremapper-post/

    """

    def calculate_pore(self, molecule, output_file):
        xyz_file = output_file.replace(".json", ".xyz")
        molecule.write(xyz_file)

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                pore_data = json.load(f)
            return pore_data

        host = pm.Host.init_from_xyz_file(xyz_file)
        host = host.with_centroid([0.0, 0.0, 0.0])

        # Define calculator object.
        calculator = pm.Inflater(bead_sigma=1.2)
        # Run calculator on host object, analysing output.
        final_result = calculator.get_inflated_blob(host=host)
        pore = final_result.pore
        blob = final_result.pore.get_blob()
        windows = pore.get_windows()
        pore_data = {
            "step": final_result.step,
            "num_movable_beads": final_result.num_movable_beads,
            "windows": windows,
            "blob_max_diam": blob.get_maximum_diameter(),
            "pore_max_rad": pore.get_maximum_distance_to_com(),
            "pore_mean_rad": pore.get_mean_distance_to_com(),
            "pore_volume": pore.get_volume(),
            "asphericity": pore.get_asphericity(),
            "shape": pore.get_relative_shape_anisotropy(),
        }
        with open(output_file, "w") as f:
            json.dump(pore_data, f)
        return pore_data
