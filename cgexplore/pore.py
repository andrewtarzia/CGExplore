#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for pore analysis.

Author: Andrew Tarzia

"""

import json
import os

import numpy as np
from scipy.spatial.distance import cdist


class PoreMeasure:
    """
    Uses PoreMapper [1]_ and pyWindow [2]_ to calculare pore properties.

    References
    ----------
    .. [1] https://andrewtarzia.github.io/posts/2021/11/poremapper-post/

    .. [2] https://pubs.acs.org/doi/10.1021/acs.jcim.8b00490

    """

    def calculate_min_distance(self, molecule):
        pair_dists = cdist(
            molecule.get_position_matrix(),
            molecule.get_centroid().reshape(1, 3),
        )
        min_distance = np.min(pair_dists.flatten())
        return {
            "min_distance": min_distance,
        }

    def calculate_pore(self, molecule, output_file):
        try:
            import pore_mapper as pm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "PoreMapper should be installed to use this."
            )
        xyz_file = output_file.replace(".json", "_h.xyz")
        por_file = output_file.replace(".json", "_p.xyz")
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

        pore.write_xyz_file(por_file)
        return pore_data

    def calculate_pw(self, molecule, output_file):
        try:
            import pywindow as pw
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "pyWindow should be installed to use this "
                "(pip install pywindowx)."
            )
        xyz_file = output_file.replace(".json", ".xyz")

        # Check if output file exists.
        if not os.path.exists(output_file):
            # Load cage into pywindow.
            molecule.write(xyz_file)
            pw_cage = pw.MolecularSystem.load_file(xyz_file)
            pw_cage_mol = pw_cage.system_to_molecule()
            os.system(f"rm {xyz_file}")

            # Calculate pore size.
            try:
                pw_cage_mol.calculate_pore_diameter_opt()
                pw_cage_mol.calculate_pore_volume_opt()
                pw_cage_mol.calculate_windows()
            except ValueError:
                # Handle failure.
                pw_cage_mol.properties["pore_volume_opt"] = 0
                pw_cage_mol.properties["pore_diameter_opt"] = {
                    "diameter": 0,
                    "atom_1": 0,
                    "centre_of_mass": [0, 0, 0],
                }
                pw_cage_mol.properties["windows"] = {
                    "diameters": [],
                    "centre_of_mass": [],
                }

            # Save files.
            pw_cage_mol.dump_properties_json(output_file)

        # Get data.
        with open(output_file, "r") as f:
            pw_data = json.load(f)

        return pw_data
