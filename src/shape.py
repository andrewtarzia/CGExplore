#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for shape analysis.

Author: Andrew Tarzia

"""

import subprocess as sp
import os
import shutil

import env_set


class ShapeMeasure:
    """
    Uses Shape [1]_ to calculate the shape of coordinates.

    References
    ----------
    .. [1] http://www.ee.ub.edu/

    """

    def __init__(self, output_dir, target_atmnum, shape_string):
        self._output_dir = output_dir
        self._target_atmnum = target_atmnum
        self._shape_dict = self._ref_shape_dict()[shape_string]

    def _ref_shape_dict(self):
        return {
            "cube": {
                "vertices": "8",
                "label": "CU-8",
                "code": "4",
            },
            "octagon": {
                "vertices": "8",
                "label": "OP-8",
                "code": "1",
            },
        }

    def _collect_all_shape_values(self, output_file):
        """
        Collect shape values from output.

        """

        with open(output_file, "r") as f:
            lines = f.readlines()

        label_idx_map = {}
        for line in reversed(lines):
            if "Structure" in line:
                line = [
                    i.strip()
                    for i in line.rstrip().split("]")[1].split(" ")
                    if i.strip()
                ]
                for idx, symb in enumerate(line):
                    label_idx_map[symb] = idx
                break
            line = [i.strip() for i in line.rstrip().split(",")]
            values = line

        shapes = {
            i: float(values[1 + label_idx_map[i]])
            for i in label_idx_map
        }

        return shapes

    def _write_input_file(
        self,
        input_file,
        structure_string,
        num_vertices,
        central_atom_id,
        ref_shapes,
    ):
        """
        Write input file for shape.

        """

        title = "$shape run by Andrew Tarzia.\n"
        size_of_poly = f"{num_vertices} {central_atom_id}\n"
        codes = " ".join(ref_shapes) + "\n"

        string = title + size_of_poly + codes + structure_string

        with open(input_file, "w") as f:
            f.write(string)

    def _run_calculation(self, structure_string):
        """
        Calculate the shape of a molecule.

        """

        input_file = "shp.dat"
        std_out = "shp.out"
        output_file = "shp.tab"

        self._write_input_file(
            input_file=input_file,
            structure_string=structure_string,
            num_vertices=self._shape_dict["vertices"],
            central_atom_id=0,
            ref_shapes=self._shape_dict["code"],
        )

        cmd = f"{env_set.shape_path()} {input_file}"
        with open(std_out, "w") as f:
            # Note that sp.call will hold the program until completion
            # of the calculation.
            sp.call(
                cmd,
                stdin=sp.PIPE,
                stdout=f,
                stderr=sp.PIPE,
                # Shell is required to run complex arguments.
                shell=True,
            )

        shapes = self._collect_all_shape_values(output_file)
        return shapes

    def _get_centroids(self, molecule):
        bb_ids = {}
        for ai in molecule.get_atom_infos():
            aibbid = ai.get_building_block_id()
            if ai.get_atom().get_atomic_number() == self._target_atmnum:
                if aibbid not in bb_ids:
                    bb_ids[aibbid] = []
                bb_ids[aibbid].append(ai.get_atom().get_id())

        centroids = []
        for n in bb_ids:
            centroids.append(molecule.get_centroid(atom_ids=bb_ids[n]))

        with open("cents.xyz", "w") as f:
            f.write(f"{len(centroids)}\n\n")
            for c in centroids:
                f.write(f"Zn {c[0]} {c[1]} {c[2]}\n")

        return centroids

    def calculate(self, molecule):
        output_dir = os.path.abspath(self._output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        init_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            centroids = self._get_centroids(molecule)
            structure_string = "shape run by AT\n"
            for c in centroids:
                structure_string += f"Zn {c[0]} {c[1]} {c[2]}\n"

            shapes = self._run_calculation(structure_string)
            shape_measure = shapes[self._shape_dict["label"]]

        finally:
            os.chdir(init_dir)

        return shape_measure
