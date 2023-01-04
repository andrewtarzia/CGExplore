#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for pymol visualisation.

Author: Andrew Tarzia

"""

import os

from env_set import pymol_path


class Pymol:
    def __init__(self, output_dir, file_prefix):
        self._output_dir = output_dir
        self._file_prefix = file_prefix
        self._pymol = pymol_path()

    def _write_pymol_script(
        self,
        structure_files,
        structure_colours,
        pml_file,
    ):
        lstring = ""
        cstring = ""
        lnames = []
        for sf, col in zip(structure_files, structure_colours):
            lstring += f"load {sf}\n"
            lname = str(sf.name).replace(".mol", "")
            lnames.append(lname)
            col = col.replace("#", "0x")
            cstring += f"color {col} (name {lname})\n"

        string = (
            f"{lstring}\n"
            f"{cstring}\n"
            "set grid_mode, 1\n"
            "as sticks\n"
            "set stick_radius, 0.7\n"
            "show spheres\n"
            "alter all,vdw=0.8\n"
            "rebuild\n"
            "orient\n"
            "zoom center, 25\n"
            "bg_color white\n"
            # "set ray_trace_mode, 1\n"
            "ray 2400, 2400\n"
            f"png {self._output_dir / self._file_prefix}.png\n"
            # "quit\n"
        )

        with open(pml_file, "w") as f:
            f.write(string)

    def visualise(self, structure_files, structure_colours):
        pml_file = self._output_dir / f"{self._file_prefix}.pml"
        self._write_pymol_script(
            structure_files=structure_files,
            structure_colours=structure_colours,
            pml_file=pml_file,
        )
        os.system(f"{self._pymol} {pml_file}")
        return None
