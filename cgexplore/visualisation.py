#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for pymol visualisation.

Author: Andrew Tarzia

"""

import os
import stk


class Pymol:
    def __init__(
        self,
        output_dir,
        file_prefix,
        pymol_path,
        settings=None,
    ):
        self._output_dir = output_dir
        self._file_prefix = file_prefix
        self._pymol = pymol_path
        if settings is None:
            self._settings = self._default_settings()
        else:
            for setting in self._default_settings():
                sett = self._default_settings()[setting]
                if setting not in settings:
                    settings[setting] = sett
            self._settings = settings

    def _default_settings(self):
        return {
            "grid_mode": 1,
            "stick_rad": 0.7,
            "vdw": 0.8,
            "rayx": 2400,
            "rayy": 2400,
            "zoom_string": "zoom",
        }

    def _get_zoom_string(self, structure_files):

        max_max_diam = 0
        for fi in structure_files:
            max_diam = stk.BuildingBlock.init_from_file(
                path=str(fi),
            ).get_maximum_diameter()
            print(max_diam)
            max_max_diam = max((max_diam, max_max_diam))
        return f"zoom center, {max_max_diam/2}"

    def _write_pymol_script(
        self,
        structure_files,
        structure_colours,
        pml_file,
        orient_atoms,
        big_colour,
    ):

        if self._settings["zoom_string"] == "custom":
            zoom_string = self._get_zoom_string(structure_files)
        else:
            zoom_string = self._settings["zoom_string"]

        if structure_colours is None:
            show_colours = ["#000000" for i in structure_files]
        else:
            show_colours = structure_colours

        if orient_atoms is None:
            orient_string = "orient"
        else:
            orient_string = f"orient (name {orient_atoms}*)"

        if big_colour is None:
            big_colour_string = ""
        else:
            big_colour_string = f"orient (name {orient_atoms}*)"

        lstring = ""
        cstring = ""
        lnames = []
        for sf, col in zip(structure_files, show_colours):
            lstring += f"load {sf}\n"
            lname = str(sf.name).replace(".mol", "")
            lnames.append(lname)
            col = col.replace("#", "0x")
            if structure_colours is None:
                cstring += "color orange, (name C*)\n"
            else:
                cstring += f"color {col}, {lname}\n"

        string = (
            f"{lstring}\n"
            f"{cstring}\n"
            f"{big_colour_string}\n"
            f"set grid_mode, {self._settings['grid_mode']}\n"
            "as sticks\n"
            f"set stick_radius, {self._settings['stick_rad']}\n"
            "show spheres\n"
            f"alter all,vdw={self._settings['vdw']}\n"
            "rebuild\n"
            f"{orient_string}\n"
            # "zoom center, 25\n"
            f"{zoom_string}\n"
            "bg_color white\n"
            # "set ray_trace_mode, 1\n"
            f"ray {self._settings['rayx']}, {self._settings['rayy']}\n"
            f"png {self._output_dir / self._file_prefix}.png\n"
            "quit\n"
        )

        with open(pml_file, "w") as f:
            f.write(string)

    def visualise(
        self,
        structure_files,
        structure_colours=None,
        orient_atoms=None,
        big_colour=None,
    ):
        pml_file = self._output_dir / f"{self._file_prefix}.pml"
        self._write_pymol_script(
            structure_files=structure_files,
            structure_colours=structure_colours,
            pml_file=pml_file,
            orient_atoms=orient_atoms,
            big_colour=big_colour,
        )
        os.system(f"{self._pymol} {pml_file}")
        return None