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

from env_set import shape_path


class ShapeMeasure:
    """
    Uses Shape [1]_ to calculate the shape of coordinates.

    References
    ----------
    .. [1] http://www.ee.ub.edu/

    """

    def __init__(
        self,
        output_dir,
        target_atmnums=None,
        shape_string=None,
    ):
        self._output_dir = output_dir
        self._target_atmnums = target_atmnums
        if shape_string is None:
            self._shape_dict = self.reference_shape_dict()
        else:
            self._shape_dict = {
                shape_string: self.reference_shape_dict()[shape_string]
            }
        self._num_vertex_options = tuple(
            set(
                int(self._shape_dict[i]["vertices"])
                for i in self._shape_dict
            )
        )

    def reference_shape_dict(self):
        return {
            "L-2": {
                "code": "1",
                "label": "L-2",
                "vertices": "2",
                "shape": "Linear D∞h",
            },
            "vT-2": {
                "code": "2",
                "label": "vT-2",
                "vertices": "2",
                "shape": "Divacant tetrahedron (V-shape, 109.47º) C2v",
            },
            "vOC-2": {
                "code": "3",
                "label": "vOC-2",
                "vertices": "2",
                "shape": "Tetravacant octahedron (L-shape, 90º) C2v",
            },
            "TP-3": {
                "vertices": "3",
                "code": "1",
                "label": "TP-3",
                "shape": "Trigonal planar D3h",
            },
            "vT-3": {
                "vertices": "3",
                "code": "2",
                "label": "vT-3",
                "shape": "Pyramid‡ (vacant tetrahedron) C3v",
            },
            "fac-vOC-3": {
                "vertices": "3",
                "code": "3",
                "label": "fac-vOC-3",
                "shape": "fac-Trivacant octahedron C3v",
            },
            "mer-vOC-3": {
                "vertices": "3",
                "code": "4",
                "label": "mer-vOC-3",
                "shape": "mer-Trivacant octahedron (T-shape) C2v",
            },
            "SP-4": {
                "code": "1",
                "label": "SP-4",
                "vertices": "4",
                "shape": "Square D4h",
            },
            "T-4": {
                "code": "2",
                "label": "T-4",
                "vertices": "4",
                "shape": "Tetrahedron Td",
            },
            "SS-4": {
                "code": "3",
                "label": "SS-4",
                "vertices": "4",
                "shape": "Seesaw or sawhorse‡ (cis-divacant octahedron) C2v",
            },
            "vTBPY-4": {
                "code": "4",
                "label": "vTBPY-4",
                "vertices": "4",
                "shape": "Axially vacant trigonal bipyramid C3v",
            },
            "PP-5": {
                "code": "1",
                "vertices": "5",
                "label": "PP-5",
                "shape": "Pentagon D5h",
            },
            "vOC-5": {
                "code": "2",
                "vertices": "5",
                "label": "vOC-5",
                "shape": "Vacant octahedron‡ (Johnson square pyramid, J1) C4v",
            },
            "TBPY-5": {
                "code": "3",
                "vertices": "5",
                "label": "TBPY-5",
                "shape": "Trigonal bipyramid D3h",
            },
            "SPY-5": {
                "code": "4",
                "vertices": "5",
                "label": "SPY-5",
                "shape": "Square pyramid § C4v",
            },
            "JTBPY-5": {
                "code": "5",
                "vertices": "5",
                "label": "JTBPY-5",
                "shape": "Johnson trigonal bipyramid (J12) D3h",
            },
            "HP-6": {
                "code": "1",
                "label": "HP-6",
                "vertices": "6",
                "shape": "Hexagon D6h",
            },
            "PPY-6": {
                "code": "2",
                "label": "PPY-6",
                "vertices": "6",
                "shape": "Pentagonal pyramid C5v",
            },
            "OC-6": {
                "code": "3",
                "label": "OC-6",
                "vertices": "6",
                "shape": "Octahedron Oh",
            },
            "TPR-6": {
                "code": "4",
                "label": "TPR-6",
                "vertices": "6",
                "shape": "Trigonal prism D3h",
            },
            "JPPY-5": {
                "code": "5",
                "label": "JPPY-5",
                "vertices": "6",
                "shape": "Johnson pentagonal pyramid (J2) C5v",
            },
            "HP-7": {
                "code": "1",
                "vertices": "7",
                "label": "HP-7",
                "shape": "Heptagon D7h",
            },
            "HPY-7": {
                "code": "2",
                "vertices": "7",
                "label": "HPY-7",
                "shape": "Hexagonal pyramid C6v",
            },
            "PBPY-7": {
                "code": "3",
                "vertices": "7",
                "label": "PBPY-7",
                "shape": "Pentagonal bipyramid D5h",
            },
            "COC-7": {
                "code": "4",
                "vertices": "7",
                "label": "COC-7",
                "shape": "Capped octahedron * C3v",
            },
            "CTPR-7": {
                "code": "5",
                "vertices": "7",
                "label": "CTPR-7",
                "shape": "Capped trigonal prism * C2v",
            },
            "JPBPY-7": {
                "code": "6",
                "vertices": "7",
                "label": "JPBPY-7",
                "shape": "Johnson pentagonal bipyramid (J13) D5h",
            },
            "JETPY-7": {
                "code": "7",
                "vertices": "7",
                "label": "JETPY-7",
                "shape": "Elongated triangular pyramid (J7) C3v",
            },
            "OP-8": {
                "code": "1",
                "label": "OP-8",
                "vertices": "8",
                "shape": "Octagon D8h",
            },
            "HPY-8": {
                "code": "2",
                "label": "HPY-8",
                "vertices": "8",
                "shape": "Heptagonal pyramid C7v",
            },
            "HBPY-8": {
                "code": "3",
                "label": "HBPY-8",
                "vertices": "8",
                "shape": "Hexagonal bipyramid D6h",
            },
            "CU-8": {
                "code": "4",
                "label": "CU-8",
                "vertices": "8",
                "shape": "Cube Oh",
            },
            "SAPR-8": {
                "code": "5",
                "label": "SAPR-8",
                "vertices": "8",
                "shape": "Square antiprism D4d",
            },
            "TDD-8": {
                "code": "6",
                "label": "TDD-8",
                "vertices": "8",
                "shape": "Triangular dodecahedron D2d",
            },
            "JGBF-8": {
                "code": "7",
                "label": "JGBF-8",
                "vertices": "8",
                "shape": "Johnson - Gyrobifastigium (J26) D2d",
            },
            "JETBPY-8": {
                "code": "8",
                "label": "JETBPY-8",
                "vertices": "8",
                "shape": "Johnson - Elongated triangular bipyramid (J14) D3h",
            },
            "JBTP-8": {
                "code": "9",
                "label": "JBTP-8",
                "vertices": "8",
                "shape": "Johnson - Biaugmented trigonal prism (J50) C2v",
            },
            "BTPR-8": {
                "code": "10",
                "label": "BTPR-8",
                "vertices": "8",
                "shape": "Biaugmented trigonal prism C2v",
            },
            "JSD-8": {
                "code": "11",
                "label": "JSD-8",
                "vertices": "8",
                "shape": "Snub disphenoid (J84) D2d",
            },
            "TT-8": {
                "code": "12",
                "label": "TT-8",
                "vertices": "8",
                "shape": "Triakis tetrahedron Td",
            },
            "ETBPY-8": {
                "code": "13",
                "label": "ETBPY-8",
                "vertices": "8",
                "shape": "Elongated trigonal bipyramid (see 8) D3h",
            },
            "EP-9": {
                "code": "1",
                "vertices": "9",
                "label": "EP-9",
                "shape": "Enneagon D9h",
            },
            "OPY-9": {
                "code": "2",
                "vertices": "9",
                "label": "OPY-9",
                "shape": "Octagonal pyramid C8v",
            },
            "HBPY-9": {
                "code": "3",
                "vertices": "9",
                "label": "HBPY-9",
                "shape": "Heptagonal bipyramid D7h",
            },
            "JTC-9": {
                "code": "4",
                "vertices": "9",
                "label": "JTC-9",
                "shape": (
                    "Triangular cupola (J3) = trivacant cuboctahedron "
                    "C3v"
                ),
            },
            "JCCU-9": {
                "code": "5",
                "vertices": "9",
                "label": "JCCU-9",
                "shape": (
                    "Capped cube (Elongated square pyramid, J8) C4v"
                ),
            },
            "CCU-9": {
                "code": "6",
                "vertices": "9",
                "label": "CCU-9",
                "shape": "Capped cube C4v",
            },
            "JCSAPR-9": {
                "code": "7",
                "vertices": "9",
                "label": "JCSAPR-9",
                "shape": (
                    "Capped sq. antiprism (Gyroelongated square "
                    "pyramid J10) C4v"
                ),
            },
            "CSAPR-9": {
                "code": "8",
                "vertices": "9",
                "label": "CSAPR-9",
                "shape": "Capped square antiprism C4v",
            },
            "JTCTPR-9": {
                "code": "9",
                "vertices": "9",
                "label": "JTCTPR-9",
                "shape": "Tricapped trigonal prism (J51) D3h",
            },
            "TCTPR-9": {
                "code": "10",
                "vertices": "9",
                "label": "TCTPR-9",
                "shape": "Tricapped trigonal prism D3h",
            },
            "JTDIC-9": {
                "code": "11",
                "vertices": "9",
                "label": "JTDIC-9",
                "shape": "Tridiminished icosahedron (J63) C3v",
            },
            "HH-9": {
                "code": "12",
                "vertices": "9",
                "label": "HH-9",
                "shape": "Hula-hoop C2v",
            },
            "MFF-9": {
                "code": "13",
                "vertices": "9",
                "label": "MFF-9",
                "shape": "Muffin Cs",
            },
            "DP-10": {
                "code": "1",
                "vertices": "10",
                "label": "DP-10",
                "shape": "Decagon D10h",
            },
            "EPY-10": {
                "code": "2",
                "vertices": "10",
                "label": "EPY-10",
                "shape": "Enneagonal pyramid C9v",
            },
            "OBPY-10": {
                "code": "3",
                "vertices": "10",
                "label": "OBPY-10",
                "shape": "Octagonal bipyramid D8h",
            },
            "PPR-10": {
                "code": "4",
                "vertices": "10",
                "label": "PPR-10",
                "shape": "Pentagonal prism D5h",
            },
            "PAPR-10": {
                "code": "5",
                "vertices": "10",
                "label": "PAPR-10",
                "shape": "Pentagonal antiprism D5d",
            },
            "JBCCU-10": {
                "code": "6",
                "vertices": "10",
                "label": "JBCCU-10",
                "shape": (
                    "Bicapped cube (Elongated square bipyramid J15) D4h"
                ),
            },
            "JBCSAPR-10": {
                "code": "7",
                "vertices": "10",
                "label": "JBCSAPR-10",
                "shape": (
                    "Bicapped square antiprism (Gyroelongated square "
                    "bipyramid J17) D4d"
                ),
            },
            "JMBIC-10": {
                "code": "8",
                "vertices": "10",
                "label": "JMBIC-10",
                "shape": "Metabidiminished icosahedron (J62) C2v",
            },
            "JATDI-10": {
                "code": "9",
                "vertices": "10",
                "label": "JATDI-10",
                "shape": "Augmented tridiminished icosahedron (J64) C3v",
            },
            "JSPC-10": {
                "code": "10",
                "vertices": "10",
                "label": "JSPC-10",
                "shape": "Sphenocorona (J87) C2v",
            },
            "SDD-10": {
                "code": "11",
                "vertices": "10",
                "label": "SDD-10",
                "shape": "Staggered dodecahedron (2:6:2) # D2",
            },
            "TD-10": {
                "code": "12",
                "vertices": "10",
                "label": "TD-10",
                "shape": "Tetradecahedron (2:6:2) C2v",
            },
            "HD-10": {
                "code": "13",
                "vertices": "10",
                "label": "HD-10",
                "shape": "Hexadecahedron (2:6:2, or 1:4:4:1) D4h",
            },
            "HP-11": {
                "code": "1",
                "vertices": "11",
                "label": "HP-11",
                "shape": "Hendecagon D11h",
            },
            "DPY-11": {
                "code": "2",
                "vertices": "11",
                "label": "DPY-11",
                "shape": "Decagonal pyramid C10v",
            },
            "EBPY-11": {
                "code": "3",
                "vertices": "11",
                "label": "EBPY-11",
                "shape": "Enneagonal bipyramid D9h",
            },
            "JCPPR-11": {
                "code": "4",
                "vertices": "11",
                "label": "JCPPR-11",
                "shape": (
                    "Capped pent. Prism (Elongated pentagonal pyramid "
                    "J9) C5v"
                ),
            },
            "JCPAPR-11": {
                "code": "5",
                "vertices": "11",
                "label": "JCPAPR-11",
                "shape": (
                    "Capped pent. antiprism (Gyroelongated pentagonal "
                    "pyramid J11) C5v"
                ),
            },
            "JAPPR-11": {
                "code": "6",
                "vertices": "11",
                "label": "JAPPR-11",
                "shape": "Augmented pentagonal prism (J52) C2v",
            },
            "JASPC-11": {
                "code": "7",
                "vertices": "11",
                "label": "JASPC-11",
                "shape": "Augmented sphenocorona (J87) Cs",
            },
            "DP-12": {
                "code": "1",
                "vertices": "12",
                "label": "DP-12",
                "shape": "Dodecagon D12h",
            },
            "HPY-12": {
                "code": "2",
                "vertices": "12",
                "label": "HPY-12",
                "shape": "Hendecagonal pyramid C11v",
            },
            "DBPY-12": {
                "code": "3",
                "vertices": "12",
                "label": "DBPY-12",
                "shape": "Decagonal bipyramid D10h",
            },
            "HPR-12": {
                "code": "4",
                "vertices": "12",
                "label": "HPR-12",
                "shape": "Hexagonal prism D6h",
            },
            "HAPR-12": {
                "code": "5",
                "vertices": "12",
                "label": "HAPR-12",
                "shape": "Hexagonal antiprism D6d",
            },
            "TT-12": {
                "code": "6",
                "vertices": "12",
                "label": "TT-12",
                "shape": "Truncated tetrahedron Td",
            },
            "COC-12": {
                "code": "7",
                "vertices": "12",
                "label": "COC-12",
                "shape": "Cuboctahedron Oh",
            },
            "ACOC-12": {
                "code": "8",
                "vertices": "12",
                "label": "ACOC-12",
                "shape": (
                    "Anticuboctahedron (Triangular orthobicupola J27) "
                    "D3h"
                ),
            },
            "IC-12": {
                "code": "9",
                "vertices": "12",
                "label": "IC-12",
                "shape": "Icosahedron Ih",
            },
            "JSC-12": {
                "code": "10",
                "vertices": "12",
                "label": "JSC-12",
                "shape": "Square cupola (J4) C4v",
            },
            "JEPBPY-12": {
                "code": "11",
                "vertices": "12",
                "label": "JEPBPY-12",
                "shape": "Elongated pentagonal bipyramid (J16) D6h",
            },
            "JBAPPR-12": {
                "code": "12",
                "vertices": "12",
                "label": "JBAPPR-12",
                "shape": "Biaugmented pentagonal prism (J53) C2v",
            },
            "JSPMC-12": {
                "code": "13",
                "vertices": "12",
                "label": "JSPMC-12",
                "shape": "Sphenomegacorona (J88) Cs",
            },
            "DD-20": {
                "code": "1",
                "vertices": "20",
                "label": "DD-20",
                "shape": "Dodecahedron † Ih",
            },
            "TCU-24": {
                "code": "1",
                "vertices": "24",
                "label": "TCU-24",
                "shape": "Truncated cube Oh",
            },
            "TOC-24": {
                "code": "2",
                "vertices": "24",
                "label": "TOC-24",
                "shape": "Truncated octahedron Oh",
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
    ):
        """
        Write input file for shape.

        """
        num_vertices = len(structure_string.split("\n")) - 2

        possible_shapes = self._get_possible_shapes(num_vertices)
        shape_numbers = tuple(i["code"] for i in possible_shapes)

        title = "$shape run by Andrew Tarzia - central atom=0 always.\n"
        if num_vertices == 12:
            fix_perm = r"%fixperm 0\n"
        else:
            fix_perm = "\n"
        size_of_poly = f"{num_vertices} 0\n"
        codes = " ".join(shape_numbers) + "\n"

        string = (
            title + fix_perm + size_of_poly + codes + structure_string
        )

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
        )

        cmd = f"{shape_path()} {input_file}"
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
            if (
                ai.get_atom().get_atomic_number()
                in self._target_atmnums
            ):
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

    def _get_possible_shapes(self, num_vertices):
        ref_dict = self.reference_shape_dict()
        return tuple(
            ref_dict[i]
            for i in ref_dict
            if int(ref_dict[i]["vertices"]) == num_vertices
        )

    def calculate(self, molecule):
        output_dir = os.path.abspath(self._output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        init_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            structure_string = "shape run by AT\n"
            num_centroids = 0
            pos_mat = molecule.get_position_matrix()
            for a in molecule.get_atoms():
                c = pos_mat[a.get_id()]
                structure_string += (
                    f"{a.__class__.__name__} {c[0]} {c[1]} {c[2]}\n"
                )
                num_centroids += 1

            if num_centroids not in self._num_vertex_options:
                raise ValueError(
                    f"you gave {num_centroids} vertices, but expected "
                    "to calculate shapes with "
                    f"{self._num_vertex_options} options"
                )

            shapes = self._run_calculation(structure_string)

        finally:
            os.chdir(init_dir)

        return shapes

    def calculate_from_centroids(self, constructed_molecule):
        if self._target_atmnums is None:
            raise ValueError(
                "you need to set the atom numbers to use to "
                "calculate centroids."
            )
        output_dir = os.path.abspath(self._output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        init_dir = os.getcwd()
        try:
            os.chdir(output_dir)
            centroids = self._get_centroids(constructed_molecule)
            structure_string = "shape run by AT\n"
            num_centroids = 0
            for c in centroids:
                structure_string += f"Zn {c[0]} {c[1]} {c[2]}\n"
                num_centroids += 1

            if num_centroids not in self._num_vertex_options:
                raise ValueError(
                    f"you gave {num_centroids} vertices, but expected "
                    "to calculate shapes with "
                    f"{self._num_vertex_options} options"
                )

            shapes = self._run_calculation(structure_string)

        finally:
            os.chdir(init_dir)

        return shapes
