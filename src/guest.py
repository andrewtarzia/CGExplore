#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate CG models of MnL2n systems.

Author: Andrew Tarzia

"""

import sys
import stk
import molellipsize as mes
from scipy.spatial import ConvexHull

import logging

from env_set import guest_structures


class PointCloud:
    def __init__(self, points):
        self._points = points
        self._convexhull = ConvexHull(points, incremental=True)

    def get_points(self):
        return self._points

    def write_points_to_xyz(self, filename, atomtype):
        with open(filename, "w") as f:
            f.write(f"{len(self._points)}\n\n")
            for g in self._points:
                x, y, z = g
                f.write(f"{atomtype} {x} {y} {z}\n")

    def get_convexhull(self):
        return self._convexhull

    def write_convexhull_to_xyz(self, filename, atomtype):
        with open(filename, "w") as f:
            f.write(f"{len(self._convexhull.vertices)}\n\n")
            for gs in self._convexhull.vertices:
                x, y, z = self._points[gs]
                f.write(f"{atomtype} {x} {y} {z}\n")


class Guest:
    def __init__(self, stk_molecule):
        self._stk_molecule = stk_molecule
        self._mes_molecule = mes.Molecule(
            rdkitmol=self._stk_molecule.to_rdkit_mol(),
            conformers=[0],
        )
        self._pointcloud = PointCloud(self._get_hitpoints())

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return str(self)

    def get_stk_molecule(self):
        return self._stk_molecule

    def get_mes_molecule(self):
        return self._mes_molecule

    def _get_hitpoints(self):
        box, sideLen, shape = self._mes_molecule.get_molecule_shape(
            conformer=self._stk_molecule.to_rdkit_mol().GetConformer(0),
            cid=0,
            vdwscale=0.9,
            boxmargin=4.0,
            spacing=0.5,
        )

        hit_points = self._mes_molecule.get_hitpoints(shape)
        return hit_points

    def write_hitpoints(self, filename, atomtype):
        self._pointcloud.write_points_to_xyz(filename, atomtype)

    def write_convexhul(self, filename, atomtype):
        self._pointcloud.write_convexhull_to_xyz(filename, atomtype)


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = guest_structures()
    manual_structures = struct_output / "manual_builds"

    guest_bbs = {
        "1": stk.BuildingBlock("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        "2": stk.BuildingBlock("c1ccccc1"),
        "3": stk.BuildingBlock.init_from_file(
            str(manual_structures / "C60.mol")
        ),
        "4": stk.BuildingBlock.init_from_file(
            str(manual_structures / "C70.mol")
        ),
        "5": stk.BuildingBlock.init_from_file(
            str(manual_structures / "C20.mol")
        ),
        "6": stk.BuildingBlock.init_from_file(
            str(manual_structures / "PCBM.mol")
        ),
    }

    for name in guest_bbs:
        molecule_output = str(struct_output / f"{name}.mol")
        hitpoint_output = str(struct_output / f"{name}_hp.xyz")
        convexhu_output = str(struct_output / f"{name}_ch.xyz")

        bb = guest_bbs[name]
        bb.write(molecule_output)
        guest = Guest(bb)
        guest.write_hitpoints(filename=hitpoint_output, atomtype="C")
        guest.write_convexhul(filename=convexhu_output, atomtype="H")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
