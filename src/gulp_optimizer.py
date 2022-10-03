#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG Gulp optimizer.

Author: Andrew Tarzia

"""

import os
import numpy as np
import re
from string import digits

from env_set import gulp_path
from utilities import get_all_angles, angle_between


class CGGulpOptimizer:
    def __init__(self, fileprefix, output_dir):
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._gulp_in = os.path.join(
            self._output_dir, f"{self._fileprefix}.gin"
        )
        self._gulp_out = os.path.join(
            self._output_dir, f"{self._fileprefix}.ginout"
        )
        self._output_xyz = os.path.join(
            self._output_dir, f"{self._fileprefix}_final.xyz"
        )
        self._mass = 1
        self._bond_cutoff = 30
        self._angle_cutoff = 30

    def _run_gulp(self):
        os.system(f"{gulp_path()} < {self._gulp_in} > {self._gulp_out}")

    def _extract_gulp(self):
        with open(self._gulp_out, "r") as f:
            lines = f.readlines()

        nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
        run_data = {"traj": {}}
        for line in lines:
            if "Cycle" in line:
                splits = line.rstrip().split()
                run_data["traj"][int(splits[1])] = {
                    "energy": float(splits[3]),
                    "gnorm": float(splits[5]),
                }

            if "Final energy" in line:
                string = nums.search(line.rstrip()).group(0)
                energy = float(string)
                run_data["final_energy"] = energy

            if "Final Gnorm" in line:
                string = nums.search(line.rstrip()).group(0)
                gnorm = float(string)
                run_data["final_gnorm"] = gnorm

        return run_data

    def define_bond_potentials(self):
        raise NotImplementedError()

    def define_angle_potentials(self):
        raise NotImplementedError()

    def _get_coord_mass_string(self, mol):
        coord_string = "cartesian\n"
        mass_string = ""

        pos_mat = mol.get_position_matrix()
        atoms = list(mol.get_atoms())
        for atom, pos_ in zip(atoms, pos_mat):
            name = f"{atom.__class__.__name__}{atom.get_id()+1}"
            coord_string += (
                f"{name} {round(pos_[0], 2)} {round(pos_[1], 2)} "
                f"{round(pos_[2], 2)}\n"
            )
            mass_string += f"mass {name} {self._mass}\n"

        return coord_string, mass_string

    def _get_bond_string(self, mol):
        bond_ks_, bond_rs_ = self.define_bond_potentials()
        bond_string = "harm\n"
        bonds = list(mol.get_bonds())

        for bond in bonds:
            atom1 = bond.get_atom1()
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            atom2 = bond.get_atom2()
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            table = str.maketrans("", "", digits)
            sorted_name = tuple(
                sorted([i.translate(table) for i in (name1, name2)])
            )

            try:
                bond_k = bond_ks_[sorted_name]
                bond_r = bond_rs_[sorted_name]
            except KeyError:
                continue

            bond_string += (
                f"{name1} {name2}  {bond_k} {bond_r} "
                f"{self._bond_cutoff}\n"
            )
        return bond_string

    def _get_angle_string(self, mol):
        angle_string = "three\n"
        angle_ks_, angle_thetas_ = self.define_angle_potentials()
        angles = get_all_angles(mol)
        pos_mat = mol.get_position_matrix()

        for angle in angles:
            atom1, atom2, atom3 = angle
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            name3 = f"{atom3.__class__.__name__}{atom3.get_id()+1}"
            table = str.maketrans("", "", digits)
            sorted_name = tuple(
                sorted(
                    [i.translate(table) for i in (name1, name2, name3)]
                )
            )

            try:
                angle_k = angle_ks_[sorted_name]
                angle_theta = angle_thetas_[sorted_name]
                if isinstance(angle_theta, int) or isinstance(
                    angle_theta, float
                ):
                    pass
                elif angle_theta[0] == "check":
                    a1id = atom1.get_id()
                    a2id = atom2.get_id()
                    a3id = atom3.get_id()
                    vector1 = pos_mat[a2id] - pos_mat[a1id]
                    vector2 = pos_mat[a2id] - pos_mat[a3id]
                    curr_angle = np.degrees(
                        angle_between(vector1, vector2)
                    )
                    if curr_angle < angle_theta[1]["cut"]:
                        angle_theta = angle_theta[1]["min"]
                    elif curr_angle >= angle_theta[1]["cut"]:
                        angle_theta = angle_theta[1]["max"]

            except KeyError:
                continue

            angle_string += (
                f"{name2} {name1} {name3} {angle_k} {angle_theta} "
                f"{self._angle_cutoff} {self._angle_cutoff} "
                f"{self._angle_cutoff} \n"
            )

        return angle_string

    def _write_gulp_input(self, mol):
        top_string = "opti conv cartesian\n"
        coord_string, mass_string = self._get_coord_mass_string(mol)
        bond_string = self._get_bond_string(mol)
        angle_string = self._get_angle_string(mol)
        settings_string = (
            "\nmaxcyc 500\n"
            # f'output xyz movie {filename}_traj.xyz\n'
            f"output xyz {self._output_xyz}\n"
        )

        with open(self._gulp_in, "w") as f:
            f.write(top_string)
            f.write(coord_string)
            f.write(mass_string)
            f.write(bond_string)
            f.write(angle_string)
            f.write(settings_string)

    def optimize(self, molecule):
        self._write_gulp_input(mol=molecule)
        self._run_gulp()
        return self._extract_gulp()
