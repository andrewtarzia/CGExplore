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
from dataclasses import dataclass

from env_set import gulp_path
from utilities import get_all_angles, angle_between


class IntSet:
    def __init__(self, interactions: tuple):
        self._interactions = interactions
        int_set = {}
        for inter in interactions:
            if inter.get_types() not in int_set:
                int_set[inter.get_types()] = []
            int_set[inter.get_types()].append(inter)

        self._int_set = int_set

    def get_set_dict(self):
        return self._int_set

    def get_keys(self):
        return self._int_set.keys()


@dataclass
class HarmBond:
    atom1_type: str
    atom2_type: str
    bond_r: float
    bond_k: float

    def get_types(self):
        return tuple(
            sorted([i for i in (self.atom1_type, self.atom2_type)])
        )

    def get_unsortedtypes(self):
        return tuple([i for i in (self.atom1_type, self.atom2_type)])


@dataclass
class ThreeAngle:
    atom1_type: str
    atom2_type: str
    atom3_type: str
    theta: float
    angle_k: float

    def get_types(self):
        return tuple(
            sorted(
                [
                    i
                    for i in (
                        self.atom1_type,
                        self.atom2_type,
                        self.atom3_type,
                    )
                ]
            )
        )

    def get_unsortedtypes(self):
        return tuple(
            [
                i
                for i in (
                    self.atom1_type,
                    self.atom2_type,
                    self.atom3_type,
                )
            ]
        )


@dataclass
class CheckedThreeAngle(ThreeAngle):
    atom1_type: str
    atom2_type: str
    atom3_type: str
    cut_angle: float
    min_angle: float
    max_angle: float


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
        bond_set = self.define_bond_potentials()
        bond_set_pairs = bond_set.get_set_dict()
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
                pair = bond_set_pairs[sorted_name]
                for pot in pair:
                    bond_string += (
                        f"{name1} {name2}  {pot.bond_k} {pot.bond_r} "
                        f"{self._bond_cutoff}\n"
                    )
            except KeyError:
                # logging.info(f"{sorted_name} not assigned.")
                continue

        return bond_string

    def _get_angle_string(self, mol):
        angle_set = self.define_angle_potentials()
        angle_set_triplets = angle_set.get_set_dict()

        angle_string = "three\n"
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
                triplet = angle_set_triplets[sorted_name]
                for pot in triplet:
                    # Want to check ordering here.
                    if pot.atom1_type in name1:
                        centre_atom = name1
                        outer1 = name2
                        outer2 = name3
                    elif pot.atom1_type in name2:
                        centre_atom = name2
                        outer1 = name1
                        outer2 = name3
                    elif pot.atom1_type in name3:
                        centre_atom = name3
                        outer1 = name1
                        outer2 = name2

                    if isinstance(pot, CheckedThreeAngle):
                        a1id = re.findall(r"\d+", outer1)
                        a2id = re.findall(r"\d+", centre_atom)
                        a3id = re.findall(r"\d+", outer2)
                        print(sorted_name)
                        print(triplet)
                        print(pot)
                        print(
                            atom1, atom2, atom3, pot, a1id, a2id, a3id
                        )
                        vector1 = pos_mat[a2id] - pos_mat[a1id]
                        vector2 = pos_mat[a2id] - pos_mat[a3id]
                        curr_angle = np.degrees(
                            angle_between(vector1, vector2)
                        )
                        if curr_angle < pot.angle_cut:
                            angle_theta = pot.angle_min
                        elif curr_angle >= pot.angle_cut:
                            angle_theta = pot.angle_max
                        raise SystemExit("need to check this")
                    else:
                        angle_k = pot.angle_k
                        angle_theta = pot.theta

                    angle_string += (
                        f"{centre_atom} {outer1} {outer2} "
                        f"{angle_k} {angle_theta} "
                        f"{self._angle_cutoff} {self._angle_cutoff} "
                        f"{self._angle_cutoff} \n"
                    )

            except KeyError:
                # logging.info(f"{sorted_name} not assigned.")
                continue

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
