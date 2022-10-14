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
import logging
from dataclasses import dataclass
from itertools import combinations

from env_set import gulp_path
from utilities import get_all_angles, angle_between, get_all_torsions


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
class Pair:
    atom1_type: str
    atom2_type: str

    def get_types(self):
        return tuple(
            sorted([i for i in (self.atom1_type, self.atom2_type)])
        )

    def get_unsortedtypes(self):
        return tuple([i for i in (self.atom1_type, self.atom2_type)])


@dataclass
class HarmBond(Pair):
    bond_r: float
    bond_k: float


@dataclass
class LennardJones(Pair):
    epsilon: float
    sigma: float


@dataclass
class Angle:
    atom1_type: str
    atom2_type: str
    atom3_type: str

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
class ThreeAngle(Angle):
    theta: float
    angle_k: float


@dataclass
class CheckedThreeAngle(Angle):
    cut_angle: float
    min_angle: float
    max_angle: float
    angle_k: float


@dataclass
class Torsion:
    atom1_type: str
    atom2_type: str
    atom3_type: str
    atom4_type: str
    n: int
    k: float
    phi0: float

    def get_types(self):
        return tuple(
            sorted(
                [
                    i
                    for i in (
                        self.atom1_type,
                        self.atom2_type,
                        self.atom3_type,
                        self.atom4_type,
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
                    self.atom4_type,
                )
            ]
        )


class CGGulpOptimizer:
    def __init__(self, fileprefix, output_dir, param_pool):
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._param_pool = param_pool
        self._gulp_in = os.path.join(
            self._output_dir, f"{self._fileprefix}.gin"
        )
        self._gulp_out = os.path.join(
            self._output_dir, f"{self._fileprefix}.ginout"
        )
        self._output_xyz = os.path.join(
            self._output_dir, f"{self._fileprefix}_opted.xyz"
        )
        self._mass = 1
        # This is the underlying scale for distances.
        self._sigma = 1
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 15

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

    def define_torsion_potentials(self):
        torsions = ()
        new_torsions = self._update_torsions(torsions)

        return IntSet(new_torsions)

    def define_vdw_potentials(self):
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
                # logging.info(f"{sorted_name} bond not assigned.")
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
                        a1id = int(re.findall(r"\d+", outer1)[0]) - 1
                        a2id = (
                            int(re.findall(r"\d+", centre_atom)[0]) - 1
                        )
                        a3id = int(re.findall(r"\d+", outer2)[0]) - 1
                        vector1 = pos_mat[a2id] - pos_mat[a1id]
                        vector2 = pos_mat[a2id] - pos_mat[a3id]
                        curr_angle = np.degrees(
                            angle_between(vector1, vector2)
                        )
                        if curr_angle < pot.cut_angle:
                            angle_theta = pot.min_angle
                        elif curr_angle >= pot.cut_angle:
                            angle_theta = pot.max_angle
                        angle_k = pot.angle_k
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
                # logging.info(f"{sorted_name} angle not assigned.")
                continue

        return angle_string

    def _get_torsion_string(self, mol):
        torsion_set = self.define_torsion_potentials()
        torsion_set_dict = torsion_set.get_set_dict()

        torsion_string = "torsion\n"
        torsions = get_all_torsions(mol)

        for torsion in torsions:
            atom1, atom2, atom3, atom4 = torsion
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            name3 = f"{atom3.__class__.__name__}{atom3.get_id()+1}"
            name4 = f"{atom4.__class__.__name__}{atom4.get_id()+1}"
            table = str.maketrans("", "", digits)
            sorted_name = tuple(
                sorted(
                    [
                        i.translate(table)
                        for i in (name1, name2, name3, name4)
                    ]
                )
            )

            try:
                tset = torsion_set_dict[sorted_name]

                for pot in tset:
                    # Want to check ordering here.
                    if pot.atom2_type in name2:
                        centre_atom1 = name2
                        centre_atom2 = name3
                        outer1 = name1
                        outer2 = name4
                    elif pot.atom2_type in name3:
                        centre_atom1 = name3
                        centre_atom2 = name2
                        outer1 = name4
                        outer2 = name1

                    n = pot.n
                    k = pot.k
                    phi0 = pot.phi0

                    torsion_string += (
                        f"{outer1} {centre_atom1} {centre_atom2} "
                        f"{outer2} {k} {n} {phi0} "
                        f"{self._torsion_cutoff} "
                        f"{self._torsion_cutoff} "
                        f"{self._torsion_cutoff} "
                        f"{self._torsion_cutoff} \n"
                    )

            except KeyError:
                # logging.info(f"{sorted_name} torsion not assigned.")
                continue

        return torsion_string

    def _get_vdw_string(self, mol):
        vdw_set = self.define_vdw_potentials()
        vdw_set_pairs = vdw_set.get_set_dict()
        vdw_string = "lennard epsilon\n"
        pairs = combinations(mol.get_atoms(), 2)

        for pair in pairs:
            atom1, atom2 = pair
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            table = str.maketrans("", "", digits)
            sorted_name = tuple(
                sorted([i.translate(table) for i in (name1, name2)])
            )
            try:
                pair = vdw_set_pairs[sorted_name]
                for pot in pair:
                    vdw_string += (
                        f"{name1} {name2}  {pot.epsilon} {pot.sigma} "
                        f"{self._lj_cutoff}\n"
                    )
            except KeyError:
                # logging.info(f"{sorted_name} vdw not assigned.")
                continue

        return vdw_string

    def _update_bonds(self, bonds):
        new_bonds = []
        for bond in bonds:
            bond_type = bond.get_unsortedtypes()
            if bond_type in self._param_pool["bonds"]:
                k, r = self._param_pool["bonds"][bond_type]
                nbond = HarmBond(bond.atom1_type, bond.atom2_type, r, k)
                new_bonds.append(nbond)
            else:
                new_bonds.append(bond)

        new_bonds = tuple(new_bonds)
        return new_bonds

    def _update_angles(self, angles):
        new_angles = []
        for angle in angles:
            angle_type = angle.get_unsortedtypes()
            if angle_type in self._param_pool["angles"]:
                k, theta = self._param_pool["angles"][angle_type]
                nangle = ThreeAngle(
                    angle.atom1_type,
                    angle.atom2_type,
                    angle.atom3_type,
                    theta,
                    k,
                )
                new_angles.append(nangle)
            else:
                new_angles.append(angle)

        new_angles = tuple(new_angles)
        return new_angles

    def _update_torsions(self, torsions):
        new_torsions = []
        for torsion in torsions:
            torsion_type = torsion.get_unsortedtypes()
            if torsion_type in self._param_pool["torsions"]:
                n, k, phi0 = self._param_pool["torsions"][torsion_type]
                ntorsion = Torsion(
                    torsion.atom1_type,
                    torsion.atom2_type,
                    torsion.atom3_type,
                    torsion.atom4_type,
                    n=n,
                    k=k,
                    phi0=phi0,
                )
                new_torsions.append(ntorsion)
            else:
                new_torsions.append(torsion)

        new_torsions = tuple(new_torsions)
        return new_torsions

    def _update_pairs(self, pairs):
        new_pairs = []
        for pair in pairs:
            pair_type = pair.get_unsortedtypes()
            if pair_type in self._param_pool["pairs"]:
                epsilon, sigma = self._param_pool["pairs"][pair_type]
                npair = LennardJones(
                    pair.atom1_type,
                    pair.atom2_type,
                    epsilon,
                    sigma,
                )
                new_pairs.append(npair)
            else:
                new_pairs.append(pair)

        new_pairs = tuple(new_pairs)
        return new_pairs

    def _write_gulp_input(self, mol):
        top_string = "opti conv cartesian\n"
        coord_string, mass_string = self._get_coord_mass_string(mol)
        bond_string = self._get_bond_string(mol)
        angle_string = self._get_angle_string(mol)
        torsion_string = self._get_torsion_string(mol)
        vdw_string = self._get_vdw_string(mol)
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
            f.write(torsion_string)
            f.write(vdw_string)
            f.write(settings_string)

    def optimize(self, molecule):
        self._write_gulp_input(mol=molecule)
        self._run_gulp()
        return self._extract_gulp()


class CGGulpMD(CGGulpOptimizer):
    def __init__(self, fileprefix, output_dir, param_pool):

        super().__init__(fileprefix, output_dir, param_pool)
        self._gulp_in = os.path.join(
            self._output_dir, f"{self._fileprefix}_md.gin"
        )
        self._gulp_out = os.path.join(
            self._output_dir, f"{self._fileprefix}_md.ginout"
        )
        self._output_traj = os.path.join(
            self._output_dir, f"{self._fileprefix}.trg"
        )
        self._output_trajxyz = os.path.join(
            self._output_dir, f"{self._fileprefix}_traj.xyz"
        )
        self._output_xyz = os.path.join(
            self._output_dir, f"{self._fileprefix}_final.xyz"
        )

        self._integrator = "stochastic"
        self._ensemble = "nvt"
        self._temperature = 100
        self._equilbration = 1.0
        self._production = 10.0
        self._timestep = 1.0
        self._N_conformers = 20
        samples = float(self._production) / float(self._N_conformers)
        self._sample = samples
        self._write = samples

    def _convert_traj_to_xyz(self, input_xyz_template, output_traj):

        # Get atom types from an existing xyz file.
        atom_types = []
        with open(input_xyz_template, "r") as f:
            for line in f.readlines()[2:]:
                atom_types.append(line.rstrip().split(" ")[0])

        # Read in lines from trajectory file.
        with open(output_traj, "r") as f:
            lines = f.readlines()

        # Split file using strings.
        timesteps = "".join(lines).split("#  Time/KE/E/T")[1:]
        trajectory_data = {}
        xyz_traj_lines = []
        for ts, cont in enumerate(timesteps):
            ts_data = {}
            time_section = cont.split("#  Coordinates\n")[0]
            coords_section = cont.split("#  Coordinates\n")[1].split(
                "#  Velocities\n"
            )[0]
            vels_section = cont.split("#  Velocities\n")[1].split(
                "#  Derivatives \n"
            )[0]
            derivs_section = cont.split("#  Derivatives \n")[1].split(
                "#  Site energies \n"
            )[0]
            sites_section = cont.split("#  Site energies \n")[1]

            ts_data["time"] = float(
                [i for i in time_section.strip().split(" ") if i][0]
            )
            ts_data["KE"] = float(
                [i for i in time_section.strip().split(" ") if i][1]
            )
            ts_data["E"] = float(
                [i for i in time_section.strip().split(" ") if i][2]
            )
            ts_data["T"] = float(
                [i for i in time_section.strip().split(" ") if i][3]
            )
            ts_data["coords"] = [
                [i for i in li.split(" ") if i]
                for li in coords_section.split("\n")[:-1]
            ]
            ts_data["vels"] = [
                [i for i in li.split(" ") if i]
                for li in vels_section.split("\n")[:-1]
            ]
            ts_data["derivs"] = [
                [i for i in li.split(" ") if i]
                for li in derivs_section.split("\n")[:-1]
            ]
            ts_data["sites"] = [
                [i for i in li.split(" ") if i]
                for li in sites_section.split("\n")[:-1]
            ]

            trajectory_data[ts] = ts_data

            # Write XYZ string for XYZ traj file.
            xyz_string = (
                f"{len(ts_data['coords'])}\n"
                f"{ts_data['time']},{ts_data['KE']},"
                f"{ts_data['E']},{ts_data['T']}\n"
            )

            for i, coord in enumerate(ts_data["coords"]):
                site_E = ts_data["sites"][i][0]
                xyz_string += (
                    f"{atom_types[i]} {round(float(coord[0]), 5)} "
                    f"{round(float(coord[1]), 5)} "
                    f"{round(float(coord[2]), 5)} {site_E}\n"
                )

            xyz_traj_lines.append(xyz_string)

        return atom_types, trajectory_data, xyz_traj_lines

    def _calculate_lowest_energy_conformer(self, trajectory_data):
        energies = [trajectory_data[ts]["E"] for ts in trajectory_data]
        min_energy = min(energies)
        min_ts = list(trajectory_data.keys())[
            energies.index(min_energy)
        ]
        return min_ts

    def _write_conformer_xyz_file(
        self,
        ts,
        ts_data,
        filename,
        atom_types,
    ):

        coords = ts_data["coords"]
        xyz_string = (
            f"{len(ts_data['coords'])}\n"
            f"{ts}, {round(ts_data['time'], 2)}, "
            f"{round(ts_data['KE'], 2)}, "
            f"{round(ts_data['E'], 2)}, {round(ts_data['T'], 2)}\n"
        )

        for i, coord in enumerate(coords):
            xyz_string += (
                f"{atom_types[i]} {round(float(coord[0]), 5)} "
                f"{round(float(coord[1]), 5)} "
                f"{round(float(coord[2]), 5)}\n"
            )
        with open(filename, "w") as f:
            f.write(xyz_string)

    def _extract_gulp(self, input_xyz_template):

        # Convert GULP trajectory file to xyz trajectory.
        atom_types, t_data, xyz_lines = self._convert_traj_to_xyz(
            input_xyz_template=input_xyz_template,
            output_traj=self._output_traj,
        )
        # Write XYZ trajectory file.
        with open(self._output_trajxyz, "w") as f:
            for line in xyz_lines:
                f.write(line)

        # Find lowest energy conformation and output to XYZ.
        min_ts = self._calculate_lowest_energy_conformer(t_data)

        self._write_conformer_xyz_file(
            ts=min_ts,
            ts_data=t_data[min_ts],
            filename=self._output_xyz,
            atom_types=atom_types,
        )

        run_data = {"traj": t_data}

        return run_data

    def _write_gulp_input(self, mol):
        top_string = "md conv cartesian\n"
        coord_string, mass_string = self._get_coord_mass_string(mol)
        bond_string = self._get_bond_string(mol)
        angle_string = self._get_angle_string(mol)
        torsion_string = self._get_torsion_string(mol)
        vdw_string = self._get_vdw_string(mol)
        settings_string = (
            f"integrator {self._integrator}\n"
            f"ensemble {self._ensemble}\n"
            f"temperature {self._temperature}\n"
            f"equilbration {self._equilbration} ps\n"
            f"production {self._production} ps\n"
            f"timestep {self._timestep} fs\n"
            f"sample {self._sample} ps\n"
            f"write {self._write} ps\n"
            f"output trajectory ascii {self._output_traj}\n"
        )

        with open(self._gulp_in, "w") as f:
            f.write(top_string)
            f.write(coord_string)
            f.write(mass_string)
            f.write(bond_string)
            f.write(angle_string)
            f.write(torsion_string)
            f.write(vdw_string)
            f.write(settings_string)

    def optimize(self, molecule):
        input_xyz_template = "gulp_template.xyz"
        molecule.write(input_xyz_template)
        self._write_gulp_input(mol=molecule)
        self._run_gulp()
        return self._extract_gulp(input_xyz_template)
