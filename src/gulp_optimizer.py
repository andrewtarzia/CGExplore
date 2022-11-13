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
from dataclasses import dataclass
from itertools import combinations
import logging

from env_set import gulp_path
from utilities import get_all_angles, angle_between, get_all_torsions

from beads import guest_beads


def lorentz_berthelot_sigma_mixing(sigma1, sigma2):
    return (sigma1 + sigma2) / 2


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
            [
                i
                for i in (
                    self.atom1_type,
                    self.atom2_type,
                    self.atom3_type,
                )
            ]
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
    def __init__(
        self,
        fileprefix,
        output_dir,
        param_pool,
        bonds,
        angles,
        torsions,
        vdw,
        max_cycles=500,
        conjugate_gradient=False,
    ):
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._param_pool = {i.element_string: i for i in param_pool}
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
        self._bond_cutoff = 80
        self._angle_cutoff = 80
        self._torsion_cutoff = 30
        self._lj_cutoff = 10
        self._maxcycles = max_cycles
        self._conjugate_gradient = conjugate_gradient
        self._vdw_on_types = tuple(
            i.element_string for i in guest_beads()
        )
        self._bonds = bonds
        self._angles = angles
        self._torsions = torsions
        self._vdw = vdw

    def _run_gulp(self):
        os.system(f"{gulp_path()} < {self._gulp_in} > {self._gulp_out}")

    def extract_gulp(self):
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
        if self._bonds is False:
            return ""
        logging.info(
            "OPT: you are not yet assigning different k values"
        )
        bond_k = 10

        bond_string = "harm\n"
        bonds = list(mol.get_bonds())

        for bond in bonds:
            atom1 = bond.get_atom1()
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            atom2 = bond.get_atom2()
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__

            try:
                cgbead1 = self._param_pool[estring1]
                cgbead2 = self._param_pool[estring2]
                bond_r = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.sigma,
                    sigma2=cgbead2.sigma,
                )
                bond_string += (
                    f"{name1} {name2}  {bond_k} {bond_r} "
                    f"{self._bond_cutoff}\n"
                )
            except KeyError:
                logging.info(
                    f"OPT: {(name1, name2)} bond not assigned."
                )
                continue

        return bond_string

    def _get_angle_string(self, mol):
        if self._angles is False:
            return ""
        logging.info(
            "OPT: you are not yet assigning different k values"
        )
        angle_k = 20

        angle_string = "three\n"
        angles = get_all_angles(mol)
        pos_mat = mol.get_position_matrix()

        for angle in angles:
            outer_atom1, centre_atom, outer_atom2 = angle
            outer_name1 = (
                f"{outer_atom1.__class__.__name__}"
                f"{outer_atom1.get_id()+1}"
            )
            centre_name = (
                f"{centre_atom.__class__.__name__}"
                f"{centre_atom.get_id()+1}"
            )
            outer_name2 = (
                f"{outer_atom2.__class__.__name__}"
                f"{outer_atom2.get_id()+1}"
            )
            # outer_estring1 = outer_atom1.__class__.__name__
            centre_estring = centre_atom.__class__.__name__
            # outer_estring2 = outer_atom2.__class__.__name__
            try:
                # outer_cgbead1 = self._param_pool[outer_estring1]
                centre_cgbead = self._param_pool[centre_estring]
                # outer_cgbead2 = self._param_pool[outer_estring2]

                acentered = centre_cgbead.angle_centered
                if isinstance(acentered, int) or isinstance(
                    acentered, float
                ):
                    angle_theta = acentered

                elif isinstance(acentered, tuple):
                    min_angle, max_angle, cut_angle = acentered
                    vector1 = (
                        pos_mat[centre_atom.get_id()]
                        - pos_mat[outer_atom1.get_id()]
                    )
                    vector2 = (
                        pos_mat[centre_atom.get_id()]
                        - pos_mat[outer_atom2.get_id()]
                    )
                    curr_angle = np.degrees(
                        angle_between(vector1, vector2)
                    )
                    if curr_angle < cut_angle:
                        angle_theta = min_angle
                    elif curr_angle >= cut_angle:
                        angle_theta = max_angle

                angle_string += (
                    f"{centre_name} {outer_name1} {outer_name2} "
                    f"{angle_k} {angle_theta} "
                    f"{self._angle_cutoff} {self._angle_cutoff} "
                    f"{self._angle_cutoff} \n"
                )

            except KeyError:
                logging.info(
                    f"OPT: {(outer_name1, centre_name, outer_name2)} "
                    f"angle not assigned (centered on {centre_name})."
                )
                continue

        return angle_string

    def _get_torsion_string(self, mol):
        if self._torsions is False:
            return ""
        logging.info("OPT: not setting torsion ks yet.")
        torsion_k = 1
        torsion_n = 1

        torsion_string = "torsion\n"
        torsions = get_all_torsions(mol)

        for torsion in torsions:
            atom1, atom2, atom3, atom4 = torsion
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            name3 = f"{atom3.__class__.__name__}{atom3.get_id()+1}"
            name4 = f"{atom4.__class__.__name__}{atom4.get_id()+1}"

            atom2_estring = atom2.__class__.__name__
            atom3_estring = atom3.__class__.__name__

            try:
                cgbead2 = self._param_pool[atom2_estring]
                cgbead3 = self._param_pool[atom3_estring]

                phi0 = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead2.angle_centered,
                    sigma2=cgbead3.angle_centered,
                )

                torsion_string += (
                    f"{name1} {name2} {name3} "
                    f"{name4} {torsion_k} {torsion_n} {phi0} "
                    f"{self._torsion_cutoff} "
                    f"{self._torsion_cutoff} "
                    f"{self._torsion_cutoff} "
                    f"{self._torsion_cutoff} \n"
                )

            except KeyError:
                logging.info(
                    f"OPT: {(name1, name2, name3, name4)} "
                    f"angle not assigned."
                )
                continue

        return torsion_string

    def _get_vdw_string(self, mol):
        if self._vdw is False:
            return ""
        logging.info(
            "OPT: only vdw interactions between host and guest."
        )

        vdw_string = "lennard epsilon\n"
        pairs = combinations(mol.get_atoms(), 2)

        for pair in pairs:
            atom1, atom2 = pair
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__
            guest_estrings = tuple(
                i
                for i in (estring1, estring2)
                if i in self._vdw_on_types
            )
            if len(guest_estrings) != 1:
                continue

            try:
                cgbead1 = self._param_pool[estring1]
                cgbead2 = self._param_pool[estring2]
                sigma = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.sigma,
                    sigma2=cgbead2.sigma,
                )
                epsilon = self._param_pool[guest_estrings[0]].epsilon
                vdw_string += (
                    f"{name1} {name2}  {epsilon} {sigma} "
                    f"{self._lj_cutoff}\n"
                )

            except KeyError:
                # logging.info(f"OPT: {sorted_name} vdw not assigned.")
                continue

        return vdw_string

    def _get_fix_atom_string(self, mol):

        string = "\n"
        count = 0
        for atom in mol.get_atoms():
            atom_no = atom.get_id() + 1
            estring = atom.__class__.__name__
            if estring in self._vdw_on_types:
                string += f"fix_atom {atom_no}\n"
                count += 1

        logging.info(f"OPT: fixing {count} guest beads.")
        return string

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
        if self._conjugate_gradient:
            top_string = "opti conj unit conv cartesian\n"
        else:
            top_string = "opti conv cartesian\n"
        coord_string, mass_string = self._get_coord_mass_string(mol)
        bond_string = self._get_bond_string(mol)
        angle_string = self._get_angle_string(mol)
        torsion_string = self._get_torsion_string(mol)
        vdw_string = self._get_vdw_string(mol)
        fix_string = self._get_fix_atom_string(mol)
        settings_string = (
            f"\nmaxcyc {self._maxcycles}\n"
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
            f.write(fix_string)
            f.write(settings_string)

    def optimize(self, molecule):
        self._write_gulp_input(mol=molecule)
        self._run_gulp()
        return self.extract_gulp()


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

    def get_xyz_atom_types(self, xyz_file):
        # Get atom types from an existing xyz file.
        atom_types = []
        with open(xyz_file, "r") as f:
            for line in f.readlines()[2:]:
                atom_types.append(line.rstrip().split(" ")[0])
        return atom_types

    def _convert_traj_to_xyz(self, input_xyz_template, output_traj):

        atom_types = self.get_xyz_atom_types(input_xyz_template)

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

    def write_conformer_xyz_file(
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

    def extract_gulp(self, input_xyz_template):

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

        self.write_conformer_xyz_file(
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
            "mdmaxtemp 100000000\n"
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
        return self.extract_gulp(input_xyz_template)
