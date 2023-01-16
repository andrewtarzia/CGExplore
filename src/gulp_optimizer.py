#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG Gulp optimizer.

Author: Andrew Tarzia

"""

import os
import re
import logging

from env_set import gulp_path

from beads import guest_beads
from optimizer import CGOptimizer


def extract_gulp_optimisation(gulp_out):
    with open(gulp_out, "r") as f:
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


class MDEmptyTrajcetoryError(Exception):
    ...


class CGGulpOptimizer(CGOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        param_pool,
        custom_torsion_set,
        bonds,
        angles,
        torsions,
        vdw,
        max_cycles=500,
        conjugate_gradient=False,
    ):
        super().__init__(
            fileprefix,
            output_dir,
            param_pool,
            bonds,
            angles,
            torsions,
            vdw,
        )
        self._gulp_in = os.path.join(
            self._output_dir, f"{self._fileprefix}.gin"
        )
        self._gulp_out = os.path.join(
            self._output_dir, f"{self._fileprefix}.ginout"
        )
        self._output_xyz = os.path.join(
            self._output_dir, f"{self._fileprefix}_opted.xyz"
        )
        self._maxcycles = max_cycles
        self._conjugate_gradient = conjugate_gradient
        self._vdw_on_types = tuple(
            i.element_string for i in guest_beads()
        )
        self._custom_torsion_set = custom_torsion_set

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

    def _get_coord_string(self, molecule):
        coord_string = "cartesian\n"
        mass_string = ""

        pos_mat = molecule.get_position_matrix()
        atoms = list(molecule.get_atoms())
        for atom, pos_ in zip(atoms, pos_mat):
            name = f"{atom.__class__.__name__}{atom.get_id()+1}"
            coord_string += (
                f"{name} {round(pos_[0], 2)} {round(pos_[1], 2)} "
                f"{round(pos_[2], 2)}\n"
            )
            mass_string += f"mass {name} {self._mass}\n"

        return coord_string, mass_string

    def _get_bond_string(self, molecule):
        bond_string = "harm\n"
        for bond_info in self._yield_bonds(molecule):
            name1, name2, cgbead1, cgbead2, bond_k, bond_r = bond_info
            bond_string += (
                f"{name1} {name2}  {bond_k} {bond_r} "
                f"{self._bond_cutoff}\n"
            )

        return bond_string

    def _get_angle_string(self, molecule):
        angle_string = "three\n"
        for angle_info in self._yield_angles(molecule):
            (
                centre_name,
                outer_name1,
                outer_name2,
                centre_cgbead,
                outer_cgbead1,
                outer_cgbead2,
                angle_k,
                angle_theta,
            ) = angle_info

            angle_string += (
                f"{centre_name} {outer_name1} {outer_name2} "
                f"{angle_k} {angle_theta} "
                f"{self._angle_cutoff} {self._angle_cutoff} "
                f"{self._angle_cutoff} \n"
            )

        return angle_string

    def _yield_custom_torsions(self, molecule, chain_length=5):
        if self._custom_torsion_set is None:
            return ""

        torsions = self._get_new_torsions(molecule, chain_length)
        for torsion in torsions:
            atom1, atom2, atom3, atom4, atom5 = torsion
            names = list(
                f"{i.__class__.__name__}{i.get_id()+1}" for i in torsion
            )

            atom_estrings = list(i.__class__.__name__ for i in torsion)
            cgbeads = list(
                self._get_cgbead_from_element(i) for i in atom_estrings
            )
            cgbead_types = tuple(i.bead_type for i in cgbeads)
            if cgbead_types in self._custom_torsion_set:
                phi0 = self._custom_torsion_set[cgbead_types][0]
                torsion_k = self._custom_torsion_set[cgbead_types][1]
                torsion_n = -1
                yield (
                    names[0],
                    names[1],
                    names[3],
                    names[4],
                    torsion_k,
                    torsion_n,
                    phi0,
                )
            continue

    def _get_torsion_string(self, molecule):
        torsion_string = "torsion\n"

        for torsion_info in self._yield_custom_torsions(molecule):
            (
                name1,
                name2,
                name3,
                name4,
                torsion_k,
                torsion_n,
                phi0,
            ) = torsion_info
            torsion_string += (
                f"{name1} {name2} {name3} "
                f"{name4} {torsion_k} {torsion_n} {phi0} "
                f"{self._torsion_cutoff} "
                f"{self._torsion_cutoff} "
                f"{self._torsion_cutoff} "
                f"{self._torsion_cutoff} \n"
            )

        return torsion_string

    def _get_vdw_string(self, molecule):
        if self._vdw is False:
            return ""
        logging.info(
            "OPT: only vdw interactions between host and guest."
        )

        vdw_string = "lennard epsilon\n"
        for pair_info in self.__yield_nonbondeds(molecule):
            name1, name2, epsilon, sigma = pair_info

            vdw_string += (
                f"{name1} {name2}  {epsilon} {sigma} "
                f"{self._lj_cutoff}\n"
            )

        return vdw_string

    def _get_fix_atom_string(self, molecule):

        string = "\n"
        count = 0
        for atom in molecule.get_atoms():
            atom_no = atom.get_id() + 1
            estring = atom.__class__.__name__
            if estring in self._vdw_on_types:
                string += f"fix_atom {atom_no}\n"
                count += 1

        if count > 0:
            logging.info(f"OPT: fixing {count} guest beads.")
        return string

    def _write_gulp_input(self, molecule):
        if self._conjugate_gradient:
            top_string = "opti conj unit conv cartesian\n"
        else:
            top_string = "opti conv cartesian\n"
        coord_string, mass_string = self._get_coord_string(molecule)
        bond_string = self._get_bond_string(molecule)
        angle_string = self._get_angle_string(molecule)
        torsion_string = self._get_torsion_string(molecule)
        vdw_string = self._get_vdw_string(molecule)
        fix_string = self._get_fix_atom_string(molecule)
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
        self._write_gulp_input(molecule=molecule)
        self._run_gulp()
        return self.extract_gulp()


class CGGulpMD(CGGulpOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        param_pool,
        custom_torsion_set,
        bonds,
        angles,
        torsions,
        vdw,
    ):
        super(CGGulpOptimizer, self).__init__(
            fileprefix,
            output_dir,
            param_pool,
            bonds,
            angles,
            torsions,
            vdw,
        )

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
        self._input_xyz_template = os.path.join(
            self._output_dir, f"{self._fileprefix}_temp.xyz"
        )
        self._vdw_on_types = tuple(
            i.element_string for i in guest_beads()
        )

        self._custom_torsion_set = custom_torsion_set

    def get_xyz_atom_types(self, xyz_file):
        # Get atom types from an existing xyz file.
        atom_types = []
        with open(xyz_file, "r") as f:
            for line in f.readlines()[2:]:
                atom_types.append(line.rstrip().split(" ")[0])
        return atom_types

    def _convert_traj_to_xyz(self):

        atom_types = self.get_xyz_atom_types(self._input_xyz_template)

        # Read in lines from trajectory file.
        try:
            with open(self._output_traj, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            new_file = str(self._output_traj).replace(".trg", ".tr.trg")
            with open(new_file, "r") as f:
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
        try:
            min_energy = min(energies)
        except ValueError:
            raise MDEmptyTrajcetoryError(
                "MD failed to complete. Trajectory empty."
            )
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

    def extract_gulp(self):

        # Convert GULP trajectory file to xyz trajectory.
        atom_types, t_data, xyz_lines = self._convert_traj_to_xyz()
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

    def _write_gulp_input(self, molecule):
        top_string = "md conv cartesian\n"
        coord_string, mass_string = self._get_coord_string(molecule)
        bond_string = self._get_bond_string(molecule)
        angle_string = self._get_angle_string(molecule)
        torsion_string = self._get_torsion_string(molecule)
        vdw_string = self._get_vdw_string(molecule)

        integrator = "stochastic"
        ensemble = "nvt"
        temperature = 10
        tau_thermostat = 1.0
        equilbration = 1.0
        production = 20.0
        timestep = 0.5
        N_conformers = 20
        samples = float(production) / float(N_conformers)
        write = samples

        settings_string = (
            "mdmaxtemp 100000000\n"
            f"integrator {integrator}\n"
            f"ensemble {ensemble}\n"
            f"temperature {temperature}\n"
            f"tau_thermostat {tau_thermostat}\n"
            f"equilbration {equilbration} ps\n"
            f"production {production} ps\n"
            f"timestep {timestep} fs\n"
            f"sample {samples/ 10} ps\n"
            f"write {write} ps\n"
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
        molecule.write(self._input_xyz_template)
        self._write_gulp_input(molecule=molecule)
        self._run_gulp()
        return self.extract_gulp()
