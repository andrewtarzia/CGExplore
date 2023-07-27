#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for ensemble and trajectory classes.

Author: Andrew Tarzia

"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import stk


@dataclass
class Conformer:
    molecule: stk.Molecule
    conformer_id: int = None
    energy_decomposition: dict = None
    source: str = None


@dataclass
class Timestep:
    molecule: stk.Molecule
    timestep: int


class Trajectory:
    def __init__(
        self,
        base_molecule,
        data_path,
        traj_path,
        forcefield_path,
        output_path,
        temperature,
        random_seed,
        num_steps,
        time_step,
        friction,
        reporting_freq,
        traj_freq,
    ):
        self._base_molecule = base_molecule
        self._data_path = data_path
        self._traj_path = traj_path
        self._output_path = output_path
        self._temperature = temperature
        self._num_steps = num_steps
        self._time_step = time_step
        self._reporting_freq = reporting_freq
        self._traj_freq = traj_freq
        self._num_confs = int(self._num_steps / self._traj_freq)

    def get_data(self):
        return pd.read_csv(self._data_path)

    def yield_conformers(self):
        raise NotImplementedError()

    def get_base_molecule(self):
        return self._base_molecule

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(steps={self._num_steps}, "
            f"conformers={self._num_confs})"
        )

    def __repr__(self) -> str:
        return str(self)


class Ensemble:
    def __init__(
        self,
        base_molecule,
        base_mol_path,
        conformer_xyz,
        data_json,
        overwrite,
    ):
        self._base_molecule = base_molecule
        self._molecule_num_atoms = base_molecule.get_num_atoms()
        self._base_mol_path = base_mol_path
        self._base_molecule.write(self._base_mol_path)
        self._conformer_xyz = conformer_xyz
        self._data_json = data_json
        if overwrite:
            if os.path.exists(self._conformer_xyz):
                os.remove(self._conformer_xyz)
            if os.path.exists(self._data_json):
                os.remove(self._data_json)
            self._data = {}
            self._trajectory = {}
        else:
            self._data = self.load_data()
            self._trajectory = self.load_trajectory()

    def get_num_conformers(self):
        return len(self._data)

    def write_conformers_to_file(self):
        with open(self._conformer_xyz, "w") as f:
            for conf in self._trajectory:
                xyz_string = self._trajectory[conf]
                f.write("\n".join(xyz_string))
        with open(self._data_json, "w") as f:
            json.dump(self._data, f, indent=4)

    def add_conformer(self, conformer, source):
        if conformer.conformer_id is None:
            conf_id = self.get_num_conformers()
        else:
            conf_id = conformer.conformer_id

        assert conf_id not in self._trajectory
        conformer_label = f"conf {conf_id}"

        # Add structure to XYZ file.
        xyz_string = stk.XyzWriter().to_string(conformer.molecule)
        # Add label to xyz string.
        xyz_string = xyz_string.split("\n")
        xyz_string[1] = conformer_label
        self._trajectory[conf_id] = xyz_string
        # Add data to JSON dictionary.
        conf_data = {
            i: conformer.energy_decomposition[i]
            for i in conformer.energy_decomposition
        }
        conf_data["source"] = source
        self._data[conf_id] = conf_data

    def load_trajectory(self):
        num_atoms = self._molecule_num_atoms
        trajectory = {}
        with open(self._conformer_xyz, "r") as f:
            lines = f.readlines()
            conformer_starts = range(0, len(lines), num_atoms + 2)
            for cs in conformer_starts:
                conf_lines = [
                    i.strip() for i in lines[cs : cs + num_atoms + 2]
                ]
                conf_id = conf_lines[1].rstrip().split()[1]
                trajectory[conf_id] = conf_lines
        return trajectory

    def _yield_from_xyz(self):
        num_atoms = self._molecule_num_atoms
        new_pos_mat = []

        with open(self._conformer_xyz, "r") as f:
            lines = f.readlines()
            conformer_starts = range(0, len(lines), num_atoms + 2)

            for cs in conformer_starts:
                conf_lines = lines[cs : cs + num_atoms + 2]
                conf_id = int(conf_lines[1].rstrip().split()[1])
                new_pos_mat = []
                for exyz in conf_lines[2:]:
                    x = float(exyz.rstrip().split()[1])
                    y = float(exyz.rstrip().split()[2])
                    z = float(exyz.rstrip().split()[3])
                    new_pos_mat.append([x, y, z])

                if len(new_pos_mat) != num_atoms:
                    raise ValueError(
                        f"num atoms ({num_atoms}) does not match "
                        "size of collected position matrix "
                        f"({len(new_pos_mat)})."
                    )
                yield Conformer(
                    molecule=(
                        self._base_molecule.with_position_matrix(
                            np.array(new_pos_mat)
                        )
                    ),
                    conformer_id=conf_id,
                )

    def yield_conformers(self):
        for conf_id in self._trajectory:
            yield self.get_conformer(conf_id)

    def get_lowest_e_conformer(self):
        try:
            min_energy_conformerid = 0
            min_energy = self._data[min_energy_conformerid]["total energy"][0]
        except KeyError:
            min_energy_conformerid = "0"
            min_energy = self._data[min_energy_conformerid]["total energy"][0]

        for confid in self._data:
            conf_energy = self._data[confid]["total energy"][0]
            if conf_energy < min_energy:
                min_energy = conf_energy
                min_energy_conformerid = confid

        return self.get_conformer(min_energy_conformerid)

    def get_conformer(self, conf_id):
        if conf_id not in self._data:
            raise ValueError(f"conformer {conf_id} not found in ensemble.")
        conf_lines = self._trajectory[conf_id]
        extracted_id = int(conf_lines[1].strip().split()[1])
        if extracted_id != int(conf_id):
            raise ValueError(f"Asked for {conf_id}, got {extracted_id}")
        new_pos_mat = [
            [float(j) for j in i.strip().split()[1:]]
            for i in conf_lines[2:]
            if i != ""
        ]
        if len(new_pos_mat) != self._molecule_num_atoms:
            raise ValueError(
                f"Num atoms ({len(new_pos_mat)}) in xyz does not match "
                f"base molecule ({self._molecule_num_atoms})"
            )
        conf_data = self._data[conf_id]
        source = conf_data["source"]
        energy_decomp = {i: conf_data[i] for i in conf_data if i != "source"}
        return Conformer(
            molecule=(
                self._base_molecule.with_position_matrix(np.array(new_pos_mat))
            ),
            conformer_id=conf_id,
            energy_decomposition=energy_decomp,
            source=source,
        )

    def get_base_molecule(self):
        return self._base_molecule

    def get_molecule_num_atoms(self):
        return self._molecule_num_atoms

    def load_data(self):
        try:
            with open(self._data_json, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_confs={self.get_num_conformers()})"
        )

    def __repr__(self) -> str:
        return str(self)
