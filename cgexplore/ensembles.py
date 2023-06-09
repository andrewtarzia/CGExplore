#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for ensemble and trajectory classes.

Author: Andrew Tarzia

"""

import stk
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Conformer:
    molecule: stk.Molecule
    conformer_id: int = None
    energy_decomposition: dict = None


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

    def get_num_conformers(self):
        if not os.path.exists(self._data_json):
            return 0
        else:
            data = self.get_data()
            return len(data)

    def add_conformer(self, structure, energy_decomposition, source):
        conf_id = self.get_num_conformers()
        conformer_label = f"conf {conf_id}"

        # Add structure to XYZ file.
        xyz_string = stk.XyzWriter().to_string(structure)
        # Add label to xyz string.
        xyz_string = xyz_string.split("\n")
        xyz_string[1] = conformer_label
        with open(self._conformer_xyz, "a") as f:
            f.write("\n".join(xyz_string))

        # Add data to JSON dictionary.
        conf_data = {
            i: energy_decomposition[i] for i in energy_decomposition
        }
        conf_data["source"] = source
        data = self.get_data()
        data[conf_id] = conf_data
        with open(self._data_json, "w") as f:
            json.dump(data, f, indent=4)

    def _yield_from_xyz(self):
        num_atoms = self._molecule_num_atoms
        new_pos_mat = []

        with open(self._conformer_xyz, "r") as f:
            lines = f.readlines()
            conformer_starts = range(0, len(lines), num_atoms + 2)

            for cs in conformer_starts:
                conf_lines = lines[cs : cs + num_atoms + 2]
                conf_id = conf_lines[1].rstrip().split()[1]
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
        yield from self._yield_from_xyz()

    def get_conformer(self, conf_id):
        count = 0
        for conformer in self._yield_from_xyz():
            count += 1
            if conformer.conformer_id == conf_id:
                return conformer

        raise ValueError(
            f"conformer {conf_id} not found in ensemble with "
            f"{count} conformers"
        )

    def get_base_molecule(self):
        return self._base_molecule

    def get_molecule_num_atoms(self):
        return self._molecule_num_atoms

    def get_data(self):
        try:
            with open(self._data_json, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return str(self)
