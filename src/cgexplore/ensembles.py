#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for ensemble and trajectory classes.

Author: Andrew Tarzia

"""

import json
import os
from collections import abc
from dataclasses import dataclass

import numpy as np
import stk


@dataclass
class Conformer:
    molecule: stk.Molecule
    energy_decomposition: dict
    conformer_id: int | None = None
    source: str | None = None


@dataclass
class Timestep:
    molecule: stk.Molecule
    timestep: float


class Ensemble:
    def __init__(
        self,
        base_molecule: stk.Molecule,
        base_mol_path: str,
        conformer_xyz: str,
        data_json: str,
        overwrite: bool,
    ) -> None:
        self._base_molecule = base_molecule
        self._molecule_num_atoms = base_molecule.get_num_atoms()
        self._base_mol_path = base_mol_path
        self._conformer_xyz = conformer_xyz
        self._data_json = data_json
        if overwrite:
            self._base_molecule.write(self._base_mol_path)
            if os.path.exists(self._conformer_xyz):
                os.remove(self._conformer_xyz)
            if os.path.exists(self._data_json):
                os.remove(self._data_json)
            self._data = {}
            self._trajectory = {}
        else:
            self._data = self.load_data()
            self._trajectory = self.load_trajectory()

    def get_num_conformers(self) -> int:
        return len(self._data)

    def write_conformers_to_file(self) -> None:
        with open(self._conformer_xyz, "w") as f:
            for conf in self._trajectory:
                xyz_string = self._trajectory[conf]
                f.write("\n".join(xyz_string))
        with open(self._data_json, "w") as f:
            json.dump(self._data, f, indent=4)

    def add_conformer(self, conformer: Conformer, source: str) -> None:
        if conformer.conformer_id is None:
            conf_id = self.get_num_conformers()
        else:
            conf_id = conformer.conformer_id

        assert conf_id not in self._trajectory
        conformer_label = f"conf {conf_id}"

        # Add structure to XYZ file.
        xyz_string = stk.XyzWriter().to_string(conformer.molecule)
        # Add label to xyz string.
        xyz_list = xyz_string.split("\n")
        xyz_list[1] = conformer_label
        self._trajectory[conf_id] = xyz_list
        # Add data to JSON dictionary.
        conf_data = {
            i: conformer.energy_decomposition[i]
            for i in conformer.energy_decomposition
        }
        conf_data["source"] = source
        self._data[conf_id] = conf_data

    def load_trajectory(self) -> dict[int, list[str]]:
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
                trajectory[int(conf_id)] = conf_lines
        return trajectory

    def _yield_from_xyz(self) -> abc.Iterator[Conformer]:
        num_atoms = self._molecule_num_atoms
        new_pos_mat: list = []

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
                    energy_decomposition={},
                    conformer_id=conf_id,
                )

    def yield_conformers(self) -> abc.Iterator[Conformer]:
        for conf_id in self._trajectory:
            yield self.get_conformer(conf_id)

    def get_lowest_e_conformer(self) -> Conformer:
        temp_energy_conformerid: int | str
        try:
            temp_energy_conformerid = 0
            min_energy = self._data[temp_energy_conformerid]["total energy"][0]
        except KeyError:
            # This try statement is for backwards compatability.
            # Conformer IDs should be strings.
            temp_energy_conformerid = "0"
            data = self._data[temp_energy_conformerid]  # type: ignore[index]
            min_energy = data["total energy"][0]

        min_energy_conformerid = 0
        for confid in self._data:
            conf_energy = self._data[confid]["total energy"][0]
            if conf_energy < min_energy:
                min_energy = conf_energy
                min_energy_conformerid = confid  # type: ignore[assignment]

        return self.get_conformer(min_energy_conformerid)

    def get_conformer(self, conf_id: int | str) -> Conformer:
        if conf_id not in self._data:
            if str(conf_id) not in self._data:
                raise ValueError(
                    f"conformer {conf_id} not found in ensemble "
                    f"({self._data_json})."
                    "Strict handling of `conf_id` is coming."
                    "Current types: "
                    f"{list(type(i) for i in self._data.keys())}"
                )
            else:
                conf_id = str(conf_id)

        conf_lines = self._trajectory[int(conf_id)]
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

        conf_data = self._data[conf_id]  # type: ignore[index]
        source = conf_data["source"]
        energy_decomp = {i: conf_data[i] for i in conf_data if i != "source"}
        return Conformer(
            molecule=(
                self._base_molecule.with_position_matrix(np.array(new_pos_mat))
            ),
            conformer_id=int(conf_id),
            energy_decomposition=energy_decomp,
            source=source,
        )

    def get_base_molecule(self) -> stk.Molecule:
        return self._base_molecule

    def get_molecule_num_atoms(self) -> int:
        return self._molecule_num_atoms

    def load_data(self) -> dict[int, dict]:
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
