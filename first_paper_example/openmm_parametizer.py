#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to run OpenMM tests.

Author: Andrew Tarzia

"""

import sys
import stk
import os
from openmm import openmm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
from rdkit import RDLogger

from cgexplore.utilities import (
    check_directory,
    angle_between,
    get_dihedral,
)
from cgexplore.openmm_optimizer import CGOMMOptimizer, CGOMMDynamics
from cgexplore.beads import produce_bead_library, bead_library_check

from env_set import cages, figures


def bond_function(x, k, r0):
    return (1 / 2) * k * (x - r0 / 1) ** 2


def angle_function(x, k, theta0):
    return (1 / 2) * k * (x - theta0) ** 2


def torsion_function(x, k, theta0, n):
    return k * (1 + np.cos(n * x - theta0))


def nonbond_function(x, epsilon1, sigma1, epsilon2, sigma2):
    return (
        np.sqrt(epsilon1 * epsilon2)
        * ((sigma1 + sigma2) / (2 * x)) ** 12
    )


def c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        bond_rs=(2,),
        angles=(90,),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def points_in_circum(r, n=100):
    return [
        (
            np.cos(2 * np.pi / n * x) * r,
            np.sin(2 * np.pi / n * x) * r,
        )
        for x in range(0, n + 1)
    ]


def random_test(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{bead.element_string}]([{bead.element_string}])"
            f"[{bead.element_string}]"
        ),
        position_matrix=[[0, 0, 0], [2, 0, 0], [1, 0, 0]],
    )

    temperature = 10

    runs = {
        0: (None, 1000, "-"),
        1: ("k", 1000, "--"),
        2: ("green", 2000, "-"),
        3: ("r", 2000, "--"),
        4: ("b", None, "-"),
        5: ("gold", None, "--"),
    }

    tdict = {}
    for run in runs:
        tdict[run] = {}
        logging.info(f"running MD random test; {run}")
        opt = CGOMMDynamics(
            fileprefix=f"mdr_{run}",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
            temperature=temperature,
            random_seed=runs[run][1],
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
        )
        trajectory = opt.run_dynamics(linear_bb)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            tdict[run][timestep] = (meas_temp, pot_energy, posmat)

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))

    for run in tdict:
        if run == 0:
            continue
        data = tdict[run]
        zero_data = tdict[0]
        xs = []
        tdata = []
        rdata = []
        edata = []
        for timestep in data:
            xs.append(timestep)
            m_temp = data[timestep][0]
            z_temp = zero_data[timestep][0]
            tdata.append(m_temp - z_temp)
            m_pe = data[timestep][1]
            z_pe = zero_data[timestep][1]
            edata.append(m_pe - z_pe)
            m_posmat = data[timestep][2]
            z_posmat = zero_data[timestep][2]
            rdata.append(
                np.sqrt(
                    np.sum((m_posmat - z_posmat) ** 2) / len(m_posmat)
                )
            )

        axs[0].plot(
            xs,
            rdata,
            c=runs[run][0],
            lw=2,
            linestyle=runs[run][2],
            # s=30,
            # edgecolor="none",
            alpha=1.0,
            label=f"run {run}",
        )
        axs[1].plot(
            xs,
            tdata,
            c=runs[run][0],
            lw=2,
            linestyle=runs[run][2],
            # s=30,
            # edgecolor="none",
            alpha=1.0,
            label=f"run {run}",
        )
        axs[2].plot(
            xs,
            edata,
            c=runs[run][0],
            lw=2,
            linestyle=runs[run][2],
            # s=30,
            # edgecolor="none",
            alpha=1.0,
            label=f"run {run}",
        )

    # ax.axhline(y=0, c="k", lw=2, linestyle="--")
    # ax.axvline(x=bead.angle_centered, c="k", lw=2)

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("timestep [s]", fontsize=16)
    axs[0].set_ylabel("RMSD [A]", fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_ylabel("deltaT [K]", fontsize=16)

    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_ylabel("deltaE [kJmol-1]", fontsize=16)
    axs[2].legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "random_test.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test1(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=f"[{bead.element_string}][{bead.element_string}]",
        position_matrix=[[0, 0, 0], [2, 0, 0]],
    )

    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test1; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl1_{temp}",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=False,
            torsions=False,
            vdw=False,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
        )
        trajectory = opt.run_dynamics(linear_bb)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            distance = np.linalg.norm(posmat[1] - posmat[0])
            tdict[temp][timestep] = (meas_temp, pot_energy, distance)

    coords = np.linspace(0, 5, 20)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l1_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[1] = np.array([1, 0, 0]) * coord
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=False,
            torsions=False,
            vdw=False,
        )
        energy = opt.calculate_energy(new_bb)
        distance = np.linalg.norm(new_posmat[1] - new_posmat[0])
        xys.append(
            (
                distance,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"{bead.bond_r} A, {bead.bond_k} kJ/mol/nm2",
        fontsize=16.0,
    )

    distances = [i[0] for i in xys]
    x = np.linspace(min(distances), max(distances), 100)
    ax.plot(
        x,
        bond_function(x / 10, bead.bond_k, bead.bond_r / 10),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance [A]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test2(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{bead.element_string}][{bead.element_string}]"
            f"[{bead.element_string}]"
        ),
        position_matrix=[[0, 0, 0], [2, 0, 0], [3, 0, 0]],
    )

    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test1; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl2_{temp}",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=False,
            torsions=False,
            vdw=False,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
        )
        trajectory = opt.run_dynamics(linear_bb)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            distance1 = np.linalg.norm(posmat[1] - posmat[0])
            distance2 = np.linalg.norm(posmat[2] - posmat[1])
            tdict[temp][timestep] = (
                meas_temp,
                pot_energy,
                distance1,
                distance2,
            )

    coords1 = np.linspace(0.6, 2, 25)
    coords2 = np.linspace(0.6, 2, 25)
    xys = []
    for i, (coord1, coord2) in enumerate(
        itertools.product(coords1, coords2)
    ):
        name = f"l2_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[1] = new_posmat[1] * coord1
        new_posmat[2] = new_posmat[2] * coord2
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=False,
            torsions=False,
            vdw=False,
        )
        energy = opt.calculate_energy(new_bb)
        distance1 = np.linalg.norm(new_posmat[1] - new_posmat[0])
        distance2 = np.linalg.norm(new_posmat[2] - new_posmat[1])
        xys.append(
            (
                distance1,
                distance2,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    min_xy = None
    min_energy = 1e24
    for i in xys:
        if i[2] < min_energy:
            min_xy = i
            min_energy = i[2]

    vmax = 10
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"{bead.bond_r} A, {bead.bond_k} kJ/mol/nm2",
        fontsize=16.0,
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c=[i[2] for i in xys],
        vmin=0,
        vmax=vmax,
        alpha=1.0,
        # edgecolor="k",
        s=30,
        cmap="Blues",
    )
    ax.scatter(
        min_xy[0],
        min_xy[1],
        c="r",
        alpha=1.0,
        edgecolor="k",
        s=40,
    )

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][3] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.axhline(y=bead.bond_r, c="k", lw=1)
    ax.axvline(x=bead.bond_r, c="k", lw=1)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance 1 [A]", fontsize=16)
    ax.set_xlabel("distance 2 [A]", fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test3(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{bead.element_string}]([{bead.element_string}])"
            f"[{bead.element_string}]"
        ),
        position_matrix=[[0, 0, 0], [2, 0, 0], [1, 0, 0]],
    )

    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test1; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl3_{temp}",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
        )
        trajectory = opt.run_dynamics(linear_bb)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            vector1 = posmat[1] - posmat[0]
            vector2 = posmat[2] - posmat[0]
            angle = np.degrees(angle_between(vector1, vector2))
            tdict[temp][timestep] = (meas_temp, pot_energy, angle)

    coords = points_in_circum(r=2, n=100)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l3_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[2] = np.array([coord[0], coord[1], 0])
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        energy = opt.calculate_energy(new_bb)
        pos_mat = new_bb.get_position_matrix()
        vector1 = pos_mat[1] - pos_mat[0]
        vector2 = pos_mat[2] - pos_mat[0]
        angle = np.degrees(angle_between(vector1, vector2))
        xys.append(
            (
                angle,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"{bead.bond_r} A, {bead.angle_centered} [deg], {bead.angle_k}"
        " kJ/mol/radian2",
        fontsize=16.0,
    )

    angles = [i[0] for i in xys]
    x = np.linspace(min(angles), max(angles), 100)
    ax.plot(
        x,
        angle_function(
            np.radians(x), bead.angle_k, np.radians(bead.angle_centered)
        ),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    ax.axvline(x=bead.angle_centered, c="k", lw=2)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("angle [theta]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test4(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{bead.element_string}][{bead.element_string}]"
            f"[{bead.element_string}][{bead.element_string}]"
        ),
        position_matrix=[[0, 2, 0], [0, 0, 0], [2, 0, 0], [2, 2, 0]],
    )

    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test1; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl4_{temp}",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=True,
            vdw=False,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
        )
        trajectory = opt.run_dynamics(linear_bb)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            distance = np.linalg.norm(posmat[1] - posmat[0])
            tdict[temp][timestep] = (meas_temp, pot_energy, distance)

    coords = points_in_circum(r=2, n=20)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l4_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[3] = np.array([2, coord[0], coord[1]])
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=True,
            vdw=False,
        )
        energy = opt.calculate_energy(new_bb)
        pos_mat = new_bb.get_position_matrix()
        torsion = get_dihedral(
            pt1=pos_mat[0],
            pt2=pos_mat[1],
            pt3=pos_mat[2],
            pt4=pos_mat[3],
        )
        xys.append(
            (
                torsion,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "n: 1, k=50 [kJ/mol], phi0=180 [deg]",
        fontsize=16.0,
    )

    torsions = [i[0] for i in xys]
    x = np.linspace(min(torsions), max(torsions), 100)
    ax.plot(
        x,
        torsion_function(np.radians(x), 50, np.pi, 1),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )
    ax.axhline(y=0.0, c="k", lw=2, linestyle="--")
    ax.axvline(x=0.0, c="k", lw=2)

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("torsion [theta]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l4.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test5(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=f"[{bead.element_string}][{bead.element_string}]",
        position_matrix=[[0, 0, 0], [1, 0, 0]],
    )

    rvdw = bead.sigma
    coords = np.linspace(rvdw, 10, 50)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l5_{i}"
        new_posmat = linear_bb.get_position_matrix() * coord
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=False,
            angles=False,
            torsions=False,
            vdw=True,
            vdw_bond_cutoff=0,
        )
        energy = opt.calculate_energy(new_bb)
        distance = np.linalg.norm(new_posmat[1] - new_posmat[0])
        xys.append(
            (
                distance,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"sigma: {bead.sigma} A, epsilon: {bead.epsilon} kJ/mol",
        fontsize=16.0,
    )

    distances = [i[0] for i in xys]
    x = np.linspace(min(distances), max(distances), 100)
    ax.plot(
        x,
        nonbond_function(
            x / 10,
            bead.epsilon,
            bead.sigma / 10,
            bead.epsilon,
            bead.sigma / 10,
        ),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )
    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    # ax.axvline(x=rmin, c="gray", lw=2, linestyle="--")
    ax.axvline(x=rvdw, lw=2, linestyle="--", c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance [A]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l5.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = cages() / "ommtest"
    check_directory(struct_output)
    figure_output = figures()
    calculation_output = cages() / "ommtestcalculations"
    check_directory(calculation_output)

    # Define bead libraries.
    beads = c_beads()
    full_bead_library = list(beads.values())
    bead_library_check(full_bead_library)

    random_test(beads, calculation_output, figure_output)
    test1(beads, calculation_output, figure_output)
    test2(beads, calculation_output, figure_output)
    test3(beads, calculation_output, figure_output)
    test4(beads, calculation_output, figure_output)
    test5(beads, calculation_output, figure_output)

    raise SystemExit(
        "want to use this to define the flexibility widths?"
    )


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
