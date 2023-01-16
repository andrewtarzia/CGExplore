#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

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

from env_set import cages
from utilities import check_directory, angle_between, get_dihedral
from openmm_optimizer import CGOMMOptimizer
from beads import produce_bead_library, bead_library_check


def c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        sigmas=(2,),
        angles=(90,),
        bond_ks=(10,),
        angle_ks=(20,),
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


def test1(beads, calculation_output, figure_output):

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=f"[{bead.element_string}][{bead.element_string}]",
        position_matrix=[[0, 0, 0], [1, 0, 0]],
    )

    coords = np.linspace(0, 5, 20)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l1_{i}"
        new_posmat = linear_bb.get_position_matrix() * coord
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
        f"{bead.sigma} A, {bead.bond_k} kJ/mol/nm2",
        fontsize=16.0,
    )

    def fun(x, k, sigma):
        return (1 / 2) * k * (x - sigma / 1) ** 2

    distances = [i[0] for i in xys]
    x = np.linspace(min(distances), max(distances), 100)
    ax.plot(
        x,
        fun(x / 10, bead.bond_k, bead.sigma / 10),
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

    coords1 = np.linspace(0, 5, 25)
    coords2 = np.linspace(0, 5, 25)
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
        f"{bead.sigma} A, {bead.bond_k} kJ/mol/nm2",
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
        rasterized=True,
    )
    ax.scatter(
        min_xy[0],
        min_xy[1],
        c="r",
        alpha=1.0,
        edgecolor="k",
        s=40,
    )
    ax.axhline(y=bead.sigma, c="k", lw=1)
    ax.axvline(x=bead.sigma, c="k", lw=1)
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
        f"{bead.sigma} A, {bead.angle_centered} [deg], {bead.angle_k}"
        " kJ/mol/radian2",
        fontsize=16.0,
    )

    def fun(x, k, theta0):
        return (1 / 2) * k * (x - theta0) ** 2

    angles = [i[0] for i in xys]
    x = np.linspace(min(angles), max(angles), 100)
    ax.plot(
        x,
        fun(
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
        position_matrix=[[0, 2, 0], [0, 0, 0], [2, 0, 0], [2, 0, 0]],
    )

    coords = points_in_circum(r=2, n=20)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l4_{i}"
        print(coord)
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
        "n: 1, k=-5 [kJ/mol], phi0=0 [deg]",
        fontsize=16.0,
    )

    def fun(x, k, theta0, n):
        return k * (1 + np.cos(n * x - theta0))

    torsions = [i[0] for i in xys]
    x = np.linspace(min(torsions), max(torsions), 100)
    ax.plot(
        x,
        fun(np.radians(x), -5, 0, 1),
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

    rmin = bead.sigma * (2 ** (1 / 6))
    rvdw = rmin / 2
    print(rmin, rvdw)
    coords = np.linspace(rvdw + 0.8, 15, 50)
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
        f"{bead.sigma} A, epsilon: {bead.bond_k} kJ/mol",
        fontsize=16.0,
    )

    def fun(x, epsilon1, sigma1, epsilon2, sigma2):
        return (
            np.sqrt(epsilon1 * epsilon2)
            * ((sigma1 + sigma2) / (2 * x)) ** 12
        )

    distances = [i[0] for i in xys]
    x = np.linspace(min(distances), max(distances), 100)
    ax.plot(
        x,
        fun(
            x / 10,
            bead.bond_k,
            bead.sigma / 10,
            bead.bond_k,
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
    ax.axvline(x=rmin, c="k", lw=2, linestyle="--")
    ax.axvline(x=rvdw, lw=2, linestyle="--", c="r")
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
    figure_output = cages() / "ommfigures"
    check_directory(figure_output)
    calculation_output = cages() / "ommtestcalculations"
    check_directory(calculation_output)

    # Define bead libraries.
    beads = c_beads()
    full_bead_library = list(beads.values())
    bead_library_check(full_bead_library)

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
