#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to run test of flexibility measures.

Author: Andrew Tarzia

"""

import sys
import stk
import os
from openmm import openmm
import matplotlib.pyplot as plt
import numpy as np
import logging
from rdkit import RDLogger

from env_set import cages
from utilities import check_directory
from geom import GeomMeasure
from openmm_optimizer import CGOMMDynamics, CGOMMOptimizer
from beads import produce_bead_library, bead_library_check


def c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        bond_rs=(3,),
        angles=(120,),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def measure_flexibility(beads, calculation_output, figure_output):

    tcol = {
        300: "#086788",
        100: "#F9A03F",
        10: "#0B2027",
    }

    bead = beads["c0000"]
    bbs = {
        "2l": {
            "num_atoms": 2,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}][{bead.element_string}]"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                ],
            ),
        },
        "3l": {
            "num_atoms": 3,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}]"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                ],
            ),
        },
        "4l": {
            "num_atoms": 4,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 6, 0],
                ],
            ),
        },
        "5l": {
            "num_atoms": 5,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}]"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 6, 0],
                    [0, 8, 0],
                ],
            ),
        },
        "6l": {
            "num_atoms": 6,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 6, 0],
                    [0, 8, 0],
                    [0, 10, 0],
                ],
            ),
        },
        "7l": {
            "num_atoms": 7,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}]"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 6, 0],
                    [0, 8, 0],
                    [0, 10, 0],
                    [0, 12, 0],
                ],
            ),
        },
        "6r": {
            "num_atoms": 6,
            "bb": stk.BuildingBlock(
                smiles=(
                    f"[{bead.element_string}]1[{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]"
                    f"[{bead.element_string}][{bead.element_string}]1"
                ),
                position_matrix=[
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 6, 0],
                    [2, 2, 0],
                    [2, 0, 0],
                ],
            ),
        },
    }

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        for system in bbs:
            tdict[temp][system] = {}
            num_atoms = bbs[system]["num_atoms"]
            bb = bbs[system]["bb"]

            opt = CGOMMOptimizer(
                fileprefix=f"flexopt_{system}_{temp}",
                output_dir=calculation_output,
                param_pool=beads,
                custom_torsion_set=None,
                bonds=True,
                angles=True,
                torsions=False,
                vdw=True,
                # max_iterations=1000,
                vdw_bond_cutoff=2,
            )
            linear_bb = opt.optimize(bb)

            logging.info(f"running MD of {system} length; {temp}")
            opt = CGOMMDynamics(
                fileprefix=f"flex_{system}_{temp}",
                output_dir=calculation_output,
                param_pool=beads,
                custom_torsion_set=None,
                bonds=True,
                angles=True,
                torsions=False,
                vdw=True,
                temperature=temp,
                random_seed=1000,
                num_steps=10000,
                time_step=1 * openmm.unit.femtoseconds,
                friction=1.0 / openmm.unit.picosecond,
                reporting_freq=100,
                traj_freq=100,
                vdw_bond_cutoff=2,
            )
            trajectory = opt.run_dynamics(linear_bb)

            # traj_log = trajectory.get_data()
            rgs = []
            for conformer in trajectory.yield_conformers():
                # timestep = conformer.timestep
                # row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
                # energies.append(
                #     float(row["Potential Energy (kJ/mole)"])
                # )
                g_measure = GeomMeasure()
                rgs.append(
                    g_measure.calculate_radius_gyration(
                        conformer.molecule
                    )
                )
            flexibility = np.std(rgs)
            tdict[temp][system] = (num_atoms, flexibility)

    fig, ax = plt.subplots(figsize=(8, 5))

    for temp in tdict:
        data = tdict[temp]
        ax.plot(
            [data[i][0] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            lw=3,
            markersize=9,
            marker="o",
            # edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("number of atoms", fontsize=16)
    ax.set_ylabel("flexibility", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "flexibility_measure.pdf"),
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

    measure_flexibility(beads, calculation_output, figure_output)


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
