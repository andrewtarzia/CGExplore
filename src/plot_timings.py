#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

Author: Andrew Tarzia

"""

import sys
import os
import json
import matplotlib.pyplot as plt
import logging

from env_set import cages


def openmm_timings():
    calculation_output = cages() / "ommcalculations"

    timings_file = calculation_output / "omm_timings.json"

    if os.path.exists(timings_file):
        with open(timings_file, "r") as f:
            output = json.load(f)
        return output

    out_files = calculation_output.glob("*_omm.out")
    output = {}
    for of in out_files:
        with open(of, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "atoms:" in line:
                num_atoms = int(line.strip().split()[-1])
            if "total time:" in line:
                cpu_time = float(line.strip().split()[-2])

        output[of.name] = (num_atoms, cpu_time)

    with open(timings_file, "w") as f:
        json.dump(output, f, indent=4)

    return output


def gulp_timings():

    gulp_output = cages() / "calculations"

    timings_file = gulp_output / "timings.json"

    if os.path.exists(timings_file):
        with open(timings_file, "r") as f:
            output = json.load(f)
        return output

    ginout_files = gulp_output.glob("*.ginout")
    output = {}
    for gf in ginout_files:
        with open(gf, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "Total number atoms/shells" in line:
                num_atoms = int(line.strip().split()[-1])
            if "Total CPU time" in line:
                cpu_time = float(line.strip().split()[-1])

        output[gf.name] = (num_atoms, cpu_time)

    with open(timings_file, "w") as f:
        json.dump(output, f, indent=4)

    return output


def plot_timings(gulp_timings, omm_timings):
    figure_output = cages() / "ommfigures"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        [gulp_timings[i][0] for i in gulp_timings],  # if "toff" in i],
        [gulp_timings[i][1] for i in gulp_timings],  # if "toff" in i],
        c="gray",
        s=30,
        edgecolor="none",
        alpha=0.5,
        label="GULP",
        rasterized=True,
    )
    ax.scatter(
        [gulp_timings[i][0] for i in gulp_timings if "o1" in i],
        [gulp_timings[i][1] for i in gulp_timings if "o1" in i],
        c="r",
        s=30,
        edgecolor="none",
        alpha=0.5,
        label="GULP-CG=True",
        rasterized=True,
    )
    ax.scatter(
        [omm_timings[i][0] for i in omm_timings if "voff" in i],
        [omm_timings[i][1] for i in omm_timings if "voff" in i],
        c="skyblue",
        s=20,
        edgecolor="none",
        alpha=0.6,
        label="OpenMM",
        rasterized=True,
    )
    ax.scatter(
        [omm_timings[i][0] for i in omm_timings if "von" in i],
        [omm_timings[i][1] for i in omm_timings if "von" in i],
        c="gold",
        s=20,
        edgecolor="none",
        alpha=0.6,
        label="OpenMM/w-dispersion",
        rasterized=True,
    )
    ax.scatter(
        [omm_timings[i][0] for i in omm_timings if "o2" in i],
        [omm_timings[i][1] for i in omm_timings if "o2" in i],
        c="green",
        s=20,
        edgecolor="none",
        alpha=0.6,
        label="OpenMM-MD",
        rasterized=True,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num. atoms", fontsize=16)
    ax.set_ylabel("CPU time [s]", fontsize=16)
    ax.set_yscale("log")
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "timings.pdf"),
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

    gts = gulp_timings()
    oms = openmm_timings()

    plot_timings(gts, oms)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
