#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

Author: Andrew Tarzia

"""

import sys
import numpy as np
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

    num_atoms = sorted(
        set(
            [gulp_timings[i][0] for i in gulp_timings]
            # + [omm_timings[i][0] for i in omm_timings]
        )
    )
    print(num_atoms)
    time_means = {
        "gulp": [0 for i in num_atoms],
        # "gulp-CG": [0 for i in num_atoms],
        "OpenMM": [0 for i in num_atoms],
        "OpenMM/vdw": [0 for i in num_atoms],
        "OpenMM-MD/vdw": [0 for i in num_atoms],
    }
    time_stds = {
        "gulp": [0 for i in num_atoms],
        # "gulp-CG": [0 for i in num_atoms],
        "OpenMM": [0 for i in num_atoms],
        "OpenMM/vdw": [0 for i in num_atoms],
        "OpenMM-MD/vdw": [0 for i in num_atoms],
    }

    for i, na in enumerate(num_atoms):
        gulp_trim = {
            i: gulp_timings[i]
            for i in gulp_timings
            if gulp_timings[i][0] == na
        }
        g_all_timings = [gulp_trim[i][1] for i in gulp_trim]
        # g_cg_timings = [gulp_trim[i][1] for i in gulp_trim if "o1" in i]

        omm_trim = {
            i: omm_timings[i]
            for i in omm_timings
            if omm_timings[i][0] == na
        }
        o_voff_timings = [
            omm_trim[i][1]
            for i in omm_trim
            if "o2" not in i
            if "voff" in i
        ]
        o_von_timings = [
            omm_trim[i][1]
            for i in omm_trim
            if "o2" not in i
            if "von" in i
        ]
        o_md_timings = [
            omm_trim[i][1] for i in omm_trim if "o2" in i and "von" in i
        ]

        for idx in omm_trim:
            if omm_trim[idx][1] > 10:
                print(idx, omm_trim[idx][1])

        time_means["gulp"][i] = np.mean(g_all_timings)
        time_means["OpenMM"][i] = np.mean(o_voff_timings)
        time_means["OpenMM/vdw"][i] = np.mean(o_von_timings)
        time_means["OpenMM-MD/vdw"][i] = np.mean(o_md_timings)

        time_stds["gulp"][i] = np.std(g_all_timings)
        time_stds["OpenMM"][i] = np.std(o_voff_timings)
        time_stds["OpenMM/vdw"][i] = np.std(o_von_timings)
        time_stds["OpenMM-MD/vdw"][i] = np.std(o_md_timings)
    print(time_means)
    print(time_stds)

    # x = np.arange(len(num_atoms))
    # width = 0.3
    # multiplier = 0

    fig, ax = plt.subplots(figsize=(8, 5))

    gulp_means = time_means["gulp"]

    for run_type in time_means:
        if run_type == "gulp":
            continue
        means = time_means[run_type]
        relative_means = [i / j for i, j in zip(means, gulp_means)]
        # stds = time_stds[run_type]

        # offset = width * multiplier
        # ax.bar(
        #     x + offset,
        #     relative_means,
        #     width,
        #     # yerr=stds,
        #     label=run_type,
        # )
        ax.plot(
            num_atoms,
            relative_means,
            # width,
            # yerr=stds,
            marker="o",
            markersize=8,
            lw=2,
            label=run_type,
        )
        # ax.bar_label(rects, padding=3)
        # multiplier += 1

    ax.axhline(y=1, c="k", linestyle="--")
    # for i in x:
    #     ax.axvline(
    #         x=i + 1.0 - (width / 2) - 0.05, c="gray", linestyle="--"
    #     )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num. beads", fontsize=16)
    # ax.set_ylabel("CPU time [s]", fontsize=16)
    ax.set_ylabel("mean time/mean gulp time [s]", fontsize=16)
    ax.set_yscale("log")
    # ax.set_xticks(x)
    # ax.set_xticklabels(num_atoms)

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
