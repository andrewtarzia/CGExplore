#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot parities.

Author: Andrew Tarzia

"""

import logging
import os
import sys

import matplotlib.pyplot as plt
from analysis import (
    convert_topo,
    data_to_array,
    eb_str,
    get_lowest_energy_data,
    isomer_energy,
    topology_labels,
)
from env_set import cages, calculations, figures, outputdata
from matplotlib.lines import Line2D

from cgexplore.utilities import check_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def energy_parity(all_data, dupl_data, figure_output):
    logging.info("running energy_parity")
    print(len(all_data), len(dupl_data))

    cmap = {
        "4P6": "#086788",
        "4P62": "#F9A03F",
        "8P12": "#CA1551",
        # "#CA1551"
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, row in dupl_data.iterrows():
        index_name = str(row["index"])
        tstr = str(row["topology"])
        orig_row = all_data[all_data["index"] == index_name]
        dupl_energy = float(row["energy_per_bb"])
        orig_energy = float(orig_row["energy_per_bb"])
        if dupl_energy < 0.01 and orig_energy < 0.01:
            continue
        if abs(dupl_energy - orig_energy) > 0.01:
            print(index_name, dupl_energy, orig_energy)
            ax.scatter(
                orig_energy,
                dupl_energy,
                c=cmap[tstr],
                edgecolor="white",
                s=60,
                alpha=0.8,
            )
        else:
            ax.scatter(
                orig_energy,
                dupl_energy,
                c="gray",
                edgecolor="none",
                s=20,
                alpha=0.1,
            )

    # ax.plot((0, 20), (0, 20), c="k", ls="--", lw=2, alpha=0.2)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"first run {eb_str()}", fontsize=16)
    ax.set_ylabel(f"second run {eb_str()}", fontsize=16)
    ax.set_xlim(0.01, 20)
    ax.set_ylim(0.01, 20)
    ax.axhline(y=isomer_energy(), c="k", ls="--", lw=2, alpha=0.2)
    ax.axvline(x=isomer_energy(), c="k", ls="--", lw=2, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")

    legend_elements = []
    for tstr in cmap:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=convert_topo(tstr),
                markerfacecolor=cmap[tstr],
                markersize=7,
                markeredgecolor="white",
                alpha=0.8,
            )
        )
    ax.legend(handles=legend_elements, fontsize=16, ncol=1)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "energy_parity.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    raise SystemExit()

    tcmap = topology_labels(short="P")
    tcpos = {tstr: i for i, tstr in enumerate(tcmap) if tstr != "6P8"}

    vdata = all_data[all_data["vdws"] == "von"]

    for tstr in tcmap:
        if tstr in ("6P8",):
            continue

        tdata = vdata[vdata["topology"] == tstr]
        data1 = tdata[tdata["torsions"] == "ton"]
        data2 = tdata[tdata["torsions"] == "toff"]

        out_data = data1.merge(
            data2,
            on="cage_name",
        )

        ydata = list(out_data["energy_per_bb_x"])
        xdata = list(out_data["energy_per_bb_y"])
        diffdata = [y - x for x, y, in zip(xdata, ydata)]
        xpos = tcpos[tstr]

        parts = ax.violinplot(
            [i for i in diffdata],
            [xpos],
            # points=200,
            vert=True,
            widths=0.8,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            # bw_method=0.5,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("#086788")
            pc.set_edgecolor("none")
            pc.set_alpha(1.0)


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    figure_output = figures()
    calculation_output = calculations()
    data_output = outputdata()
    dupl_calculation_output = cages() / "duplicate_calculations"
    dupl_data_output = cages() / "duplicate_outputdata"
    check_directory(dupl_data_output)

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=data_output,
    )
    low_e_data = get_lowest_energy_data(
        all_data=all_data,
        output_dir=data_output,
    )
    logging.info(f"there are {len(all_data)} collected data")

    dupl_all_data = data_to_array(
        json_files=dupl_calculation_output.glob("*_res.json"),
        output_dir=dupl_data_output,
    )
    dupl_low_e_data = get_lowest_energy_data(
        all_data=dupl_all_data,
        output_dir=dupl_data_output,
    )
    logging.info(f"there are {len(dupl_all_data)} collected data")

    energy_parity(low_e_data, dupl_low_e_data, figure_output)


if __name__ == "__main__":
    main()
