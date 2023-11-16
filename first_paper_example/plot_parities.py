#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Script to plot parities.

Author: Andrew Tarzia

"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from analysis import (
    convert_topo,
    convert_tors,
    data_to_array,
    get_paired_cage_name,
    topology_labels,
)
from env_set import calculations, figures, outputdata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def parity_1(all_data, figure_output):
    logging.info("running parity_1")

    tcmap = topology_labels(short="P")
    tcpos = {tstr: i for i, tstr in enumerate(tcmap) if tstr != "6P8"}

    vdata = all_data[all_data["vdws"] == "von"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for tstr in tcmap:
        if tstr in ("6P8",):
            continue

        tdata = vdata[vdata["topology"] == tstr]
        data1 = tdata[tdata["torsions"] == "ton"]

        diffdata = []
        for _idx, row in data1.iterrows():
            d1_energy = float(row["energy_per_bb"])
            cage_name = str(row["cage_name"])
            pair_name = get_paired_cage_name(cage_name)
            d2_row = tdata[tdata["cage_name"] == pair_name]
            d2_energy = float(d2_row["energy_per_bb"].iloc[0])

            diffdata.append(d1_energy - d2_energy)

        xpos = tcpos[tstr]

        parts = ax.violinplot(
            list(diffdata),
            [xpos],
            vert=True,
            widths=0.8,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("#086788")
            pc.set_edgecolor("none")
            pc.set_alpha(1.0)

    ax.plot((-1, 13), (0, 0), c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(
        "restricted - not restricted [kJmol$^{-1}$]",
        fontsize=16,
    )
    ax.set_xlim(-0.5, 11.5)
    ax.set_xticks([tcpos[i] for i in tcpos])
    ax.set_xticklabels(
        [convert_topo(i) for i in tcpos],
        rotation=45,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "par_1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def parity_2(all_data, geom_data, figure_output):
    logging.info("running parity_2")

    c2labels = ("Pb_Ba_Ag",)
    cmap = {"ton": "#086788", "toff": "#F9A03F"}
    topologies = [i for i in topology_labels(short="P") if i != "6P8"]

    fig, axs = plt.subplots(
        ncols=4,
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    for ax, tstr in zip(flat_axs, topologies, strict=True):
        topo_frame = all_data[all_data["topology"] == tstr]
        # for tors, ax in zip(("ton", "toff"), flat_axs, strict=True):
        for tors in ("ton", "toff"):
            tor_frame = topo_frame[topo_frame["torsions"] == tors]

            target_c2s = []
            measured_c2 = []
            energies = []
            for i, row in tor_frame.iterrows():
                target_bite_angle = float(row["target_bite_angle"])
                target_angle = (target_bite_angle / 2) + 90
                name = str(row["index"])
                energy = float(row["energy_per_bb"])
                c2data = geom_data[name]["angles"]

                for lbl in c2labels:
                    if lbl in c2data:
                        lbldata = c2data[lbl]
                        for mc2 in lbldata:
                            measured_c2.append(mc2)
                            target_c2s.append(target_angle)
                            energies.append(energy)

            comp_values = {i: [] for i in sorted(set(target_c2s))}
            for i, j in zip(target_c2s, measured_c2, strict=True):
                comp_values[i].append(j)

            ax.plot(
                list(comp_values),
                [np.mean(comp_values[i]) for i in comp_values],
                alpha=1.0,
                marker="o",
                markerfacecolor=cmap[tors],
                markeredgecolor="k",
                linewidth=1,
                color=cmap[tors],
                markersize=6,
                label=convert_tors(tors, num=False),
            )
            ax.fill_between(
                list(comp_values),
                y1=[np.min(comp_values[i]) for i in comp_values],
                y2=[np.max(comp_values[i]) for i in comp_values],
                alpha=0.6,
                color=cmap[tors],
                edgecolor=(0, 0, 0, 2.0),
                lw=0,
            )

        ax.plot(
            (0, 200),
            (0, 200),
            c="k",
            lw=1,
            linestyle="--",
            alpha=0.5,
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(r"target [$^\circ$]", fontsize=16)
        ax.set_ylabel(r"observed [$^\circ$]", fontsize=16)
        ax.set_title(
            (f"{convert_topo(tstr)}"),
            fontsize=16,
        )
        ax.set_xlim(90, 180)
        ax.set_ylim(70, 190)
        if tstr == "2P3":
            ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "par_2.pdf"),
        dpi=320,
        bbox_inches="tight",
    )
    plt.close()


def pore_b2b_distance(all_data, figure_output):
    logging.info("running pore_b2b parity")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        all_data["pore"],
        all_data["min_b2b_distance"],
        c="gray",
        alpha=0.2,
        s=30,
        rasterized=True,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("pore [A]", fontsize=16)
    ax.set_ylabel("min_b2b [A]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "pore_vs_minb2b.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main():
    first_line = f"Usage: {__file__}.py"
    if len(sys.argv) != 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    figure_output = figures()
    calculation_output = calculations()
    data_output = outputdata()

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=data_output,
    )
    with open(data_output / "all_geom.json") as f:
        geom_data = json.load(f)
    logging.info(f"there are {len(all_data)} collected data")

    parity_1(all_data, figure_output)
    parity_2(all_data, geom_data, figure_output)
    pore_b2b_distance(all_data, figure_output)


if __name__ == "__main__":
    main()
