#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot parities.

Author: Andrew Tarzia

"""

import sys
import os
import json
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from env_set import cages

from analysis_utilities import (
    topology_labels,
    eb_str,
    write_out_mapping,
    get_lowest_energy_data,
    data_to_array,
    convert_topo,
    convert_tors,
)


def parity_1(all_data, figure_output):
    logging.info("running parity_1")

    tcmap = topology_labels(short="P")
    tcpos = {tstr: i for i, tstr in enumerate(tcmap) if tstr != "6P8"}

    vdata = all_data[all_data["vdws"] == "von"]

    fig, ax = plt.subplots(
        # nrows=1,
        # ncols=3,
        # sharex=True,
        # sharey=True,
        figsize=(8, 5),
    )

    # for ax, tors, vdws) in zip(
    #     flat_axs, (("ton", "voff"), ("toff", "von"), ("toff", "voff"))
    # ):
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

    ax.plot((-1, 12), (0, 0), c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(f"ton - toff [kJmol$^{-1}$]", fontsize=16)
    ax.set_xlim(-0.5, 10.5)
    # ax.set_ylim(-22, 22)
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
    tcmap = topology_labels(short="P")

    c2labels = ("Pb_Ba_Ag",)
    vmax = 10
    for tstr in tcmap:
        if tstr in ("6P8",):
            continue
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=(12, 5),
        )
        flat_axs = axs.flatten()
        topo_frame = all_data[all_data["topology"] == tstr]
        for tors, ax in zip(("ton", "toff"), flat_axs):
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
            ax.scatter(
                target_c2s,
                measured_c2,
                c=energies,
                vmin=0,
                vmax=vmax,
                alpha=1.0,
                edgecolor="k",
                linewidth=1,
                s=50,
                cmap="Blues_r",
                rasterized=True,
            )

            ax.plot((0, 200), (0, 200), c="k", lw=2, linestyle="--")

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("target [deg]", fontsize=16)
            ax.set_ylabel("observed [deg]", fontsize=16)
            ax.set_title(
                (
                    f"{convert_topo(tstr)}: "
                    f"{convert_tors(tors, num=False)}"
                ),
                fontsize=16,
            )
            ax.set_xlim(70, 190)
            ax.set_ylim(70, 190)

        cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        cmap = mpl.cm.Blues_r
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(eb_str(), fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"par_2_{tstr}.pdf"),
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
        # edgecolor="k",
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
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    figure_output = cages() / "ommfigures"
    calculation_output = cages() / "ommcalculations"

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=calculation_output,
    )
    with open(calculation_output / "all_geom.json", "r") as f:
        geom_data = json.load(f)
    logging.info(f"there are {len(all_data)} collected data")
    low_e_data = get_lowest_energy_data(
        all_data=all_data,
        output_dir=calculation_output,
    )
    write_out_mapping(all_data)

    pore_b2b_distance(low_e_data, figure_output)
    parity_1(low_e_data, figure_output)
    parity_2(low_e_data, geom_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
