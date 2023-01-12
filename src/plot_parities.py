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
import itertools
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from env_set import cages

from analysis_utilities import (
    topo_to_colormap,
    write_out_mapping,
    data_to_array,
    convert_topo_to_label,
    convert_torsion_to_label,
    max_energy,
)


def parity_1(all_data, figure_output):
    tcmap = topo_to_colormap()
    tcpos = {tstr: i for i, tstr in enumerate(tcmap)}
    lentcpos = len(tcpos)
    tcmap.update({"all": "k"})
    tcpos.update({"all": lentcpos})

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(8, 8),
    )
    # flat_axs = axs.flatten()

    ax = axs[0]
    ax2 = axs[1]
    for tstr in tcmap:
        if tstr == "all":
            ton_data = all_data[all_data["torsions"] == "ton"]
            toff_data = all_data[all_data["torsions"] == "toff"]
        else:
            tdata = all_data[all_data["topology"] == tstr]
            ton_data = tdata[tdata["torsions"] == "ton"]
            toff_data = tdata[tdata["torsions"] == "toff"]

        out_data = ton_data.merge(
            toff_data,
            on="cage_name",
        )

        ydata = list(out_data["energy_x"])
        xdata = list(out_data["energy_y"])
        diffdata = [y - x for x, y, in zip(xdata, ydata)]
        xpos = tcpos[tstr]

        ax.scatter(
            [xpos for i in diffdata],
            [i for i in diffdata],
            c="gray",
            edgecolor="none",
            s=30,
            alpha=1.0,
            rasterized=True,
        )

        ax2.scatter(
            [xpos for i in diffdata if abs(i) < 20],
            [i for i in diffdata if abs(i) < 20],
            c="gray",
            edgecolor="none",
            s=30,
            alpha=0.2,
            rasterized=True,
        )

        parts = ax2.violinplot(
            [i for i in diffdata if abs(i) < 20],
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
            pc.set_facecolor("gray")
            pc.set_edgecolor("none")
            pc.set_alpha(0.3)

    ax.plot((-1, 12), (0, 0), c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("ton - toff [eV]", fontsize=16)
    ax.set_xlim(-0.5, 11.5)
    # ax.set_xticks([tcpos[i] for i in tcpos])
    # ax.set_xticklabels([convert_topo_to_label(i) for i in tcpos])

    ax2.plot((-1, 12), (0, 0), c="k")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_ylabel("ton - toff [eV]", fontsize=16)
    ax2.set_xlim(-0.5, 11.5)
    ax2.set_ylim(-22, 22)
    ax2.set_xticks([tcpos[i] for i in tcpos])
    ax2.set_xticklabels(
        [convert_topo_to_label(i) for i in tcpos],
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
    tcmap = topo_to_colormap()

    c2labels = ("Pb_Ba_Ag",)
    vmax = max_energy()
    for tstr in tcmap:
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
                energy = float(row["energy_per_bond"])
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
                s=50,
                cmap="Blues",
                rasterized=True,
            )

            ax.plot((90, 180), (90, 180), c="k", lw=2, linestyle="--")

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("target [deg]", fontsize=16)
            ax.set_ylabel("observed [deg]", fontsize=16)
            ax.set_title(
                (
                    f"{convert_topo_to_label(tstr)}: "
                    f"{convert_torsion_to_label(tors, num=False)}"
                ),
                fontsize=16,
            )

        cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        cmap = mpl.cm.Blues
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("energy per bond formed [eV]", fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"par_2_{tstr}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def parity_3(all_data, geom_data, figure_output):
    tcmap = topo_to_colormap()

    c2labels = ("Pb_Ba_Ag",)

    for tstr in tcmap:
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
                target_angle = float(row["c2angle"])
                name = str(row["index"])
                energy = float(row["energy_per_bond"])
                c2data = geom_data[name]["angles"]

                for lbl in c2labels:
                    if lbl in c2data:
                        lbldata = c2data[lbl]
                        for mc2 in lbldata:
                            measured_c2.append(mc2)
                            target_c2s.append(target_angle)
                            energies.append(energy)
            ax.scatter(
                [j - i for i, j in zip(measured_c2, target_c2s)],
                energies,
                c="gray",
                alpha=1.0,
                edgecolor="none",
                rasterized=True,
                s=30,
            )

            ax.axhline(y=0, c="k", lw=2, linestyle="--")

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("target - observed [deg]", fontsize=16)
            ax.set_ylabel("energy per bond formed [eV]", fontsize=16)
            ax.set_title(
                (
                    f"{convert_topo_to_label(tstr)}: "
                    f"{convert_torsion_to_label(tors, num=False)}"
                ),
                fontsize=16,
            )

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"par_3_{tstr}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def size_parities(all_data, figure_output):
    print(all_data.columns)
    properties = {
        "energy": (0, 1),
        "gnorm": (0, 0.001),
        "pore": (0, 20),
        "min_b2b": (0, 1),
        "sv_n_dist": (0, 1),
        "sv_l_dist": (0, 1),
    }
    comps = {
        (2, 5): "k",
        (2, 10): "skyblue",
        (5, 10): "gold",
    }
    comp_cols = ("clsigma", "c2sigma")

    topologies = sorted(set(all_data["topology"]))
    clangles = sorted(set(all_data["clangle"]))

    for tstr, clangle, prop in itertools.product(
        topologies,
        clangles,
        properties,
    ):

        print(tstr, clangle, prop)
        tdata = all_data[all_data["topology"] == tstr]
        cdata = tdata[tdata["clangle"] == clangle]
        if len(cdata) == 0:
            continue

        c2_values = set(cdata["c2angle"])

        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            sharey=True,
            sharex=True,
            figsize=(16, 8),
        )

        for axpair, ccol in zip(axs, comp_cols):
            not_ccol = [i for i in comp_cols if i != ccol][0]
            for ax, tors in zip(axpair, ("ton", "toff")):
                x_positions = range(9)
                xlabels = []
                for i, (pair, not_val) in enumerate(
                    itertools.product(
                        comps,
                        sorted(set(cdata[not_ccol])),
                    )
                ):

                    print(tors, pair, not_val)
                    # ax = flat_axs[i]
                    xpos = x_positions[i]
                    tor_data = cdata[cdata["torsions"] == tors]
                    notdata = tor_data[tor_data[not_ccol] == not_val]
                    frame1 = notdata[notdata[ccol] == pair[0]]
                    frame2 = notdata[notdata[ccol] == pair[1]]

                    # xs = []
                    ys = []
                    for c2val in c2_values:
                        plot1 = frame1[frame1["c2angle"] == c2val]
                        plot2 = frame2[frame2["c2angle"] == c2val]
                        prop_value1 = float(plot1[prop])
                        prop_value2 = float(plot2[prop])
                        if pd.isna(prop_value1):
                            continue
                        if pd.isna(prop_value2):
                            continue
                        # xs.append(prop_value1)
                        ys.append(prop_value2 - prop_value1)

                    ax.scatter(
                        [xpos for i in ys],
                        ys,
                        c=comps[pair],
                        edgecolor="k",
                        s=120,
                        alpha=1.0,
                        label=f"{pair}",
                    )
                    # ax.scatter(
                    #     xs,
                    #     ys,
                    #     c=[size_difference for i in xs],
                    #     vmin=0,
                    #     vmax=10,
                    #     edgecolor="k",
                    #     s=70,
                    #     alpha=1.0,
                    # )
                    ax.axhline(y=0, c="k", linestyle="--", lw=2)
                    title = f"{tors}: changing {ccol}"
                    ax.set_title(title, fontsize=16)
                    ax.tick_params(
                        axis="both", which="major", labelsize=16
                    )
                    ax.set_xlabel(not_ccol, fontsize=16)
                    ax.set_ylabel(f"{prop}: large - small", fontsize=16)
                    xlbl = f"{not_val}"
                    xlabels.append(xlbl)

            ax.set_xticks([i for i in x_positions])
            ax.set_xticklabels([i for i in xlabels])

        fig.tight_layout()
        filename = f"sp_{tstr}_{clangle}_{prop}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
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

    figure_output = cages() / "figures"
    calculation_output = cages() / "calculations"

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=calculation_output,
    )
    with open(calculation_output / "all_geom.json", "r") as f:
        geom_data = json.load(f)
    logging.info(f"there are {len(all_data)} collected data")
    opt_data = all_data[all_data["optimised"]]
    logging.info(f"there are {len(opt_data)} successfully opted")
    write_out_mapping(opt_data)

    size_parities(all_data, figure_output)
    parity_1(opt_data, figure_output)
    parity_2(opt_data, geom_data, figure_output)
    parity_3(opt_data, geom_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
