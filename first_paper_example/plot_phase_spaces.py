#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot phase spaces.

Author: Andrew Tarzia

"""

import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import (
    angle_str,
    convert_prop,
    convert_topo,
    convert_tors,
    data_to_array,
    get_lowest_energy_data,
    isomer_energy,
    map_cltype_to_topology,
    pore_str,
    rg_str,
    stoich_map,
    topology_labels,
)
from env_set import calculations, figures, outputdata
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def phase_space_2(all_data, figure_output):
    logging.info("doing phase space 2")

    vdata = all_data[all_data["vdws"] == "von"]

    bbs = set(vdata["bbpair"])
    bb_data = {}
    for bb_pair in bbs:
        bbd = vdata[vdata["bbpair"] == bb_pair]

        for tor in ("ton", "toff"):
            data = bbd[bbd["torsions"] == tor]
            stable_energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in data.iterrows()
                if float(row["energy_per_bb"]) < isomer_energy()
            }
            if len(stable_energies) == 0:
                continue
            stoichiometries = {i: stoich_map(i) for i in stable_energies}
            min_stoichiometry = min(stoichiometries.values())
            kinetic_energies = {
                i: stable_energies[i]
                for i in stoichiometries
                if stoichiometries[i] == min_stoichiometry
            }

            if len(kinetic_energies) == 1:
                # Self-sorted.
                tstr = list(kinetic_energies.keys())[0]
                epb = list(kinetic_energies.values())[0]
                kinetic_data = data[data["topology"] == tstr]

                if tstr == "6P8":
                    colour_choice = "k"
                elif "3C1" == str(bbd["cltitle"].iloc[0]):
                    colour_choice = "#086788"
                elif "4C1" == str(bbd["cltitle"].iloc[0]):
                    colour_choice = "#F9A03F"

                if tor == "ton":
                    tor_colour = "#0B2027"
                elif tor == "toff":
                    tor_colour = "#CA1551"

                node_measure = float(kinetic_data["sv_n_dist"].iloc[0])
                ligand_measure = float(kinetic_data["sv_n_dist"].iloc[0])
                if node_measure is None:
                    shape_measure = ligand_measure
                elif ligand_measure is None:
                    shape_measure = node_measure
                else:
                    shape_measure = min((node_measure, ligand_measure))
                bb_data[(bb_pair, tor)] = {
                    "energy_per_bb": epb,
                    "topology": tstr,
                    "pore": float(kinetic_data["pore"].iloc[0]),
                    "radius_gyration": float(
                        kinetic_data["radius_gyration"].iloc[0]
                    ),
                    "shape_measure": shape_measure,
                    "cltitle": str(kinetic_data["cltitle"].iloc[0]),
                    "clangle": float(kinetic_data["clangle"].iloc[0]),
                    "c2angle": float(kinetic_data["c2angle"].iloc[0]),
                    "c3angle": float(kinetic_data["c3angle"].iloc[0]),
                    "cmap": colour_choice,
                    "torsionmap": tor_colour,
                }

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    def cltitleconversion(clstr):
        return {"4C1": 4, "3C1": 3}[clstr]

    def no_conversion(value):
        return value

    axmap = (
        {
            "ax": flat_axs[0],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "topology",
            "xlbl": "num. building blocks",
            "xmapfun": stoich_map,
            "xlim": (None, None),
            "c": "cmap",
        },
        {
            "ax": flat_axs[1],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "cltitle",
            "xlbl": "largest coordination",
            "xmapfun": cltitleconversion,
            "xlim": (2.5, 4.5),
            "c": "cmap",
        },
        {
            "ax": flat_axs[2],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "clangle",
            "xlbl": angle_str(),
            "xmapfun": no_conversion,
            "xlim": (48, 121),
            "c": "cmap",
        },
        {
            "ax": flat_axs[5],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "c2angle",
            "xlbl": angle_str(2),
            "xmapfun": no_conversion,
            "xlim": (89, 181),
            "c": "cmap",
        },
        {
            "ax": flat_axs[4],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "radius_gyration",
            "xlbl": rg_str(),
            "xmapfun": no_conversion,
            "xlim": (None, None),
            "c": "torsionmap",
        },
        {
            "ax": flat_axs[3],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "radius_gyration",
            "xlbl": rg_str(),
            "xmapfun": no_conversion,
            "xlim": (None, None),
            "c": "cmap",
        },
    )
    for i, axd in enumerate(axmap):
        ax = axd["ax"]

        # Want highlighting lines.
        if axd["xlbl"] == "number building blocks":
            ax.axvline(x=10, linestyle="--", c="gray", alpha=0.5)
            ax.axvline(x=15, linestyle="--", c="gray", alpha=0.5)

        # Do convexhull of full data set.
        if axd["x"] not in ("shape_measure", "cltitle"):
            full_data_y = list(all_data[axd["y"]])
            full_data_x = [axd["xmapfun"](i) for i in all_data[axd["x"]]]
            points = np.column_stack((full_data_x, full_data_y))
            points = points[~np.isnan(points).any(axis=1), :]
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(
                    points[simplex, 0],
                    points[simplex, 1],
                    "r",
                    linestyle="--",
                )

        if axd["x"] == "cltitle":
            for x in (3, 4):
                yvalues = [
                    bb_data[i][axd["y"]]
                    for i in bb_data
                    if axd["xmapfun"](bb_data[i][axd["x"]]) == x
                ]
                parts = ax.violinplot(
                    yvalues,
                    [x],
                    # points=200,
                    vert=True,
                    widths=0.6,
                    showmeans=False,
                    showextrema=False,
                    showmedians=False,
                    # bw_method=0.5,
                )

                for pc in parts["bodies"]:
                    pc.set_facecolor("#7A8B99")
                    pc.set_edgecolor("k")
                    pc.set_alpha(1.0)

            ax.set_xticks((3, 4))
            ax.set_xticklabels(("3", "4"))

        else:
            yvalues = [bb_data[i][axd["y"]] for i in bb_data]
            xvalues = [axd["xmapfun"](bb_data[i][axd["x"]]) for i in bb_data]
            cvalues = [bb_data[i][axd["c"]] for i in bb_data]
            ax.scatter(
                xvalues,
                yvalues,
                c=cvalues,
                edgecolor="white",
                s=100,
                alpha=0.8,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"{axd['xlbl']}", fontsize=16)
        ax.set_ylabel(f"{axd['ylbl']}", fontsize=16)
        ax.set_xlim(axd["xlim"])

        if i in (0, 4):
            if i == 4:
                legend_map = (
                    (convert_tors("ton", num=False), "#0B2027"),
                    (convert_tors("toff", num=False), "#CA1551"),
                )
            else:
                legend_map = (
                    ("3C", "#086788"),
                    ("4C", "#F9A03F"),
                    (convert_topo("6P8"), "k"),
                )

            legend_elements = []
            for labl, c in legend_map:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=labl,
                        markerfacecolor=c,
                        markersize=10,
                        markeredgecolor="k",
                        alpha=0.8,
                    )
                )
            ax.legend(handles=legend_elements, fontsize=16, ncol=1)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_6p8(all_data, figure_output):
    logging.info("doing 6p8 phase space")

    vdata = all_data[all_data["vdws"] == "von"]
    fig, ax = plt.subplots(figsize=(8, 2))
    bbs = (
        ("4C1m1b13C1n1b1f9", 1),
        ("4C1m1b13C1n1b1f28", 2),
        ("4C1m1b13C1n1b1f32", 3),
        ("4C1m1b13C1n1b1f36", 4),
        ("4C1m1b13C1n1b1f35", 5),
    )
    for bb_pair, idex in bbs:
        bbd = vdata[vdata["bbpair"] == bb_pair]
        data = bbd[bbd["torsions"] == "toff"]
        pore_size = float(data["pore"])
        ax.scatter(
            idex,
            pore_size,
            c="#F9A03F",
            edgecolor="k",
            s=120,
            alpha=1.0,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.set_xlabel(clangle_str(4), fontsize=16)
    ax.set_ylabel(pore_str(), fontsize=16)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_6p8.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_3(all_data, figure_output):
    raise NotImplementedError()
    logging.info("doing phase space 3")

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_topology()

    vdata = all_data[all_data["vdws"] == "von"]
    groups = vdata.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"]},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"]},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"]},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"]},
    }

    for gid, dfi in groups:
        bbtitle = gid[:3]
        if "mixed" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["mixed"] = 0
        if "mixed" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["mixed"] = 0
        if "unstable" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["unstable"] = 0
        if "unstable" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["unstable"] = 0

        for tors in ("ton", "toff"):
            fin_data = dfi[dfi["torsions"] == tors]
            if "6P8" in set(fin_data["topology"]):
                continue
            mixed_energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in fin_data.iterrows()
                if float(row["energy_per_bb"]) < isomer_energy()
            }

            if len(mixed_energies) == 0:
                topo_str = "unstable"
            elif len(mixed_energies) > 1:
                topo_str = "mixed"
            else:
                topo_str = list(mixed_energies.keys())[0]

            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data, strict=True):
        coords = data[(bbtitle, torsion)]
        bars = ax.bar(
            [convert_topo(i) for i in coords.keys()],
            coords.values(),
            color="#086788",
            edgecolor="k",
        )

        ax.bar_label(bars, padding=2, fontsize=16)

        title = f"{convert_tors(torsion, num=False)}"
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)
        ax.set_ylim(0, 140)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_5(all_data, figure_output):
    logging.info("doing ps5, shape vectors vs Energy")
    raise NotImplementedError("if you want this, fix it.")
    tstrs = topology_labels(short="P")
    clangles = sorted(set(all_data["clangle"]))
    for tstr, clangle in itertools.product(tstrs, clangles):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=(16, 5),
        )

        tdata = all_data[all_data["topology"] == tstr]
        cdata = tdata[tdata["clangle"] == clangle]
        if len(cdata) == 0:
            continue

        for ax, tor in zip(axs, ("ton", "toff"), strict=True):
            findata = cdata[cdata["torsions"] == tor]
            xvalues = list(findata["energy_per_bb"])
            yvalues = list(findata["sv_n_dist"])
            lvalues = list(findata["sv_l_dist"])
            to_plot_x = []
            to_plot_y = []
            to_plot_l = []
            for x, y, lval in zip(xvalues, yvalues, lvalues, strict=True):
                if pd.isna(x) or pd.isna(y):
                    continue
                to_plot_x.append(x)
                to_plot_y.append(y)
                if pd.isna(lval):
                    continue
                to_plot_l.append(lval)

            ax.scatter(
                to_plot_x,
                to_plot_y,
                c="#086788",
                edgecolor="none",
                s=50,
                alpha=1,
            )
            if len(to_plot_l) == len(to_plot_x):
                ax.scatter(
                    to_plot_x,
                    to_plot_l,
                    c="white",
                    edgecolor="k",
                    s=60,
                    alpha=1,
                    zorder=-1,
                )
            elif len(to_plot_l) != 0:
                raise ValueError(f"{len(to_plot_l)} l != {len(to_plot_x)}")

            ax.set_title(
                (
                    f"{convert_topo(tstr)}: "
                    f"{convert_tors(tor, num=False)}: {clangle}"
                ),
                fontsize=16,
            )
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel(convert_prop("energy_per_bb"), fontsize=16)
            ax.set_ylabel(convert_prop("both_sv_n_dist"), fontsize=16)
            ax.set_ylim(0.0, 1.05)

        ax.scatter(
            None,
            None,
            c="#086788",
            edgecolor="none",
            s=50,
            alpha=1,
            label="node",
        )
        ax.scatter(
            None,
            None,
            c="white",
            edgecolor="k",
            s=60,
            alpha=1,
            label="ligand",
        )
        ax.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"ps_5_{tstr}_{clangle}.pdf"),
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

    figure_output = figures()
    calculation_output = calculations()
    data_output = outputdata()

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=data_output,
    )
    low_e_data = get_lowest_energy_data(
        all_data=all_data,
        output_dir=data_output,
    )
    logging.info(f"there are {len(all_data)} collected data")

    phase_space_2(low_e_data, figure_output)
    phase_space_6p8(low_e_data, figure_output)
    phase_space_3(low_e_data, figure_output)
    raise SystemExit()
    phase_space_5(low_e_data, figure_output)


if __name__ == "__main__":
    main()
