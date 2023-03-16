#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot phase spaces.

Author: Andrew Tarzia

"""

import sys
import os
import logging
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

from env_set import cages

from analysis_utilities import (
    write_out_mapping,
    get_lowest_energy_data,
    map_cltype_to_topology,
    pore_str,
    rg_str,
    convert_tors,
    convert_prop,
    convert_topo,
    topology_labels,
    stoich_map,
    data_to_array,
    isomer_energy,
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
            stoichiometries = {
                i: stoich_map(i) for i in stable_energies
            }
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

                node_measure = float(kinetic_data["sv_n_dist"])
                ligand_measure = float(kinetic_data["sv_n_dist"])
                if node_measure is None:
                    shape_measure = ligand_measure
                elif ligand_measure is None:
                    shape_measure = node_measure
                else:
                    shape_measure = min((node_measure, ligand_measure))
                bb_data[(bb_pair, tor)] = {
                    "energy_per_bb": epb,
                    "topology": tstr,
                    "pore": float(kinetic_data["pore"]),
                    "radius_gyration": float(
                        kinetic_data["radius_gyration"]
                    ),
                    "shape_measure": shape_measure,
                    "cltitle": str(kinetic_data["cltitle"].iloc[0]),
                    "clangle": float(kinetic_data["clangle"]),
                    "c3angle": float(kinetic_data["c3angle"]),
                    "target_bite_angle": float(
                        kinetic_data["target_bite_angle"]
                    ),
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
            "xlbl": "cage stoichiometry",
            "xmapfun": stoich_map,
            "xlim": (None, None),
        },
        {
            "ax": flat_axs[1],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "cltitle",
            "xlbl": "largest coordination",
            "xmapfun": cltitleconversion,
            "xlim": (2.8, 4.2),
        },
        {
            "ax": flat_axs[2],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "clangle",
            "xlbl": "Cl angle [deg]",
            "xmapfun": no_conversion,
            "xlim": (48, 121),
        },
        {
            "ax": flat_axs[3],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "target_bite_angle",
            "xlbl": "target bite angle [deg]",
            "xmapfun": no_conversion,
            "xlim": (-1, 181),
        },
        {
            "ax": flat_axs[4],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "shape_measure",
            "xlbl": "min(shape measure)",
            "xmapfun": no_conversion,
            "xlim": (-0.1, 1.1),
        },
        {
            "ax": flat_axs[5],
            "y": "pore",
            "ylbl": pore_str(),
            "x": "radius_gyration",
            "xlbl": rg_str(),
            "xmapfun": no_conversion,
            "xlim": (None, None),
        },
    )
    for axd in axmap:

        ax = axd["ax"]

        # Do convexhull of full data set.
        if axd["x"] not in ("shape_measure", "cltitle"):
            full_data_y = list(all_data[axd["y"]])
            full_data_x = [
                axd["xmapfun"](i) for i in all_data[axd["x"]]
            ]
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

        yvalues = [bb_data[i][axd["y"]] for i in bb_data]
        xvalues = [
            axd["xmapfun"](bb_data[i][axd["x"]]) for i in bb_data
        ]
        ax.scatter(
            xvalues,
            yvalues,
            c="#086788",
            edgecolor="k",
            s=80,
        )

        if axd["x"] == "cltitle":
            ax.set_xticks((3, 4))

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"{axd['xlbl']}", fontsize=16)
        ax.set_ylabel(f"{axd['ylbl']}", fontsize=16)
        ax.set_xlim(axd["xlim"])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_3(all_data, figure_output):
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
        for tors in ("ton", "toff"):

            fin_data = dfi[dfi["torsions"] == tors]
            if "6P8" in set(fin_data["topology"]):
                continue
            energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in fin_data.iterrows()
            }
            if len(energies) < 1:
                continue

            num_mixed = len(
                tuple(
                    i
                    for i in list(energies.values())
                    if i < isomer_energy()
                )
            )

            min_energy = min(energies.values())
            if min_energy > isomer_energy():
                topo_str = "unstable"
            elif num_mixed > 1:
                topo_str = "mixed"
            else:
                topo_str = list(energies.keys())[
                    list(energies.values()).index(min_energy)
                ]
            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

        if "mixed" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["mixed"] = 0
        if "mixed" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["mixed"] = 0
        if "unstable" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["unstable"] = 0
        if "unstable" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["unstable"] = 0

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        bars = ax.bar(
            [convert_topo(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        ax.bar_label(bars, padding=3, fontsize=16)

        title = f"{bbtitle}, {torsion}: {isomer_energy()}eV"
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_11(all_data, figure_output):
    logging.info("doing phase space 11")
    raise SystemExit("redefine pers, then rerun,")
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_topology()

    opt_data = all_data[all_data["optimised"]]
    groups = opt_data.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"].values()},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"].values()},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"].values()},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"].values()},
    }
    for gid, dfi in groups:
        bbtitle = gid[:3]
        for tors in ("ton", "toff"):
            fin_data = dfi[dfi["torsions"] == tors]
            for tstr in topologies[bbtitle].values():
                t_data = fin_data[fin_data["topology"] == tstr]
                if len(t_data) != 1:
                    continue
                if t_data.iloc[0]["persistent"]:
                    topo_str = tstr
                else:
                    topo_str = "not"

                if topo_str not in data[(bbtitle, tors)]:
                    data[(bbtitle, tors)][topo_str] = 0
                data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        bars = ax.bar(
            [convert_topo(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        ax.bar_label(bars, padding=3, fontsize=16)

        title = f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("num. persistent", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_11.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_12(all_data, figure_output):
    logging.info("doing phase space 12")
    raise SystemExit("redefine pers, then rerun,")
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_topology()

    opt_data = all_data[all_data["optimised"]]
    groups = opt_data.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"].values()},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"].values()},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"].values()},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"].values()},
    }
    for gid, dfi in groups:
        bbtitle = gid[:3]
        for tors in ("ton", "toff"):
            fin_data = dfi[dfi["torsions"] == tors]
            per_data = fin_data[fin_data["persistent"]]
            present_topologies = list(per_data["topology"])
            if len(present_topologies) == 1:
                topo_str = present_topologies[0]
            else:
                topo_str = "mixed"

            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        bars = ax.bar(
            [convert_topo(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        ax.bar_label(bars, padding=3, fontsize=16)

        title = f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("num. persistent and sorted", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_12.pdf"),
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

        for ax, tor in zip(axs, ("ton", "toff")):
            findata = cdata[cdata["torsions"] == tor]
            xvalues = list(findata["energy_per_bb"])
            yvalues = list(findata["sv_n_dist"])
            lvalues = list(findata["sv_l_dist"])
            to_plot_x = []
            to_plot_y = []
            to_plot_l = []
            for x, y, l in zip(xvalues, yvalues, lvalues):
                if pd.isna(x) or pd.isna(y):
                    continue
                to_plot_x.append(x)
                to_plot_y.append(y)
                if pd.isna(l):
                    continue
                to_plot_l.append(l)

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
                raise ValueError(
                    f"{len(to_plot_l)} l != {len(to_plot_x)}"
                )

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

    figure_output = cages() / "ommfigures"
    calculation_output = cages() / "ommcalculations"

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=calculation_output,
    )
    low_e_data = get_lowest_energy_data(
        all_data=all_data,
        output_dir=calculation_output,
    )
    logging.info(f"there are {len(all_data)} collected data")
    write_out_mapping(all_data)

    phase_space_2(low_e_data, figure_output)
    raise SystemExit()
    phase_space_3(low_e_data, figure_output)
    phase_space_11(low_e_data, figure_output)
    phase_space_12(low_e_data, figure_output)
    phase_space_5(low_e_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
