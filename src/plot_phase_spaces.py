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

from env_set import cages

from analysis_utilities import (
    write_out_mapping,
    map_cltype_to_shapetopology,
    convert_tors,
    convert_prop,
    convert_topo,
    topo_to_colormap,
    max_energy,
    stoich_map,
    data_to_array,
    isomer_energy,
)


def phase_space_2(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    axmap = (
        {
            "ax": flat_axs[0],
            "tor": "toff",
            "x": "pore",
            "y": "energy",
        },
        {
            "ax": flat_axs[4],
            "tor": "ton",
            "x": "pore",
            "y": "energy",
        },
        {
            "ax": flat_axs[1],
            "tor": "toff",
            "x": "min_b2b",
            "y": "energy",
        },
        {
            "ax": flat_axs[5],
            "tor": "ton",
            "x": "min_b2b",
            "y": "energy",
        },
        {
            "ax": flat_axs[2],
            "tor": "toff",
            "x": "sv_l_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[6],
            "tor": "ton",
            "x": "sv_l_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[3],
            "tor": "toff",
            "x": "sv_n_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[7],
            "tor": "ton",
            "x": "sv_n_dist",
            "y": "energy",
        },
    )
    for axd in axmap:
        ax = axd["ax"]
        tdata = all_data[all_data["torsions"] == axd["tor"]]
        edata = tdata[tdata["energy"] < max_energy() * 50]
        xvalues = edata[axd["x"]]
        yvalues = edata[axd["y"]]
        hb = ax.hexbin(
            xvalues,
            yvalues,
            gridsize=20,
            cmap="inferno",
            bins="log",
            vmax=len(all_data),
        )
        cbar = fig.colorbar(hb, ax=ax, label="log10(N)")
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"{axd['x']}", fontsize=16)
        ax.set_ylabel(f"{axd['y']}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_3(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

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
            energies = {
                str(row["topology"]): float(row["energy"])
                / stoich_map(str(row["topology"]))
                for i, row in fin_data.iterrows()
            }
            num_mixed = len(
                tuple(
                    i
                    for i in list(energies.values())
                    if i < isomer_energy()
                )
            )
            min_energy = min(energies.values())
            if min_energy > max_energy():
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

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
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
    raise SystemExit("redefine pers, then rerun,")
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

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

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
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
    raise SystemExit("redefine pers, then rerun,")
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

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

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
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

    cmap = topo_to_colormap()
    clangles = sorted(set(all_data["clangle"]))
    for tstr, clangle in itertools.product(cmap, clangles):
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
            xvalues = list(findata["energy_per_bond"])
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
            ax.set_xlabel(convert_prop("energy_per_bond"), fontsize=16)
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

    figure_output = cages() / "figures"
    calculation_output = cages() / "calculations"

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=calculation_output,
    )
    logging.info(f"there are {len(all_data)} collected data")
    opt_data = all_data[all_data["optimised"]]
    logging.info(f"there are {len(opt_data)} successfully opted")
    write_out_mapping(opt_data)

    phase_space_2(opt_data, figure_output)
    phase_space_5(opt_data, figure_output)
    phase_space_3(opt_data, figure_output)
    phase_space_11(opt_data, figure_output)
    phase_space_12(opt_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
