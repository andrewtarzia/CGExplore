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
    write_out_mapping,
    eb_str,
    pore_str,
    rg_str,
    data_to_array,
    isomer_energy,
    get_lowest_energy_data,
    convert_topo,
    convert_tors,
    topology_labels,
    target_shapes,
    mapshape_to_topology,
)


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


def shape_vector_distributions(all_data, figure_output):
    logging.info("running shape_vector_distributions")

    trim = all_data[all_data["vdws"] == "von"]

    present_shape_values = [
        i
        for i in all_data.columns
        if i[:2] in ("l_", "n_") and i[2:] in target_shapes()
    ]

    xmin = 0
    xmax = 60
    num_bins = 60
    cmap = {
        "ton": ("#086788", "stepfilled"),
        "toff": ("#F9A03F", "stepfilled"),
    }
    for shape_type in ("n", "l"):

        fig, ax = plt.subplots(figsize=(8, 10))
        height = 0
        for i, shape in enumerate(present_shape_values):
            if shape[0] != shape_type:
                continue

            shape_type = shape[0]

            for tor in ("toff", "ton"):
                tdata = trim[trim["torsions"] == tor]
                values = [
                    i for i in list(tdata[shape]) if not pd.isna(i)
                ]

                ax.hist(
                    x=values,
                    bins=np.linspace(xmin, xmax, num_bins),
                    density=True,
                    bottom=height,
                    histtype=cmap[tor][1],
                    color=cmap[tor][0],
                    alpha=0.7,
                )
            ax.text(
                x=50,
                y=height + 0.7,
                # s=f"{shape_type}: {convert_tors(tor, num=False)}",
                s=f"{shape[2:]}",
                fontsize=16,
            )
            topology_options = ", ".join(
                [
                    convert_topo(i)
                    for i in mapshape_to_topology(
                        mode=shape_type,
                        from_shape=True,
                    )[shape[2:]]
                ]
            )
            ax.text(
                x=50,
                y=height + 0.4,
                # s=f"{shape_type}: {convert_tors(tor, num=False)}",
                s=topology_options,
                fontsize=16,
            )
            ax.axhline(y=height, c="k")

            # if lshape in keys:
            #     filt_data = tdata[tdata[lshape].notna()]
            #     l_values = list(filt_data[lshape])
            #     if len(l_values) == 0:
            #         continue
            #     ax.hist(
            #         x=l_values,
            #         bins=np.linspace(xmin, xmax, num_bins),
            #         bottom=0.6,
            #         density=True,
            #         histtype="stepfilled",
            #         color=cmap[(tor, "l")][0],
            #         lw=cmap[(tor, "l")][1],
            #         facecolor=cmap[(tor, "l")][0],
            #         # linestyle=torsion_dict[tors],
            #         # label=f"ligand: {convert_tors(tors, num=False)}",
            #         label=f"ligand: {convert_tors(tor, num=False)}",
            #     )
            #     ax.axhline(y=0.6, c="k", lw=1)

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("shape measure", fontsize=16)
            ax.set_ylabel("frequency", fontsize=16)
            # ax.set_xlim(0, xmax)
            # ax.set_yscale("log")
            ax.set_ylim(0, height + 1)
            ax.set_yticks([])

            height += 1

        # ax.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                figure_output, f"shape_vectors_{shape_type}.pdf"
            ),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def shape_vectors_2(all_data, figure_output):
    logging.info("running shape_vectors_2")
    tstrs = topology_labels(short="P")
    ntstrs = mapshape_to_topology(mode="n")
    ltstrs = mapshape_to_topology(mode="l")

    torsion_dict = {
        "ton": "-",
        "toff": "--",
    }

    fig, axs = plt.subplots(
        nrows=4,
        ncols=4,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    mtstrs = []
    for i, tstr in enumerate(tstrs):
        if tstr in ntstrs or tstr in ltstrs:
            mtstrs.append(tstr)

    for ax, tstr in zip(flat_axs, mtstrs):
        t_data = all_data[all_data["topology"] == tstr]
        count = 0
        for tors in torsion_dict:
            tor_data = t_data[t_data["torsions"] == tors]

            filt_data = tor_data[tor_data["sv_n_dist"].notna()]
            if len(filt_data) > 0:
                n_values = list(filt_data["sv_n_dist"])
                ax.hist(
                    x=n_values,
                    bins=50,
                    density=False,
                    histtype="step",
                    color="#DD1C1A",
                    linestyle=torsion_dict[tors],
                    lw=3,
                    label=f"node: {convert_tors(tors, num=False)}",
                )
                count += 1

            filt_data = tor_data[tor_data["sv_l_dist"].notna()]
            if len(filt_data) > 0:
                l_values = list(filt_data["sv_l_dist"])
                ax.hist(
                    x=l_values,
                    bins=50,
                    density=False,
                    histtype="step",
                    color="k",
                    lw=2,
                    linestyle=torsion_dict[tors],
                    label=f"ligand: {convert_tors(tors, num=False)}",
                )
                count += 1

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("cosine similarity", fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        ax.set_title(tstr, fontsize=16)
        # ax.set_yscale("log")
        if count == 4:
            ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_vectors_2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def shape_vectors_3(all_data, figure_output):
    logging.info("running shape_vectors_3")
    tstrs = topology_labels(short="P")
    ntstrs = mapshape_to_topology(mode="n")
    ltstrs = mapshape_to_topology(mode="l")

    torsion_dict = {
        "ton": "-",
        "toff": "--",
    }

    fig, axs = plt.subplots(
        nrows=4,
        ncols=4,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    mtstrs = {}
    for i, tstr in enumerate(tstrs):
        if tstr in ntstrs:
            mtstrs[tstr] = [f"n_{ntstrs[tstr]}"]
        if tstr in ltstrs:
            if tstr not in mtstrs:
                mtstrs[tstr] = []
            mtstrs[tstr].append(f"l_{ltstrs[tstr]}")

    for ax, tstr in zip(flat_axs, mtstrs):
        t_data = all_data[all_data["topology"] == tstr]

        ax_xlbl = set()
        for tors in torsion_dict:
            tor_data = t_data[t_data["torsions"] == tors]

            for scol in mtstrs[tstr]:
                ax_xlbl.add(scol.split("_")[-1])
                filt_data = tor_data[tor_data[scol].notna()]
                if len(filt_data) > 0:
                    values = list(filt_data[scol])
                    if "n" in scol:
                        c = "#DD1C1A"
                        lbl = f"node: {convert_tors(tors, num=False)}"
                        lw = 3
                    elif "l" in scol:
                        c = "k"
                        lbl = f"ligand: {convert_tors(tors, num=False)}"
                        lw = 2
                    ax.hist(
                        x=values,
                        bins=50,
                        density=False,
                        histtype="step",
                        color=c,
                        linestyle=torsion_dict[tors],
                        lw=lw,
                        label=lbl,
                    )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("; ".join(list(ax_xlbl)), fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        ax.set_title(tstr, fontsize=16)
        # ax.set_yscale("log")
        if len(mtstrs[tstr]) == 2:
            ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_vectors_3.pdf"),
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

    shape_vector_distributions(low_e_data, figure_output)
    phase_space_5(low_e_data, figure_output)
    shape_vectors_2(low_e_data, figure_output)
    shape_vectors_3(low_e_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
