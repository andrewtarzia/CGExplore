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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import numpy as np

from env_set import cages

from analysis_utilities import (
    write_out_mapping,
    get_lowest_energy_data,
    convert_tors,
    convert_topo,
    topology_labels,
    data_to_array,
    isomer_energy,
    target_shapes,
    mapshape_to_topology,
)


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
                s=topology_options,
                fontsize=16,
            )
            ax.axhline(y=height, c="k")

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("shape measure", fontsize=16)
            ax.set_ylabel("frequency", fontsize=16)
            ax.set_ylim(0, height + 1)
            ax.set_yticks([])

            height += 1

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                figure_output, f"shape_vectors_{shape_type}.pdf"
            ),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def shape_similarities(all_data, figure_output):
    logging.info("running shape_similarities")
    tstrs = topology_labels(short="P")

    xmin = 0
    xmax = 1
    num_bins = 25
    yjump = 50
    cmap = {
        ("ton", "n"): ("#086788", "stepfilled"),
        ("toff", "n"): ("#F9A03F", "stepfilled"),
        ("ton", "l"): ("#0B2027", "stepfilled"),
        ("toff", "l"): ("#7A8B99", "stepfilled"),
    }

    fig, ax = plt.subplots(figsize=(8, 10))
    height = 0
    for tstr in tstrs:
        t_data = all_data[all_data["topology"] == tstr]
        for tor in ("ton", "toff"):
            tor_data = t_data[t_data["torsions"] == tor]
            node_similarities = [
                i for i in list(tor_data["sv_n_dist"]) if not pd.isna(i)
            ]
            ligand_similarities = [
                i for i in list(tor_data["sv_l_dist"]) if not pd.isna(i)
            ]
            ax.hist(
                x=node_similarities,
                bins=np.linspace(xmin, xmax, num_bins),
                density=True,
                bottom=height,
                histtype=cmap[(tor, "n")][1],
                color=cmap[(tor, "n")][0],
                alpha=0.7,
            )
            ax.hist(
                x=ligand_similarities,
                bins=np.linspace(xmin, xmax, num_bins),
                density=True,
                bottom=height,
                histtype=cmap[(tor, "l")][1],
                color=cmap[(tor, "l")][0],
                alpha=0.7,
            )
        ax.text(
            x=0.8,
            y=height + 0.7 * yjump,
            s=f"{convert_topo(tstr)}",
            fontsize=16,
        )
        ax.set_ylim(0, height + yjump)
        ax.set_yticks([])
        ax.axhline(y=height, c="k")
        height += yjump

    legend_elements = []
    for i in cmap:
        legend_elements.append(
            Patch(
                facecolor=cmap[i][0],
                label=f"{i[1]}: {convert_tors(i[0], num=False)}",
                alpha=0.7,
            ),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("cosine similarity", fontsize=16)
    ax.set_ylabel("frequency", fontsize=16)
    ax.legend(ncol=2, handles=legend_elements, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_similarities.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def shape_input_relationships(all_data, figure_output):
    logging.info("running shape_input_relationships")

    color_map = topology_labels(short="P")

    vmax = 1
    vmin = 0.9

    for tstr in color_map:
        tdata = all_data[all_data["topology"] == tstr]
        for shape_type in ("n", "l"):
            if tstr == "6P8":
                fig, ax = plt.subplots(figsize=(8, 5))
                tor_tests = ("toff",)
                flat_axs = (ax,)

            else:
                fig, axs = plt.subplots(
                    ncols=2,
                    nrows=1,
                    sharey=True,
                    figsize=(16, 5),
                )
                tor_tests = ("ton", "toff")
                flat_axs = axs.flatten()

            if shape_type == "n":
                c_column = "sv_n_dist"
            elif shape_type == "l":
                c_column = "sv_l_dist"

            for ax, tor in zip(flat_axs, tor_tests):
                pdata = tdata[tdata["torsions"] == tor]
                if tstr == "6P8":
                    ax.set_xlabel("c3 angle [deg]", fontsize=16)

                else:
                    ax.set_xlabel("bite angle [deg]", fontsize=16)
                    ax.set_title(
                        f"{convert_tors(tor, num=False)}",
                        fontsize=16,
                    )

                xdata = []
                ydata = []
                cdata = []
                for i, row in pdata.iterrows():
                    dist = float(row[c_column])

                    if pd.isna(dist):
                        continue
                    energy = float(row["energy_per_bb"])
                    cdata.append(dist)
                    if tstr == "6P8":
                        xdata.append(float(row["c3angle"]))
                        ydata.append(float(row["clangle"]))

                    else:
                        xdata.append(float(row["target_bite_angle"]))
                        ydata.append(float(row["clangle"]))
                    if energy < isomer_energy():
                        ax.scatter(
                            xdata[-1],
                            ydata[-1],
                            c="white",
                            s=400,
                            marker="s",
                            lw=3,
                            alpha=0.7,
                            edgecolor="firebrick",
                        )

                if len(cdata) == 0:
                    continue
                ax.scatter(
                    xdata,
                    ydata,
                    c=cdata,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=1.0,
                    # edgecolor="k",
                    s=200,
                    marker="s",
                    cmap="Blues",
                )

                ax.set_title(
                    f"{convert_tors(tor, num=False)}",
                    fontsize=16,
                )
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_ylabel("cl angle [deg]", fontsize=16)

            cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
            cmap = mpl.cm.Blues
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation="vertical",
            )
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label("cosine similarity", fontsize=16)

            fig.tight_layout()
            filename = f"shape_{shape_type}_map_{tstr}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def shape_energy_input_relationships(all_data, figure_output):
    logging.info("running shape_input_relationships")

    color_map = topology_labels(short="P")

    vmax = 1
    vmin = 0.9

    for tstr in color_map:
        tdata = all_data[all_data["topology"] == tstr]
        for shape_type in ("n", "l"):
            if tstr == "6P8":
                fig, ax = plt.subplots(figsize=(8, 5))
                tor_tests = ("toff",)
                flat_axs = (ax,)

            else:
                fig, axs = plt.subplots(
                    ncols=2,
                    nrows=1,
                    sharey=True,
                    figsize=(16, 5),
                )
                tor_tests = ("ton", "toff")
                flat_axs = axs.flatten()

            if shape_type == "n":
                c_column = "sv_n_dist"
            elif shape_type == "l":
                c_column = "sv_l_dist"

            for ax, tor in zip(flat_axs, tor_tests):
                pdata = tdata[tdata["torsions"] == tor]
                if tstr == "6P8":
                    ax.set_xlabel("c3 angle [deg]", fontsize=16)

                else:
                    ax.set_xlabel("bite angle [deg]", fontsize=16)
                    ax.set_title(
                        f"{convert_tors(tor, num=False)}",
                        fontsize=16,
                    )
                xdata = []
                ydata = []
                cdata = []
                for i, row in pdata.iterrows():
                    dist = float(row[c_column])

                    if pd.isna(dist):
                        continue
                    energy = float(row["energy_per_bb"])
                    if energy < isomer_energy():
                        cdata.append(dist)
                        if tstr == "6P8":
                            xdata.append(float(row["c3angle"]))
                            ydata.append(float(row["clangle"]))

                        else:
                            xdata.append(
                                float(row["target_bite_angle"])
                            )
                            ydata.append(float(row["clangle"]))

                if len(cdata) == 0:
                    continue
                ax.scatter(
                    xdata,
                    ydata,
                    c=cdata,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=1.0,
                    edgecolor="k",
                    s=200,
                    marker="s",
                    cmap="Blues",
                )

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_ylabel("cl angle [deg]", fontsize=16)

            cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
            cmap = mpl.cm.Blues
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation="vertical",
            )
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label("cosine similarity", fontsize=16)

            fig.tight_layout()
            filename = f"shape_energy_{shape_type}_map_{tstr}.pdf"
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
    shape_similarities(low_e_data, figure_output)
    shape_input_relationships(low_e_data, figure_output)
    shape_energy_input_relationships(low_e_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
