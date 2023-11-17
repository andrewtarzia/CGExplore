#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Script to plot phase spaces.

Author: Andrew Tarzia

"""

import logging
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import (
    Xc_map,
    angle_str,
    convert_topo,
    convert_tors,
    data_to_array,
    eb_str,
    isomer_energy,
    mapshape_to_topology,
    shape_threshold,
    target_shapes,
    topology_labels,
)
from env_set import calculations, figures, outputdata
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
                values = [i for i in list(tdata[shape]) if not pd.isna(i)]

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
            os.path.join(figure_output, f"shape_vectors_{shape_type}.pdf"),
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


def shape_topology(all_data, figure_output):
    logging.info("running shape_topology")

    color_map = topology_labels(short="P")
    trim = all_data[all_data["vdws"] == "von"]

    xmin, xmax, width = 0, 100, 30

    for tstr in color_map:
        tdata = trim[trim["topology"] == tstr]
        for shape_type in ("n", "l"):
            if tstr == "6P8":
                fig, ax = plt.subplots(figsize=(8, 4))
                tor_tests = ("toff",)
                flat_axs = (ax,)

            else:
                fig, axs = plt.subplots(
                    ncols=2,
                    nrows=1,
                    sharey=True,
                    figsize=(16, 4),
                )
                tor_tests = ("ton", "toff")
                flat_axs = axs.flatten()

            try:
                target_shape = mapshape_to_topology(shape_type, False)[tstr]
            except KeyError:
                continue

            c_column = f"{shape_type}_{target_shape}"

            for ax, tor in zip(flat_axs, tor_tests, strict=True):
                pdata = tdata[tdata["torsions"] == tor]

                lowevalues = []
                values = []
                for _i, row in pdata.iterrows():
                    dist = float(row[c_column])

                    if pd.isna(dist):
                        continue
                    energy = float(row["energy_per_bb"])
                    values.append(dist)
                    if energy < isomer_energy():
                        lowevalues.append(dist)

                if len(values) == 0:
                    continue
                ax.hist(
                    values,
                    bins=np.linspace(xmin, xmax, width),
                    color="#086788",
                    alpha=1.0,
                    edgecolor="k",
                    linewidth=2,
                    density=False,
                    histtype="stepfilled",
                    label="all",
                )
                ax.hist(
                    lowevalues,
                    bins=np.linspace(xmin, xmax, width),
                    color="#F9A03F",
                    alpha=1.0,
                    edgecolor="k",
                    linewidth=2,
                    density=False,
                    histtype="stepfilled",
                    label=(
                        f"{eb_str(True)} < {isomer_energy()} " "kJmol$^{-1}$"
                    ),
                )
                ax.set_title(
                    (
                        f"{convert_topo(tstr)}: "
                        f"{convert_tors(tor, num=False)}"
                    ),
                    fontsize=16,
                )
                ax.set_xlabel(target_shape, fontsize=16)
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_ylabel("count", fontsize=16)
                ax.set_yscale("log")

            ax.legend(fontsize=16)
            fig.tight_layout()
            filename = f"shape_topology_{shape_type}_{tstr}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def shape_topology_main(all_data, figure_output):
    logging.info("running shape_topology_main")

    color_map = topology_labels(short="P")
    trim = all_data[all_data["vdws"] == "von"]

    xmin, xmax, width = 0, 40, 30

    for tstr in color_map:
        tdata = trim[trim["topology"] == tstr]
        for shape_type in ("n", "l"):
            if tstr not in ("6P9", "6P12", "8P12", "8P16"):
                continue

            else:
                fig, ax = plt.subplots(figsize=(5, 4))
                tor_tests = ("ton",)

            try:
                target_shape = mapshape_to_topology(shape_type, False)[tstr]
            except KeyError:
                continue

            c_column = f"{shape_type}_{target_shape}"

            for tor in tor_tests:
                pdata = tdata[tdata["torsions"] == tor]

                lowevalues = []
                values = []
                for _i, row in pdata.iterrows():
                    dist = float(row[c_column])

                    if pd.isna(dist):
                        continue
                    energy = float(row["energy_per_bb"])
                    values.append(dist)
                    if energy < isomer_energy():
                        lowevalues.append(dist)

                if len(values) == 0:
                    continue
                ax.hist(
                    values,
                    bins=np.linspace(xmin, xmax, width),
                    color="#086788",
                    alpha=1.0,
                    edgecolor="k",
                    linewidth=1,
                    density=False,
                    histtype="stepfilled",
                    label="all",
                )
                ax.hist(
                    lowevalues,
                    bins=np.linspace(xmin, xmax, width),
                    color="#F9A03F",
                    alpha=1.0,
                    edgecolor="k",
                    linewidth=1,
                    density=False,
                    histtype="stepfilled",
                    label=(
                        f"{eb_str(True)} < {isomer_energy()} " "kJmol$^{-1}$"
                    ),
                )
                ax.set_title(
                    (
                        f"{convert_topo(tstr)}: "
                        f"{convert_tors(tor, num=False)}"
                    ),
                    fontsize=16,
                )
                ax.set_xlabel(target_shape, fontsize=16)
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_ylabel("count", fontsize=16)
                ax.set_yscale("log")

            ax.legend(fontsize=16)
            fig.tight_layout()
            filename = f"shape_topology_{shape_type}_{tstr}_main.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def shape_input_relationships(all_data, figure_output):
    logging.info("running shape_input_relationships")

    color_map = topology_labels(short="P")
    trim = all_data[all_data["vdws"] == "von"]

    vmax = 5
    vmin = 0

    for tstr in color_map:
        tdata = trim[trim["topology"] == tstr]
        for shape_type in ("n", "l"):
            if tstr == "6P8":
                fig, ax = plt.subplots(figsize=(8, 4))
                tor_tests = ("toff",)
                flat_axs = (ax,)

            else:
                fig, axs = plt.subplots(
                    ncols=2,
                    nrows=1,
                    sharey=True,
                    figsize=(16, 4),
                )
                tor_tests = ("ton", "toff")
                flat_axs = axs.flatten()

            try:
                target_shape = mapshape_to_topology(shape_type, False)[tstr]
            except KeyError:
                continue

            c_column = f"{shape_type}_{target_shape}"

            for ax, tor in zip(flat_axs, tor_tests, strict=True):
                pdata = tdata[tdata["torsions"] == tor]
                if tstr == "6P8":
                    ax.set_xlabel(angle_str(3), fontsize=16)

                else:
                    ax.set_xlabel(
                        angle_str(2),
                        fontsize=16,
                    )
                    ax.set_title(
                        f"{convert_tors(tor, num=False)}",
                        fontsize=16,
                    )

                xdata = []
                ydata = []
                cdata = []
                for _i, row in pdata.iterrows():
                    dist = float(row[c_column])

                    if pd.isna(dist):
                        continue
                    energy = float(row["energy_per_bb"])
                    cdata.append(dist)
                    if tstr == "6P8":
                        xdata.append(float(row["c3angle"]))
                        ydata.append(float(row["clangle"]))

                    else:
                        xdata.append(float(row["c2angle"]))
                        ydata.append(float(row["clangle"]))
                    if energy < isomer_energy():
                        ax.scatter(
                            xdata[-1],
                            ydata[-1],
                            c="white",
                            s=360,
                            marker="s",
                            lw=3,
                            alpha=0.9,
                            edgecolor="k",
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
                    edgecolor="none",
                    s=180,
                    marker="s",
                    cmap="Blues_r",
                )

                ax.set_title(
                    (
                        f"{convert_topo(tstr)}: "
                        f"{convert_tors(tor, num=False)}"
                    ),
                    fontsize=16,
                )
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_ylabel(angle_str(Xc_map(tstr)), fontsize=16)

            cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
            cmap = mpl.cm.Blues_r
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation="vertical",
            )
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(f"{target_shape}", fontsize=16)

            fig.tight_layout()
            filename = f"shape_{shape_type}_map_{tstr}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def plot_topology_flex(data, comparison, mode, figure_output):
    if comparison == "shaped":
        id1 = 2
        id2 = 1
        ylabl = "shaped / total"
        show_number = False

    elif comparison == "stableshaped":
        id1 = 3
        id2 = 0
        ylabl = "shaped+stable / stable"
        show_number = False

    if mode == "l":
        title = "ditopic shape"
        tstr_ignore = (
            "6P9",
            "8P12",
            "3P6",
            "4P62",
            "4P8",
            "4P82",
            "6P12",
            "8P16",
            "12P24",
        )
        vlinepos = 1.5

    elif mode == "n":
        title = "tri/tetratopic shape"
        tstr_ignore = (
            "2P3",
            "4P62",
            "2P4",
            "12P24",
        )
        vlinepos = 2.5

    fig, ax = plt.subplots(figsize=(8, 5))

    categories_ton = {convert_topo(i): 0 for i in data if i not in tstr_ignore}
    categories_toff = {
        convert_topo(i): 0 for i in data if i not in tstr_ignore
    }
    categories_subtracted = {
        convert_topo(i): 0 for i in data if i not in tstr_ignore
    }

    for tstr in data:
        if tstr in tstr_ignore:
            continue

        try:
            categories_ton[convert_topo(tstr)] = (
                data[tstr]["ton"][mode][id1] / data[tstr]["ton"][mode][id2]
            ) * 100
            categories_toff[convert_topo(tstr)] = (
                data[tstr]["toff"][mode][id1] / data[tstr]["toff"][mode][id2]
            ) * 100
            if categories_toff[convert_topo(tstr)] == 0:
                categories_subtracted[convert_topo(tstr)] = 0
            else:
                categories_subtracted[convert_topo(tstr)] = (
                    categories_toff[convert_topo(tstr)]
                    - categories_ton[convert_topo(tstr)]
                ) / categories_toff[convert_topo(tstr)]
        except KeyError:
            continue

    ax.bar(
        categories_ton.keys(),
        categories_ton.values(),
        color="#086788",
        edgecolor="none",
        lw=2,
        label=convert_tors("ton", num=False),
    )

    ax.bar(
        categories_toff.keys(),
        categories_toff.values(),
        color="none",
        edgecolor="k",
        lw=2,
        label=convert_tors("toff", num=False),
    )

    if show_number:
        for x, tstr in enumerate(categories_toff):
            ax.text(
                x=x - 0.3,
                y=3,
                s=round(categories_subtracted[tstr], 1),
                c="white",
                fontsize=16,
            )

    ax.axvline(x=vlinepos, linestyle="--", c="gray", lw=2)

    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(ylabl, fontsize=16)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=16)
    ax.set_xticks(range(len(categories_toff)))
    ax.set_xticklabels(categories_ton.keys(), rotation=45)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figure_output,
            f"flexshapeeffect_topologies_{mode}_{comparison}.pdf",
        ),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def flexshapeeffect_per_property(all_data, figure_output):
    msg = (
        "there is a bug here in handling no stable cases and I do not use this"
    )
    raise NotImplementedError(msg)
    logging.info("running effect of flexibility distributions")

    trim = all_data[all_data["vdws"] == "von"]
    color_map = {
        "toff": ("white", "o", 120, "k"),
        "ton": ("#086788", "o", 80, "none"),
        # "toff", "3C1"): "#0B2027",
        # "toff", "4C1"): "#7A8B99",
    }

    topologies = [i for i in topology_labels(short="P") if i != "6P8"]

    topology_data = {}
    for tstr in topologies:
        tdata = trim[trim["topology"] == tstr]
        topology_data[tstr] = {}

        for tor in color_map:
            tor_data = tdata[tdata["torsions"] == tor]

            stable_data = tor_data[tor_data["energy_per_bb"] < isomer_energy()]
            topology_data[tstr][tor] = {}
            for shape_type in ("n", "l"):
                try:
                    target_shape = mapshape_to_topology(shape_type, False)[
                        tstr
                    ]
                except KeyError:
                    continue

                c_column = f"{shape_type}_{target_shape}"

                shaped_data = tor_data[tor_data[c_column] < shape_threshold()]
                stable_shaped_data = stable_data[
                    stable_data[c_column] < shape_threshold()
                ]
                topology_data[tstr][tor][shape_type] = (
                    len(stable_data),
                    len(tor_data),
                    len(shaped_data),
                    len(stable_shaped_data),
                )

    plot_topology_flex(topology_data, "shaped", "l", figure_output)
    plot_topology_flex(topology_data, "shaped", "n", figure_output)
    plot_topology_flex(topology_data, "stableshaped", "l", figure_output)
    plot_topology_flex(topology_data, "stableshaped", "n", figure_output)


def plot_shape_flex(data, mode, figure_output):
    ylabl = r"% ($s$ < 2)"

    for tstr in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        if tstr == "6P8":
            if mode == "l":
                title = "tritopic shape"
            elif mode == "n":
                title = "tetratopic shape"
        else:
            if mode == "l":
                title = "ditopic shape"
            elif mode == "n":
                if Xc_map(tstr) == 3:
                    title = "tritopic shape"
                elif Xc_map(tstr) == 4:
                    title = "tetratopic shape"

        ax.set_title(f"{convert_topo(tstr)}: {title}", fontsize=16)

        shapes = data[tstr]["toff"][mode].keys()
        if len(shapes) == 0:
            continue

        categories_ton = {i[2:]: 0 for i in list(shapes)}
        categories_toff = {i[2:]: 0 for i in list(shapes)}

        for shape in shapes:
            shape_str = shape[2:]
            if tstr != "6P8":
                categories_ton[shape_str] = (
                    data[tstr]["ton"][mode][shape][1]
                    / data[tstr]["ton"][mode][shape][0]
                ) * 100
            categories_toff[shape_str] = (
                data[tstr]["toff"][mode][shape][1]
                / data[tstr]["toff"][mode][shape][0]
            ) * 100

        if tstr != "6P8":
            ax.bar(
                categories_ton.keys(),
                categories_ton.values(),
                color="#086788",
                edgecolor="none",
                lw=2,
                label=convert_tors("ton", num=False),
            )

        ax.bar(
            categories_toff.keys(),
            categories_toff.values(),
            color="none",
            edgecolor="k",
            lw=2,
            label=convert_tors("toff", num=False),
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel(ylabl, fontsize=16)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=16)
        ax.set_xticks(range(len(categories_toff)))
        ax.set_xticklabels(categories_ton.keys(), rotation=45)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                figure_output,
                f"flexshapeeffect_shapes_{mode}_{tstr}.pdf",
            ),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def flexshapeeffect_per_shape(all_data, figure_output):
    logging.info("running shaped percent per topologies")

    trim = all_data[all_data["vdws"] == "von"]
    color_map = {
        "toff": ("white", "o", 120, "k"),
        "ton": ("#086788", "o", 80, "none"),
        # "toff", "3C1"): "#0B2027",
        # "toff", "4C1"): "#7A8B99",
    }

    columns = {
        "n": [i for i in all_data.columns if i[:2] == "n_"],
        "l": [i for i in all_data.columns if i[:2] == "l_"],
    }
    topologies = list(topology_labels(short="P"))

    topology_data = {}
    for tstr in topologies:
        tdata = trim[trim["topology"] == tstr]
        topology_data[tstr] = {}

        for tor in color_map:
            tor_data = tdata[tdata["torsions"] == tor]
            topology_data[tstr][tor] = {}
            for shape_type in columns:
                topology_data[tstr][tor][shape_type] = {}
                for column in columns[shape_type]:
                    col_values = [
                        i for i in tor_data[column] if not pd.isna(i)
                    ]
                    if len(col_values) > 0:
                        shaped_values = [
                            i for i in col_values if i < shape_threshold()
                        ]
                        topology_data[tstr][tor][shape_type][column] = (
                            len(col_values),
                            len(shaped_values),
                        )

    plot_shape_flex(topology_data, "l", figure_output)
    plot_shape_flex(topology_data, "n", figure_output)


def shape_persistence_map(all_data, figure_output):
    logging.info("running shape_input_relationships")

    trim = all_data[all_data["vdws"] == "von"]

    topologies = topology_labels(short="P")

    fig, axs = plt.subplots(
        ncols=2,
        nrows=len(topologies),
        sharex=True,
        sharey=True,
        figsize=(16, 16),
    )

    xmin = 0
    xmax = 1
    num_bins = 40

    for i, tstr in enumerate(topologies):
        tdata = trim[trim["topology"] == tstr]
        for j, tor in enumerate(("ton", "toff")):
            ax = axs[i][j]
            pdata = tdata[tdata["torsions"] == tor]
            edata = pdata[pdata["energy_per_bb"] < isomer_energy()]

            ax.hist(
                edata["sv_n_dist"],
                bins=np.linspace(xmin, xmax, num_bins),
                color="#086788",
                alpha=1.0,
                density=True,
                histtype="stepfilled",
                label="node",
            )
            ax.hist(
                edata["sv_l_dist"],
                bins=np.linspace(xmin, xmax, num_bins),
                color="#F9A03F",
                alpha=0.8,
                density=True,
                histtype="stepfilled",
                label="ligand",
            )
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_yticks(())
            if j == 0:
                ax.text(
                    x=0.05,
                    y=2,
                    s=f"{convert_topo(tstr)}",
                    fontsize=16,
                )
            if i == 0:
                ax.set_title(convert_tors(tor, num=False), fontsize=16)
            if i == len(topologies) - 1:
                if j == 0:
                    ax.set_ylabel("frequency", fontsize=16)
                    ax.legend(fontsize=16)
                ax.set_xlabel("cosine similarity", fontsize=16)

    fig.tight_layout()
    filename = "shape_persistence_map.pdf"
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def shape_summary(all_data, figure_output):
    logging.info("running shape_summary")
    color_map = topology_labels(short="P")
    trim = all_data[all_data["vdws"] == "von"]

    fig, axs = plt.subplots(
        nrows=2,
        sharex=True,
        sharey=True,
        figsize=(8, 6),
    )
    count = 0
    xpos = []
    xlabls = []
    for tstr in color_map:
        xpos.append(count)
        xlabls.append(convert_topo(tstr))
        tdata = trim[trim["topology"] == tstr]
        for shape_type in ("n", "l"):
            tor_tests = ("toff",) if tstr == "6P8" else ("ton", "toff")

            try:
                target_shape = mapshape_to_topology(shape_type, False)[tstr]
            except KeyError:
                continue

            c_column = f"{shape_type}_{target_shape}"

            if shape_type == "n":
                ax = axs[0]
            elif shape_type == "l":
                ax = axs[1]

            for tor in tor_tests:
                if tor == "ton":
                    color = "#086788"
                    shift = 0.2
                elif tor == "toff":
                    color = "#F9A03F"
                    shift = -0.2

                pdata = tdata[tdata["torsions"] == tor]

                lowevalues = []
                values = []
                for _i, row in pdata.iterrows():
                    dist = float(row[c_column])

                    if pd.isna(dist):
                        continue
                    energy = float(row["energy_per_bb"])
                    values.append(dist)
                    if energy < isomer_energy():
                        lowevalues.append(dist)
                if len(lowevalues) != 0:
                    tp = np.std(lowevalues)
                    print(
                        len(values),
                        len(lowevalues),
                        tor,
                        tstr,
                        c_column,
                        tp,
                    )
                    parts = ax.violinplot(
                        lowevalues,
                        [xpos[-1] - shift],
                        vert=True,
                        widths=0.18,
                        showmeans=True,
                        showextrema=True,
                        showmedians=True,
                    )

                    for pc in parts["bodies"]:
                        pc.set_facecolor(color)
                        pc.set_edgecolor("none")
                        pc.set_alpha(1.0)

        count += 1

    for ax in axs:
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabls, rotation=45)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel(
            (f"$s$ ({eb_str(True)} < {isomer_energy()}" " kJmol$^{-1}$)"),
            fontsize=16,
        )
        ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_summary.pdf"),
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
    logging.info(f"there are {len(all_data)} collected data")

    flexshapeeffect_per_shape(all_data, figure_output)
    shape_topology(all_data, figure_output)
    shape_topology_main(all_data, figure_output)
    shape_input_relationships(all_data, figure_output)
    shape_summary(all_data, figure_output)
    raise SystemExit
    flexshapeeffect_per_property(all_data, figure_output)
    shape_vector_distributions(all_data, figure_output)
    shape_persistence_map(all_data, figure_output)
    shape_similarities(all_data, figure_output)


if __name__ == "__main__":
    main()
