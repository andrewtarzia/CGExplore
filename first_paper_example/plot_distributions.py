#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Script to plot distribitions.

Author: Andrew Tarzia

"""

import json
import logging
import os
import sys

import matplotlib as mpl
import matplotlib.patches as mpatches
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
    get_paired_cage_name,
    isomer_energy,
    pore_str,
    rg_str,
    topology_labels,
)
from cgexplore.utilities import convert_pyramid_angle
from env_set import calculations, figures, outputdata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def identity_distributions(all_data, figure_output):
    logging.info("running identity_distributions")

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {i: 0 for i in topology_labels(short="P")}
    count1 = all_data["topology"].value_counts()

    for tstr, count in count1.items():
        categories[tstr] = count

    categories = {convert_topo(i): categories[i] for i in categories}
    num_cages = len(all_data)

    ax.bar(
        categories.keys(),
        categories.values(),
        color="teal",
        edgecolor="teal",
        lw=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_title(f"total cages: {num_cages}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "all_bb_dists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def geom_distributions(all_data, geom_data, figure_output):
    logging.info("running geom_distributions")

    comparisons = {
        "torsions": {
            "measure": "dihedrals",
            "xlabel": r"$baab$ torsion [$^\circ$]",
            "units": None,
            "column": None,
            "label_options": ("Pb_Ba_Ba_Pb",),
        },
        "clangle": {
            "measure": "angles",
            "xlabel": r"$nbn$ or $mbm$ angle",
            "units": r"$^\circ$",
            "column": "clangle",
            "label_options": (
                "Pb_Pd_Pb",
                "Pb_C_Pb",
                # "C_Pb_Ba",
                # "Pd_Pb_Ba",
            ),
        },
        "clr0": {
            "measure": "bonds",
            "xlabel": r"$nb$ or $mb$ bond length",
            "units": r"$\mathrm{\AA}$",
            "column": "clr0",
            "label_options": ("Pb_Pd", "C_Pb"),
        },
        "c2angle": {
            "measure": "angles",
            "xlabel": r"$bac$ angle",
            "units": r"$^\circ$",
            "column": "c2angle",
            "label_options": ("Pb_Ba_Ag",),
        },
        "c2backbone": {
            "measure": "angles",
            "xlabel": r"$aca$ angle [$^\circ$]",
            "units": None,
            "column": None,
            "label_options": ("Ba_Ag_Ba",),
        },
        "c2r0": {
            "measure": "bonds",
            "xlabel": r"$ac$ bond length",
            "units": r"$\mathrm{\AA}$",
            "column": "c2r0",
            "label_options": ("Ag_Ba",),
        },
        "c2bonder": {
            "measure": "bonds",
            "xlabel": r"$ba$ bond length [$\mathrm{\AA}$]",
            "units": None,
            "column": None,
            "label_options": ("Ba_Pb",),
        },
    }

    tcmap = topology_labels(short="P")
    tcpos = {tstr: i for i, tstr in enumerate(tcmap) if tstr not in ("6P8",)}
    vdw_frame = all_data[all_data["vdws"] == "von"]

    for comp in comparisons:
        cdict = comparisons[comp]
        column = cdict["column"]

        fig, axs = plt.subplots(
            ncols=2,
            nrows=1,
            sharey=True,
            figsize=(16, 5),
        )
        flat_axs = axs.flatten()
        for ax, tors in zip(flat_axs, ("ton", "toff"), strict=True):
            tor_frame = vdw_frame[vdw_frame["torsions"] == tors]

            for _i, tstr in enumerate(tcmap):
                if tstr in ("6P8",):
                    continue

                topo_frame = tor_frame[tor_frame["topology"] == tstr]

                values = []
                for _i, row in topo_frame.iterrows():
                    name = str(row["index"])
                    if column is None:
                        target = 0
                        target_oppos = None
                    else:
                        target = float(row[column])
                        if "4C1" in name and column == "clangle":
                            target_oppos = convert_pyramid_angle(target)
                        else:
                            target_oppos = None

                    obsdata = geom_data[name][cdict["measure"]]
                    for lbl in cdict["label_options"]:
                        if lbl in obsdata:
                            lbldata = obsdata[lbl]
                            for observed in lbldata:
                                if pd.isna(observed):
                                    continue

                                value = observed - target

                                # If it is possible, this checks if it
                                # matches the opposite angle in
                                # square-pyramid instead.
                                if target_oppos is not None:
                                    ovalue = observed - target_oppos
                                    value = min((abs(ovalue), abs(value)))

                                values.append(value)

                xpos = tcpos[tstr]

                if column is not None:
                    ax.axhline(y=0, lw=1, c="k", linestyle="--")

                if len(values) == 0:
                    continue
                parts = ax.violinplot(
                    list(values),
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

            ax.tick_params(axis="both", which="major", labelsize=16)
            if column is None:
                ax.set_title(
                    f"{convert_tors(tors,num=False)} ",
                    fontsize=16,
                )
                ax.set_ylabel(cdict["xlabel"], fontsize=16)
            else:
                ax.set_title(
                    (
                        f'{cdict["xlabel"]}: '
                        f"{convert_tors(tors,num=False)} "
                    ),
                    fontsize=16,
                )
                ax.set_ylabel(
                    f'observed - target [{cdict["units"]}]',
                    fontsize=16,
                )
            ax.set_xticks([tcpos[i] for i in tcpos])
            ax.set_xticklabels(
                [convert_topo(i) for i in tcpos],
                rotation=45,
            )
            ax.set_xlim(-0.5, 11.5)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"gd_{comp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def single_value_distributions(all_data, figure_output):
    logging.info("running single_value_distributions")

    to_plot = {
        "energy_per_bb": {
            "xtitle": eb_str(),
            "xlim": (0, 50),
        },
        "energy_per_bb_zoom": {
            "xtitle": eb_str(),
            "xlim": (0, isomer_energy() * 3),
        },
        "pore": {"xtitle": pore_str(), "xlim": (0, 10)},
        "min_b2b_distance": {
            "xtitle": r"min. bead-to-bead distance [$\mathrm{\AA}$]",
            "xlim": (0.9, 1.1),
        },
        "HarmonicBondForce_kJ/mol": {
            "xtitle": r"$E_{\mathrm{bond}}$ [kJmol$^{-1}$]",
            "xlim": (0, 3),
        },
        "HarmonicAngleForce_kJ/mol": {
            "xtitle": r"$E_{\mathrm{angle}}$ [kJmol$^{-1}$]",
            "xlim": (0, 400),
        },
        "CustomNonbondedForce_kJ/mol": {
            "xtitle": r"$E_{\mathrm{excl. vol.}}$ [kJmol$^{-1}$]",
            "xlim": (0, 20),
        },
        "PeriodicTorsionForce_kJ/mol": {
            "xtitle": r"$E_{\mathrm{torsion}}$ [kJmol$^{-1}$]",
            "xlim": (0, 15),
        },
        "radius_gyration": {
            "xtitle": rg_str(),
            "xlim": (0, 10),
        },
        "max_diameter": {
            "xtitle": r"$D$ [$\mathrm{\AA}$]",
            "xlim": (0, 30),
        },
        "rg_md": {
            "xtitle": r"$R_{\mathrm{g}}$ / $D$",
            "xlim": (0, 0.6),
        },
        "pore_md": {"xtitle": r"pore size / $D$", "xlim": (0, 0.5)},
        "pore_rg": {
            "xtitle": r"pore size / $R_{\mathrm{g}}$",
            "xlim": (0, 1.2),
        },
    }

    topologies = topology_labels(short="P")
    color_map = {
        ("ton", "von"): "#086788",
        # ("ton", "voff"): "#F9A03F",
        ("toff", "von"): "#0B2027",
        # ("toff", "voff"): "#7A8B99",
    }
    vdw = "von"

    for tp in to_plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        column = "energy_per_bb" if tp == "energy_per_bb_zoom" else tp

        xtitle = to_plot[tp]["xtitle"]
        xlim = to_plot[tp]["xlim"]
        count = 0
        toptions = {}
        for tstr in topologies:
            for tor, vdw in color_map:
                # for vdw in ("voff", "von"):
                color = color_map[(tor, vdw)]
                if tstr == "6P8":
                    continue
                toptions[(tstr, tor, vdw)] = (count, color)
                count += 1

        for i, topt in enumerate(toptions):
            topo_frame = all_data[all_data["topology"] == topt[0]]
            tor_frame = topo_frame[topo_frame["torsions"] == topt[1]]
            fin_frame = tor_frame[tor_frame["vdws"] == topt[2]]

            values = [i for i in fin_frame[column] if not np.isnan(i)]

            if len(values) == 0:
                continue

            xpos = toptions[topt][0]
            col = toptions[topt][1]

            parts = ax.violinplot(
                values,
                [xpos],
                vert=True,
                widths=0.8,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )

            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_edgecolor("none")
                pc.set_alpha(1.0)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel(xtitle, fontsize=16)
        ax.set_ylim(xlim)
        xticks = {}
        all_counts = {}
        for i in toptions:
            tstr = i[0]
            if tstr not in all_counts:
                all_counts[tstr] = []
            all_counts[tstr].append(toptions[i][0])

        xticks = {i: sum(all_counts[i]) / 2 for i in all_counts}
        ylines = [xticks[i] + 1.0 for i in xticks][:-1]

        for yl in ylines:
            ax.axvline(x=yl, c="gray", linestyle="--", alpha=0.4)

        ax.set_xticks([xticks[i] for i in xticks])
        ax.set_xticklabels(
            [convert_topo(i) for i in xticks],
            rotation=45,
        )

        labels = []
        for lblkey in color_map:
            tor, vdw = lblkey
            col = color_map[lblkey]
            labels.append(
                (
                    mpatches.Patch(color=col),
                    f"{convert_tors(tor, num=False)}",
                )
            )

        ax.legend(*zip(*labels, strict=True), fontsize=16, ncols=4)
        ax.set_xlim(-0.5, max(ylines) + 2.0)

        fig.tight_layout()
        tpname = tp.replace("/", "")
        fig.savefig(
            os.path.join(figure_output, f"sing_{tpname}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def plot_sorted(bb_data, color_map, figure_output):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    max_ = 6

    # for ax, cltitle in zip(axs, ("3C1", "4C1")):
    for tor, cltitle in color_map:
        data = bb_data[(tor, cltitle)]
        if cltitle == "3C1":
            flag = "2P3"
        elif cltitle == "4C1":
            flag = "2P4"

        xs = []
        ys = []
        for x in np.linspace(0.01, max_, 100):
            stable_isomers = [
                sum(i[j] < x for j in i) for i in data if flag in i
            ]

            percent_sorted = (
                len([i for i in stable_isomers if i == 1])
                / len(stable_isomers)
            ) * 100
            xs.append(x)
            ys.append(percent_sorted)

        ax.plot(
            xs,
            ys,
            c=color_map[(tor, cltitle)],
            lw=4,
            label=f"{cltitle[0]}C: {convert_tors(tor, num=False)}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"threshold {eb_str()}", fontsize=16)
    ax.set_ylabel(r"% with single species", fontsize=16)
    ax.set_xlim(0, max_ + 0.1)
    ax.set_ylim(0, None)

    ax.axvline(x=isomer_energy(), c="gray", linestyle="--", lw=2)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "sorted_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_mixed_unstable(bb_data, color_map, figure_output):
    fig, ax = plt.subplots(figsize=(8, 5))
    max_ = 3

    # for ax, cltitle in zip(axs, ("3C1", "4C1")):
    for tor, cltitle in color_map:
        if tor == "toff":
            continue
        data = bb_data[(tor, cltitle)]
        if cltitle == "3C1":
            flag = "2P3"
        elif cltitle == "4C1":
            flag = "2P4"

        xs = []
        ys_unstable = []
        ys_mixed = []
        for x in np.linspace(0.01, max_, 100):
            stable_isomers = [
                sum(i[j] < x for j in i) for i in data if flag in i
            ]

            percent_mixed = (
                len([i for i in stable_isomers if i > 1]) / len(stable_isomers)
            ) * 100
            percent_unstable = (
                len([i for i in stable_isomers if i == 0])
                / len(stable_isomers)
            ) * 100

            xs.append(x)
            ys_unstable.append(percent_unstable)
            ys_mixed.append(percent_mixed)

        ax.plot(
            xs,
            ys_unstable,
            c=color_map[(tor, cltitle)],
            lw=3,
            linestyle="-",
            label=(
                f"{cltitle[0]}C: " f"{convert_tors(tor, num=False)}, unstable"
            ),
        )
        ax.plot(
            xs,
            ys_mixed,
            c=color_map[(tor, cltitle)],
            lw=3,
            linestyle="--",
            label=(
                f"{cltitle[0]}C: " f"{convert_tors(tor, num=False)}, mixed"
            ),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"threshold {eb_str()}", fontsize=16)
    ax.set_ylabel(r"% mixed or % unstable", fontsize=16)
    ax.set_xlim(0, max_ + 0.1)
    ax.set_ylim(0, 100)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "mixed_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_clangle(cl_data, color_map, figure_output):
    max_ = 6

    clangles = sorted([int(i) for i in cl_data])
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 5))

    for ax, cltitle in zip(axs, ("3C1", "4C1"), strict=True):
        for clangle in clangles:
            data = cl_data[str(clangle)]
            if cltitle == "3C1":
                flag = "2P3"
            elif cltitle == "4C1":
                flag = "2P4"

            xs = []
            ys = []
            cs = []
            for x in np.linspace(0.01, max_, 100):
                stable_isomers = [
                    sum(i[j] < x for j in i) for i in data if flag in i
                ]
                if len(stable_isomers) == 0:
                    continue

                percent_sorted = (
                    len([i for i in stable_isomers if i == 1])
                    / len(stable_isomers)
                ) * 100

                # if percent_sorted > 5:
                xs.append(x)
                ys.append(clangle)
                cs.append(percent_sorted)

            ax.scatter(
                xs,
                ys,
                c=cs,
                vmin=0,
                vmax=30,
                s=40,
                marker="s",
                cmap="Blues",
            )

        cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
        cmap = mpl.cm.Blues
        norm = mpl.colors.Normalize(vmin=0, vmax=30)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r"% with single species", fontsize=16)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"threshold {eb_str()}", fontsize=16)
        ax.set_ylabel(angle_str(num=int(cltitle[0])), fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "clangled_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_average(bb_data, color_map, figure_output):
    fig, ax = plt.subplots(figsize=(8, 5))
    max_ = 11

    for tor, cltitle in color_map:
        data = bb_data[(tor, cltitle)]
        xs = []
        ys = []
        yerrs = []
        for x in np.linspace(0.01, max_, 100):
            stable_isomers = [sum(i[j] < x for j in i) for i in data]
            mean_stable_isomers = np.mean(stable_isomers)
            std = np.std(stable_isomers)
            xs.append(x)
            ys.append(mean_stable_isomers)
            yerrs.append(std / 2)

        ax.plot(
            xs,
            ys,
            c=color_map[(tor, cltitle)],
            lw=3,
            label=f"{cltitle}: {convert_tors(tor, num=False)}",
        )
        ax.fill_between(
            xs,
            [i - j for i, j in zip(ys, yerrs, strict=True)],
            [i + j for i, j in zip(ys, yerrs, strict=True)],
            facecolor=color_map[(tor, cltitle)],
            alpha=0.2,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"threshold {eb_str()}", fontsize=16)
    ax.set_ylabel(r"mean num. stable isomers", fontsize=16)
    ax.set_xlim(0, max_ + 0.1)
    ax.set_ylim(0, None)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "average_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def mixed_distributions(all_data, figure_output):
    logging.info("running mixed distributions")

    trim = all_data[all_data["vdws"] == "von"]
    color_map = {
        ("ton", "3C1"): "#086788",
        ("ton", "4C1"): "#F9A03F",
        ("toff", "3C1"): "#0B2027",
        ("toff", "4C1"): "#7A8B99",
    }

    cltitles = ("3C1", "4C1")

    bbs = set(trim["bbpair"])
    bb_data = {}
    cl_data = {}
    for bb_pair in bbs:
        if "4C1" in bb_pair and "3C1" in bb_pair:
            continue
        bbd = trim[trim["bbpair"] == bb_pair]

        for cltitle in cltitles:
            data = bbd[bbd["cltitle"] == cltitle]
            if len(data) == 0:
                continue

            tor = set(data["torsions"]).pop()
            energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in data.iterrows()
            }
            clangle = str(set(data["clangle"]).pop())
            if (tor, cltitle) not in bb_data:
                bb_data[(tor, cltitle)] = []
            bb_data[(tor, cltitle)].append(energies)
            if tor == "ton":
                if clangle not in cl_data:
                    cl_data[clangle] = []
                cl_data[clangle].append(energies)

    plot_sorted(bb_data, color_map, figure_output)
    plot_mixed_unstable(bb_data, color_map, figure_output)
    plot_clangle(cl_data, color_map, figure_output)
    plot_average(bb_data, color_map, figure_output)


def plot_vs_2d_distributions(data, color_map, figure_output):
    topologies = [i for i in topology_labels(short="P") if i != "6P8"]

    topology_data = {}
    fig, axs = plt.subplots(
        ncols=4,
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    for ax, tstr in zip(flat_axs, topologies, strict=True):
        tdata = data[data["topology"] == tstr]
        topology_data[tstr] = {}

        for tor in color_map:
            tor_data = tdata[tdata["torsions"] == tor]

            stable_data = tor_data[tor_data["energy_per_bb"] < isomer_energy()]
            bas = stable_data["c2angle"]
            clangles = stable_data["clangle"]
            ax.scatter(
                bas,
                clangles,
                c=color_map[tor][0],
                marker=color_map[tor][1],
                label=convert_tors(tor, num=False),
                s=color_map[tor][2],
                edgecolor=color_map[tor][3],
            )
            topology_data[tstr][tor] = (len(stable_data), len(tor_data))

        ax.set_title(convert_topo(tstr), fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(angle_str(num=2), fontsize=16)
        ax.set_ylabel(angle_str(num=Xc_map(tstr)), fontsize=16)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "flexeffect_biteangle.png"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    return topology_data


def plot_topology_flex(data, figure_output):
    fig, ax = plt.subplots(figsize=(8, 5))

    categories_ton = {convert_topo(i): 0 for i in data}
    categories_toff = {convert_topo(i): 0 for i in data}
    categories_subtracted = {convert_topo(i): 0 for i in data}

    for tstr in data:
        categories_ton[convert_topo(tstr)] = (
            data[tstr]["ton"][0] / data[tstr]["ton"][1]
        ) * 100
        categories_toff[convert_topo(tstr)] = (
            data[tstr]["toff"][0] / data[tstr]["toff"][1]
        ) * 100
        categories_subtracted[convert_topo(tstr)] = (
            categories_toff[convert_topo(tstr)]
            - categories_ton[convert_topo(tstr)]
        ) / categories_toff[convert_topo(tstr)]

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
    ax.axvline(x=4.5, linestyle="--", c="gray", lw=2)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"% accessible cages", fontsize=16)  # , color=color)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=16)
    ax.set_xticks(range(len(categories_toff)))
    ax.set_xticklabels(categories_ton.keys(), rotation=45)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "flexeffect_topologies.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_topology_pore_flex(data, figure_output):
    color_map = {"stable": "#086788", "all": "#0B2027"}

    topologies = [i for i in topology_labels(short="P") if i != "6P8"]

    fig, ax = plt.subplots(figsize=(8, 4))

    xticks = {}
    for i, tstr in enumerate(topologies):
        tdata = data[data["topology"] == tstr]
        ton_data = tdata[tdata["torsions"] == "ton"]
        cage_names = set(ton_data["cage_name"])

        differences = []
        stable_differences = []
        for cage_name in cage_names:
            cdata = ton_data[ton_data["cage_name"] == cage_name]
            pair_name = get_paired_cage_name(cage_name)
            pdata = tdata[tdata["cage_name"] == pair_name]

            ton_energy = float(cdata["energy_per_bb"].iloc[0])
            toff_energy = float(pdata["energy_per_bb"].iloc[0])
            ton_pore = float(cdata["pore"].iloc[0])
            toff_pore = float(pdata["pore"].iloc[0])
            rel_difference = (ton_pore - toff_pore) / toff_pore

            differences.append(rel_difference)
            if ton_energy > isomer_energy() or toff_energy > isomer_energy():
                continue

            stable_differences.append(rel_difference)

        xpos_ = i + 1
        xticks[tstr] = xpos_

        parts = ax.violinplot(
            differences,
            [xpos_ - 0.2],
            vert=True,
            widths=0.3,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color_map["all"])
            pc.set_edgecolor("none")
            pc.set_alpha(1.0)

        if len(stable_differences) > 0:
            parts = ax.violinplot(
                stable_differences,
                [xpos_ + 0.2],
                vert=True,
                widths=0.3,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color_map["stable"])
                pc.set_edgecolor("none")
                pc.set_alpha(1.0)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("rel. difference in pore size", fontsize=16)
    ax.set_ylim(-2, 2)

    ax.axhline(y=0, c="gray", linestyle="--", alpha=0.4)

    ylines = [xticks[i] + 0.5 for i in xticks][:-1]
    for yl in ylines:
        ax.axvline(x=yl, c="gray", linestyle="--", alpha=0.4)

    ax.set_xticks([xticks[i] for i in xticks])
    ax.set_xticklabels(
        [convert_topo(i) for i in xticks],
        rotation=45,
    )

    labels = []
    for lblkey in color_map:
        col = color_map[lblkey]
        labels.append(
            (
                mpatches.Patch(color=col),
                lblkey,
            )
        )
    ax.legend(*zip(*labels), fontsize=16, ncols=4)
    ax.set_xlim(0.5, len(topologies) + 0.5)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "flexeffect_porosites.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def flexeffect_per_property(all_data, figure_output):
    logging.info("running effect of flexibility distributions")

    trim = all_data[all_data["vdws"] == "von"]
    color_map = {
        "toff": ("white", "o", 120, "k"),
        "ton": ("#086788", "o", 80, "none"),
        # "toff", "3C1"): "#0B2027",
        # "toff", "4C1"): "#7A8B99",
    }

    topology_data = plot_vs_2d_distributions(
        data=trim,
        color_map=color_map,
        figure_output=figure_output,
    )
    plot_topology_flex(topology_data, figure_output)
    plot_topology_pore_flex(data=trim, figure_output=figure_output)


def correlation_matrix(all_data, figure_output):
    trim = all_data[all_data["torsions"] == "ton"]

    target_cols = [
        # "topology",
        # "torsions",
        # "vdws",
        # "cltopo",
        # "c2r0",
        # "c2angle",
        "target_bite_angle",
        # "cltitle",
        # "clr0",
        "clangle",
        "energy_per_bb",
        "strain_energy",
        "pore",
        # "min_b2b_distance",
        "radius_gyration",
        "max_diameter",
        "rg_md",
        "pore_md",
        "pore_rg",
        "sv_n_dist",
        "sv_l_dist",
        # "HarmonicBondForce_kjmol",
        # "HarmonicAngleForce_kjmol",
        # "CustomNonbondedForce_kjmol",
        # "PeriodicTorsionForce_kjmol",
        # "structure_dynamics",
        # "pore_dynamics",
        # "node_shape_dynamics",
        # "lig_shape_dynamics",
    ]
    targ_data = trim[target_cols].copy()

    fig, ax = plt.subplots(figsize=(5, 5))
    corr = targ_data.corr()
    ax.matshow(corr, vmin=-1, vmax=1, cmap="Spectral")

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Spectral
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(
        "correlation",
        fontsize=16,
    )

    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.set_xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation="vertical",
    )
    ax.set_yticks(range(len(corr.columns)), corr.columns)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "correlation_matrix.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def col_convert(name):
    return {
        "target_bite_angle": "ba",
        "clangle": "xC",
        "energy_per_bb": "Eb",
        "strain_energy": "Es",
        "HarmonicBondForce_kJ/mol": "HB",
        "HarmonicAngleForce_kJ/mol": "HA",
        "CustomNonbondedForce_kJ/mol": "NB",
        "PeriodicTorsionForce_kJ/mol": "PT",
    }[name]


def energy_correlation_matrix(all_data, figure_output):
    target_cols = [
        "energy_per_bb",
        "HarmonicBondForce_kJ/mol",
        "HarmonicAngleForce_kJ/mol",
        "CustomNonbondedForce_kJ/mol",
        "PeriodicTorsionForce_kJ/mol",
    ]
    trim = all_data[all_data["torsions"] == "ton"]
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
        tdata = trim[trim["topology"] == tstr]
        targ_data = tdata[target_cols].copy()

        corr = targ_data.corr()
        ax.matshow(corr, vmin=-1, vmax=1, cmap="Spectral")

        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.set_title(convert_topo(tstr), fontsize=16)
        ax.set_xticks(
            range(len(corr.columns)),
            [col_convert(i) for i in corr.columns],
            rotation="vertical",
        )
        ax.set_yticks(
            range(len(corr.columns)),
            [col_convert(i) for i in corr.columns],
        )

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Spectral
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(
        "correlation",
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "energy_correlation_matrix.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def get_min_energy_from_omm_out(outfile):
    try:
        with open(outfile) as f:
            min_energy = 1e24
            lines = f.readlines()
            for line in lines:
                if "total energy:" in line:
                    new_energy = float(line.rstrip().split()[2])
                    if new_energy < min_energy:
                        min_energy = new_energy
    except FileNotFoundError:
        min_energy = None

    return min_energy


def check_same_energy(e1, e2):
    if e1 is None or e2 is None:
        return False
    return np.isclose(e1, e2, atol=1e-1, rtol=0)


def opt_outcome_distributions(all_data, figure_output):
    logging.info("running opt_outcome_distributions")

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {
        "opt1": 0,
        "shifted": 0,
        "nearby_opt": 0,
        "smd": 0,
    }
    count1 = all_data["source"].value_counts()

    for source, count in count1.items():
        categories[source] = count

    num_cages = len(all_data)

    ax.bar(
        categories.keys(),
        categories.values(),
        color="teal",
        edgecolor="teal",
        lw=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_title(f"total cages: {num_cages}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "opt_source_counts.pdf"),
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
    opt_data = all_data[all_data["optimised"]]
    logging.info(f"there are {len(opt_data)} successfully opted")

    with open(data_output / "all_geom.json") as f:
        geom_data = json.load(f)

    identity_distributions(all_data, figure_output)
    mixed_distributions(all_data, figure_output)
    single_value_distributions(all_data, figure_output)
    geom_distributions(all_data, geom_data, figure_output)
    flexeffect_per_property(all_data, figure_output)
    correlation_matrix(all_data, figure_output)
    energy_correlation_matrix(all_data, figure_output)
    opt_outcome_distributions(all_data, figure_output)


if __name__ == "__main__":
    main()
