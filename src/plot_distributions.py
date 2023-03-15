#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot distribitions.

Author: Andrew Tarzia

"""

import sys
import os
import stk
import stko
import json
import numpy as np
import logging
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches


from env_set import cages

from utilities import convert_pyramid_angle
from analysis_utilities import (
    write_out_mapping,
    data_to_array,
    convert_topo,
    convert_tors,
    convert_vdws,
    topology_labels,
    target_shapes,
    mapshape_to_topology,
)


def identity_distributions(all_data, figure_output):
    logging.info("running identity_distributions")

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {i: 0 for i in topology_labels(short="+")}
    count1 = all_data["topology"].value_counts()

    for tstr, count in count1.items():
        categories[convert_topo(tstr)] = count

    num_cages = len(all_data)

    ax.bar(
        categories.keys(),
        categories.values(),
        # color="#06AED5",
        color="teal",
        # color="#DD1C1A",
        # color="#320E3B",
        edgecolor="teal",
        lw=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("cage identity", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_title(f"total cages: {num_cages}", fontsize=16)
    # ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "all_bb_dists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def geom_distributions(all_data, geom_data, figure_output):
    logging.info("running geom_distributions")
    # vmax = max_energy()

    comparisons = {
        "torsions": {
            "measure": "dihedrals",
            "xlabel": "torsion [deg]",
            "column": None,
            "label_options": ("Pb_Ba_Ba_Pb",),
        },
        "clangle": {
            "measure": "angles",
            "xlabel": "CL angle [deg]",
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
            "xlabel": "CL bond length [A]",
            "column": "clr0",
            "label_options": ("Pd_Pb", "C_Pb"),
        },
        "c2angle": {
            "measure": "angles",
            "xlabel": "bonder angle [deg]",
            "column": "c2angle",
            "label_options": ("Pb_Ba_Ag",),
        },
        "c2backbone": {
            "measure": "angles",
            "xlabel": "backbone angle [deg]",
            "column": None,
            "label_options": ("Ba_Ag_Ba",),
        },
        "c2r0": {
            "measure": "bonds",
            "xlabel": "C2 backbone length [A]",
            "column": "c2r0",
            "label_options": ("Ba_Ag",),
        },
        "c2bonder": {
            "measure": "bonds",
            "xlabel": "C2 bonder length [A]",
            "column": None,
            "label_options": ("Pb_Ba",),
        },
    }

    tcmap = topology_labels(short="P")
    tcpos = {
        tstr: i for i, tstr in enumerate(tcmap) if tstr not in ("6P8",)
    }

    for comp in comparisons:
        cdict = comparisons[comp]
        column = cdict["column"]

        fig, axs = plt.subplots(
            ncols=2, nrows=2, sharey=True, figsize=(16, 10)
        )
        flat_axs = axs.flatten()
        for ax, (tors, vdws) in zip(
            flat_axs,
            itertools.product(("ton", "toff"), ("von", "voff")),
        ):
            tor_frame = all_data[all_data["torsions"] == tors]
            vdw_frame = tor_frame[tor_frame["vdws"] == vdws]
            for i, tstr in enumerate(tcmap):
                if tstr in ("6P8",):
                    continue

                topo_frame = vdw_frame[vdw_frame["topology"] == tstr]

                values = []
                # energies = []
                for i, row in topo_frame.iterrows():
                    name = str(row["index"])
                    # energy = float(row["energy_per_bb"])
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

                                # For some comparisons, need to use
                                # matching rules.
                                if comp == "clr0":
                                    new_target = (target + 1) / 2
                                    value = observed - new_target
                                elif comp == "c2r0":
                                    new_target = (target + 1) / 2
                                    value = observed - new_target
                                else:
                                    value = observed - target

                                # If it is possible, this checks if it
                                # matches the opposite angle in
                                # square-pyramid instead.
                                if target_oppos is not None:
                                    ovalue = observed - target_oppos
                                    value = min(
                                        (abs(ovalue), abs(value))
                                    )

                                values.append(value)
                                # energies.append(energy)

                xpos = tcpos[tstr]

                if column is not None:
                    ax.axhline(y=0, lw=1, c="k", linestyle="--")

                if len(values) == 0:
                    continue
                parts = ax.violinplot(
                    [i for i in values],
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

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_title(
                (
                    f'{cdict["xlabel"]}: '
                    f"{convert_tors(tors,num=False)}, "
                    f"{convert_vdws(vdws)}"
                ),
                fontsize=16,
            )
            if column is None:
                ax.set_ylabel("observed", fontsize=16)
            else:
                ax.set_ylabel("observed - target", fontsize=16)
            ax.set_xticks([tcpos[i] for i in tcpos])
            ax.set_xticklabels(
                [convert_topo(i) for i in tcpos],
                rotation=45,
            )
            ax.set_xlim(-0.5, 10.5)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"gd_{comp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def rmsd_distributions(all_data, calculation_dir, figure_output):
    logging.info("running rmsd_distributions")
    raise SystemExit("if you want this, need to align MD")

    tcmap = topology_labels(short="P")

    rmsd_file = calculation_dir / "all_rmsds.json"
    if os.path.exists(rmsd_file):
        with open(rmsd_file, "r") as f:
            data = json.load(f)
    else:
        data = {}
        for tstr in tcmap:
            tdata = {}
            if tstr in ("6P8",):
                search = f"*{tstr}_*toff*von*_final.mol"
            else:
                search = f"*{tstr}_*ton*von*_final.mol"

            for o1 in calculation_dir.glob(search):
                o1 = str(o1)
                o2 = o1.replace("final", "opted1")
                o3 = o1.replace("final", "opted2")
                o4 = o1.replace("final", "opted3")
                m1 = stk.BuildingBlock.init_from_file(o1)
                m2 = stk.BuildingBlock.init_from_file(o2)
                m3 = stk.BuildingBlock.init_from_file(o3)
                m4 = stk.BuildingBlock.init_from_file(o4)

                rmsd_calc = stko.RmsdCalculator(m1)

                if tstr in ("6P8",):
                    otoff_voff = o1.replace("von", "voff")
                    mtoffvoff = stk.BuildingBlock.init_from_file(
                        otoff_voff
                    )
                    tdata[o1] = {
                        "r2": rmsd_calc.get_results(m2).get_rmsd(),
                        "r3": rmsd_calc.get_results(m3).get_rmsd(),
                        "r4": rmsd_calc.get_results(m4).get_rmsd(),
                        "rtoffvon": None,
                        "rtonvoff": None,
                        "rtoffvoff": rmsd_calc.get_results(
                            mtoffvoff
                        ).get_rmsd(),
                    }
                else:
                    otoff_von = o1.replace("ton", "toff")
                    oton_voff = o1.replace("von", "voff")
                    otoff_voff = otoff_von.replace("von", "voff")
                    mtoffvon = stk.BuildingBlock.init_from_file(
                        otoff_von
                    )
                    mtonvoff = stk.BuildingBlock.init_from_file(
                        oton_voff
                    )
                    mtoffvoff = stk.BuildingBlock.init_from_file(
                        otoff_voff
                    )
                    tdata[o1] = {
                        "r2": rmsd_calc.get_results(m2).get_rmsd(),
                        "r3": rmsd_calc.get_results(m3).get_rmsd(),
                        "r4": rmsd_calc.get_results(m4).get_rmsd(),
                        "rtoffvon": rmsd_calc.get_results(
                            mtoffvon
                        ).get_rmsd(),
                        "rtonvoff": rmsd_calc.get_results(
                            mtonvoff
                        ).get_rmsd(),
                        "rtoffvoff": rmsd_calc.get_results(
                            mtoffvoff
                        ).get_rmsd(),
                    }
            data[tstr] = tdata

        with open(rmsd_file, "w") as f:
            json.dump(data, f, indent=4)

    tcpos = {tstr: i for i, tstr in enumerate(tcmap)}

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(8, 10),
    )
    flat_axs = axs.flatten()

    for ax, col in zip(
        flat_axs,
        ("rtoffvon", "rtonvoff", "rtoffvoff"),
    ):

        for tstr in tcmap:
            if tstr in ("6P8",) and col in ("rtoffvon", "rtonvoff"):
                continue

            ydata = [i[col] for i in data[tstr].values()]
            xpos = tcpos[tstr]
            parts = ax.violinplot(
                [i for i in ydata],
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
        ax.set_ylabel("RMSD [A]", fontsize=16)
        ax.set_xlim(-0.5, 11.5)
        ax.set_title(f"{col}", fontsize=16)
        ax.set_xticks([tcpos[i] for i in tcpos])
        ax.set_xticklabels(
            [convert_topo(i) for i in tcpos],
            rotation=45,
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "all_rmsds.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def single_value_distributions(all_data, figure_output):
    logging.info("running single_value_distributions")

    to_plot = {
        "energy_per_bb": {
            "xtitle": "E_b [kJmol-1]",
            "xlim": (0, 200),
        },
        "pore": {"xtitle": "min. distance [A]", "xlim": (0, 15)},
        "min_b2b_distance": {
            "xtitle": "min. b2b distance [A]",
            "xlim": (0, 1.2),
        },
        "HarmonicBondForce_kjmol": {
            "xtitle": "E_HB [kJmol-1]",
            "xlim": (0, 40),
        },
        "HarmonicAngleForce_kjmol": {
            "xtitle": "E_HA [kJmol-1]",
            "xlim": (0, 1000),
        },
        "CustomNonbondedForce_kjmol": {
            "xtitle": "E_NB [kJmol-1]",
            "xlim": (0, 400),
        },
        "PeriodicTorsionForce_kjmol": {
            "xtitle": "E_PT [kJmol-1]",
            "xlim": (0, 400),
        },
        "radius_gyration": {
            "xtitle": "R_g [A]",
            "xlim": (0, 20),
        },
        "max_diameter": {
            "xtitle": "$D$ [A]",
            "xlim": (0, 50),
        },
        "rg_md": {"xtitle": "R_g / $D$", "xlim": (0, 0.6)},
        "pore_md": {"xtitle": "min. distance / $D$", "xlim": (0, 0.5)},
        "pore_rg": {"xtitle": "min. distance / R_g", "xlim": (0, 1.2)},
        "structure_dynamics": {
            "xtitle": r"$\sigma$($R_g$)/average [unitless]",
            "xlim": (0, 1),
        },
        "pore_dynamics": {
            "xtitle": r"$\sigma$(pore measure)/average [unitless]",
            "xlim": (0, 5),
        },
        "node_shape_dynamics": {
            "xtitle": r"$\sigma$(node shape cos.sim.) [unitless]",
            "xlim": (0, 0.5),
        },
        "lig_shape_dynamics": {
            "xtitle": r"$\sigma$(lig shape cos.sim.) [unitless]",
            "xlim": (0, 0.5),
        },
    }

    topologies = topology_labels(short="P")
    color_map = {
        ("ton", "von"): "#086788",
        # ("ton", "voff"): "#F9A03F",
        # ("toff", "von"): "#0B2027",
        # ("toff", "voff"): "#7A8B99",
    }
    tor = "ton"
    vdw = "von"

    for tp in to_plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        xtitle = to_plot[tp]["xtitle"]
        xlim = to_plot[tp]["xlim"]
        count = 0
        toptions = {}
        for tstr in topologies:
            # for tor in ("toff", "ton"):
            #     for vdw in ("voff", "von"):
            color = color_map[(tor, vdw)]
            if tstr == "6P8":
                continue
            toptions[(tstr, tor, vdw)] = (count, color)
            count += 1

        for i, topt in enumerate(toptions):
            topo_frame = all_data[all_data["topology"] == topt[0]]
            tor_frame = topo_frame[topo_frame["torsions"] == topt[1]]
            fin_frame = tor_frame[tor_frame["vdws"] == topt[2]]

            values = [i for i in fin_frame[tp] if not np.isnan(i)]

            if len(values) == 0:
                continue

            xpos = toptions[topt][0]
            col = toptions[topt][1]

            # ax.scatter(
            #     [xpos for i in values],
            #     [i for i in values],
            #     c=col,
            #     edgecolor="none",
            #     s=30,
            #     alpha=0.2,
            #     rasterized=True,
            # )

            parts = ax.violinplot(
                values,
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

        xticks = {i: sum(all_counts[i]) / 1 for i in all_counts}
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
            tor, vdw = lblkey
            col = color_map[lblkey]
            labels.append(
                (
                    mpatches.Patch(color=col),
                    f"{convert_tors(tor)}, {convert_vdws(vdw)}",
                )
            )
            # ax.scatter(
            #     None,
            #     None,
            #     c=col,
            #     edgecolor="none",
            #     s=30,
            #     alpha=0.2,
            #     label=,
            # )
        # ax.legend(*zip(*labels), fontsize=16, ncols=4)
        ax.set_xlim(-0.5, max(ylines) + 1.0)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"sing_{tp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def plot_sorted(bb_data, color_map, figure_output):

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    max_ = 51

    for ax, cltitle in zip(axs, ("3C1", "4C1")):
        for tor, vdw in color_map:
            data = bb_data[(tor, vdw)]
            if cltitle == "3C1":
                flag = "2P3"
            elif cltitle == "4C1":
                flag = "2P4"

            xs = []
            ys = []
            for x in range(1, max_):
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
                marker="o",
                c=color_map[(tor, vdw)],
                lw=2,
                label=f"{convert_tors(tor)}; {convert_vdws(vdw)}",
            )

        ax.set_title(f"{cltitle}", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("threshold [kJmol-1]", fontsize=16)
        ax.set_ylabel(r"% sorted", fontsize=16)
        ax.set_xlim(0, max_)
        ax.set_ylim(0, 50)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "sorted_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_mixed(bb_data, color_map, figure_output):

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    max_ = 51

    for ax, cltitle in zip(axs, ("3C1", "4C1")):
        for tor, vdw in color_map:
            data = bb_data[(tor, vdw)]
            if cltitle == "3C1":
                flag = "2P3"
            elif cltitle == "4C1":
                flag = "2P4"

            xs = []
            ys = []
            for x in range(1, max_):
                stable_isomers = [
                    sum(i[j] < x for j in i) for i in data if flag in i
                ]

                percent_mixed = (
                    len([i for i in stable_isomers if i > 1])
                    / len(stable_isomers)
                ) * 100
                xs.append(x)
                ys.append(percent_mixed)

            ax.plot(
                xs,
                ys,
                marker="o",
                c=color_map[(tor, vdw)],
                lw=2,
                label=f"{convert_tors(tor)}; {convert_vdws(vdw)}",
            )

        ax.set_title(f"{cltitle}", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("threshold [kJmol-1]", fontsize=16)
        ax.set_ylabel(r"% mixed", fontsize=16)
        ax.set_xlim(0, max_)
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

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    max_ = 51

    clangles = sorted([int(i) for i in cl_data.keys()])

    for ax, cltitle in zip(axs, ("3C1", "4C1")):
        for clangle in clangles:
            data = cl_data[str(clangle)]
            if cltitle == "3C1":
                flag = "2P3"
            elif cltitle == "4C1":
                flag = "2P4"

            xs = []
            ys = []
            for x in range(1, max_):
                stable_isomers = [
                    sum(i[j] < x for j in i) for i in data if flag in i
                ]
                if len(stable_isomers) == 0:
                    continue

                percent_sorted = (
                    len([i for i in stable_isomers if i == 1])
                    / len(stable_isomers)
                ) * 100

                xs.append(x)
                ys.append(percent_sorted)

            ax.plot(
                xs,
                ys,
                marker="o",
                # c=color_map[(tor, vdw)],
                lw=2,
                label=f"{clangle}",
            )

        ax.set_title(f"{cltitle}", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("threshold [kJmol-1]", fontsize=16)
        ax.set_ylabel(r"% sorted", fontsize=16)
        ax.set_xlim(0, max_)
        ax.set_ylim(0, 60)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "clangled_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_unstable(bb_data, color_map, figure_output):

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    max_ = 51

    for ax, cltitle in zip(axs, ("3C1", "4C1")):
        for tor, vdw in color_map:
            data = bb_data[(tor, vdw)]
            if cltitle == "3C1":
                flag = "2P3"
            elif cltitle == "4C1":
                flag = "2P4"

            xs = []
            ys = []
            for x in range(1, max_):
                stable_isomers = [
                    sum(i[j] < x for j in i) for i in data if flag in i
                ]

                percent_unstable = (
                    len([i for i in stable_isomers if i == 0])
                    / len(stable_isomers)
                ) * 100
                xs.append(x)
                ys.append(percent_unstable)

            ax.plot(
                xs,
                ys,
                marker="o",
                c=color_map[(tor, vdw)],
                lw=2,
                label=f"{convert_tors(tor)}; {convert_vdws(vdw)}",
            )

        ax.set_title(f"{cltitle}", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("threshold [kJmol-1]", fontsize=16)
        ax.set_ylabel(r"% unstable", fontsize=16)
        ax.set_xlim(0, max_)
        ax.set_ylim(0, 100)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "unstable_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_average(bb_data, color_map, figure_output):

    fig, ax = plt.subplots(figsize=(8, 5))
    max_ = 51

    for tor, vdw in color_map:
        data = bb_data[(tor, vdw)]
        xs = []
        ys = []
        yerrs = []
        for x in range(1, max_):
            stable_isomers = [sum(i[j] < x for j in i) for i in data]
            mean_stable_isomers = np.mean(stable_isomers)
            std = np.std(stable_isomers)
            xs.append(x)
            ys.append(mean_stable_isomers)
            yerrs.append(std / 2)

        ax.plot(
            xs,
            ys,
            marker="o",
            c=color_map[(tor, vdw)],
            lw=2,
            label=f"{convert_tors(tor)}; {convert_vdws(vdw)}",
        )
        ax.fill_between(
            xs,
            [i - j for i, j in zip(ys, yerrs)],
            [i + j for i, j in zip(ys, yerrs)],
            facecolor=color_map[(tor, vdw)],
            alpha=0.3,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("threshold [kJmol-1]", fontsize=16)
    ax.set_ylabel("mean stable isomers", fontsize=16)
    ax.set_xlim(0, max_)
    ax.set_ylim(0, 7)

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "average_isomers.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def mixed_distributions(all_data, figure_output, trim_data):
    logging.info("running mixed distributions")

    trim = all_data[all_data["torsions"] == "ton"]
    trim = trim[trim["vdws"] == "von"]
    if trim_data:
        color_map = {
            ("ton", "von"): "#086788",
        }
    else:
        color_map = {
            ("ton", "von"): "#086788",
            ("ton", "voff"): "#F9A03F",
            ("toff", "von"): "#0B2027",
            ("toff", "voff"): "#7A8B99",
        }

    # trim = all_data[all_data["clr0"] == 5]
    # trim = trim[trim["c2r0"] == 5]
    # trim = trim[trim["c2r0"] == 5]

    bbs = set(trim["bbpair"])
    bb_data = {}
    cl_data = {}
    for bb_pair in bbs:
        if "4C1" in bb_pair and "3C1" in bb_pair:
            continue
        bbd = trim[trim["bbpair"] == bb_pair]

        for tor, vdw in color_map:
            data = bbd[bbd["vdws"] == vdw]
            data = data[data["torsions"] == tor]
            energies = {
                str(row["topology"]): float(row["energy_per_bond"])
                for i, row in data.iterrows()
            }
            clangle = str(set(data["clangle"]).pop())
            if (tor, vdw) not in bb_data:
                bb_data[(tor, vdw)] = []
            bb_data[(tor, vdw)].append(energies)
            if tor == "ton" and vdw == "von":
                if clangle not in cl_data:
                    cl_data[clangle] = []
                cl_data[clangle].append(energies)

    plot_sorted(bb_data, color_map, figure_output)
    plot_mixed(bb_data, color_map, figure_output)
    plot_clangle(cl_data, color_map, figure_output)
    plot_average(bb_data, color_map, figure_output)
    plot_unstable(bb_data, color_map, figure_output)


def shape_vector_distributions(all_data, figure_output):
    logging.info("running shape_vector_distributions")

    trim = all_data[all_data["torsions"] == "ton"]
    trim = trim[trim["vdws"] == "von"]

    present_shape_values = target_shapes()
    num_cols = 2
    num_rows = 5

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(16, 16),
    )
    flat_axs = axs.flatten()

    for i, shape in enumerate(present_shape_values):
        # nv = int(shape.split("-")[1])
        # c = color_map[nv]
        ax = flat_axs[i]

        keys = tuple(trim.columns)
        nshape = f"n_{shape}"
        lshape = f"l_{shape}"

        # for tors in torsion_dict:
        #     tor_data = trim[all_data["torsions"] == tors]
        if nshape in keys:
            filt_data = trim[trim[nshape].notna()]
            n_values = list(filt_data[nshape])
            ax.hist(
                x=n_values,
                bins=50,
                density=False,
                histtype="step",
                color="#DD1C1A",
                lw=3,
                # linestyle=torsion_dict[tors],
                # label=f"node: {convert_tors(tors, num=False)}",
                label="node",
            )

        if lshape in keys:
            filt_data = trim[trim[lshape].notna()]
            l_values = list(filt_data[lshape])
            ax.hist(
                x=l_values,
                bins=50,
                density=False,
                histtype="step",
                color="k",
                lw=2,
                # linestyle=torsion_dict[tors],
                # label=f"ligand: {convert_tors(tors, num=False)}",
                label="ligand",
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(shape, fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        # ax.set_xlim(0, xmax)
        # ax.set_yscale("log")

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_vectors.pdf"),
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


def correlation_matrix(all_data, figure_output):
    trim = all_data[all_data["torsions"] == "ton"]
    trim = trim[trim["vdws"] == "von"]
    trim = trim[trim["clr0"] == 5]
    trim = trim[trim["c2r0"] == 5]

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
        "energy_per_bond",
        "pore",
        # "min_b2b_distance",
        "radius_gyration",
        # "max_diameter",
        # "rg_md",
        # "pore_md",
        # "pore_rg",
        "sv_n_dist",
        "sv_l_dist",
        # "HarmonicBondForce_kjmol",
        # "HarmonicAngleForce_kjmol",
        # "CustomNonbondedForce_kjmol",
        # "PeriodicTorsionForce_kjmol",
        "structure_dynamics",
        "pore_dynamics",
        "node_shape_dynamics",
        "lig_shape_dynamics",
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
    opt_data = all_data[all_data["optimised"]]
    logging.info(f"there are {len(opt_data)} successfully opted")
    _data = all_data[all_data["mdfailed"]]
    logging.info(f"there are {len(_data)} with failed MD")
    _data = all_data[all_data["mdexploded"]]
    logging.info(f"there are {len(_data)} with exploded MD")
    write_out_mapping(all_data)

    correlation_matrix(all_data, figure_output)
    identity_distributions(all_data, figure_output)
    mixed_distributions(all_data, figure_output, trim_data=True)
    single_value_distributions(all_data, figure_output)
    shape_vector_distributions(all_data, figure_output)
    shape_vectors_2(all_data, figure_output)
    shape_vectors_3(all_data, figure_output)
    geom_distributions(all_data, geom_data, figure_output)
    rmsd_distributions(all_data, calculation_output, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
