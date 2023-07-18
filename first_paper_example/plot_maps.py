#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot property, input maps.

Author: Andrew Tarzia

"""

import sys
import os
import itertools
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from env_set import figures, calculations, outputdata

from analysis import (
    data_to_array,
    stoich_map,
    eb_str,
    get_lowest_energy_data,
    isomer_energy,
    topology_labels,
    cltypetopo_to_colormap,
    write_out_mapping,
    convert_tors,
    convert_topo,
    clangle_str,
    Xc_map,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def nobiteangle_relationship(all_data, figure_output):
    logging.info("running clangle_relationship")
    # ensemble = Ensemble(
    #     base_molecule=molecule,
    #     base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
    #     conformer_xyz=os.path.join(
    #         output_dir, f"{name}_ensemble.xyz"
    #     ),
    #     data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
    #     overwrite=False,
    # )
    raise SystemExit("I want a plot with all energies on it.")
    trim = all_data[all_data["vdws"] == "von"]

    for tstr in topology_labels(short="P"):
        if tstr != "6P8":
            continue
        filt_data = trim[trim["topology"] == tstr]
        clangles = sorted(set(filt_data["clangle"]))
        if len(clangles) == 0:
            continue

        fig, axs = plt.subplots(
            nrows=len(clangles),
            sharex=True,
            sharey=True,
            figsize=(8, 12),
        )
        for ax, t_angle in zip(axs, clangles):
            clan_data = filt_data[filt_data["clangle"] == t_angle]
            clan_output = {}
            for run_number in clan_data["run_number"]:
                plot_data = clan_data[
                    clan_data["run_number"] == run_number
                ]
                if len(plot_data) == 0:
                    continue
                xs = list(plot_data["c3angle"])
                ys = list(plot_data["energy_per_bb"])
                xs, ys = zip(*sorted(zip(xs, ys)))
                clan_output[run_number] = {x: y for x, y in zip(xs, ys)}

            if len(clan_output) == 0:
                continue

            xys = []
            y_mins = []
            for x in xs:
                y_list = []
                for run_num in clan_output:
                    if x in clan_output[run_num]:
                        xys.append((x, clan_output[run_num][x]))
                        y_list.append(clan_output[run_num][x])
                y_mins.append(min(y_list))

            ax.scatter(
                [i[0] for i in xys],
                [i[1] for i in xys],
                marker="o",
                c="#086788",
                s=40,
                edgecolor="none",
                zorder=2,
            )
            ax.plot(
                xs,
                y_mins,
                # zs=t_angle,
                lw=1,
                c="#086788",
                alpha=1.0,
                linestyle="--",
                # label=f"C2 r0 {c1_opt}/CL r0 {c2_opt}",
                # marker="o",
            )

            ax.set_ylabel(eb_str(), fontsize=16)
            ax.set_title(
                f"{clangle_str(num=4)} = {t_angle}",
                fontsize=16,
            )
            ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(clangle_str(num=3), fontsize=16)
        ax.set_ylim(0, 10)

        fig.tight_layout()
        filename = f"cr_{tstr}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def bite_angle_relationship(all_data, figure_output):
    logging.info("running bite_angle_relationship")

    trim = all_data[all_data["vdws"] == "von"]

    cmap = {
        "toff": ("#F9A03F", "o", "-"),
        "ton": ("#086788", "o", "-"),
    }

    for tstr in topology_labels(short="P"):
        if tstr == "6P8":
            continue
        filt_data = trim[trim["topology"] == tstr]
        clangles = sorted(set(filt_data["clangle"]))
        if len(clangles) == 0:
            continue
        fig, axs = plt.subplots(
            nrows=len(clangles),
            sharex=True,
            sharey=True,
            figsize=(8, 12),
        )

        for ax, t_angle in zip(axs, clangles):
            clan_data = filt_data[filt_data["clangle"] == t_angle]
            for tors in cmap:
                tor_data = clan_data[clan_data["torsions"] == tors]
                tors_output = {}
                # for c1_opt in sorted(set(clan_data["c2r0"])):
                for run_number in clan_data["run_number"]:
                    plot_data = tor_data[
                        tor_data["run_number"] == run_number
                    ]
                    if len(plot_data) == 0:
                        continue
                    # for c2_opt in sorted(set(test_data["clr0"])):
                    # plot_data = test_data[
                    #     test_data["clr0"] == c2_opt
                    # ]
                    xs = list(plot_data["c2angle"])
                    ys = list(plot_data["energy_per_bb"])
                    xs, ys = zip(*sorted(zip(xs, ys)))
                    tors_output[run_number] = {
                        x: y for x, y in zip(xs, ys)
                    }

                if len(tors_output) == 0:
                    continue

                y_mins = []
                xys = []
                for x in xs:
                    y_list = []
                    for run_num in tors_output:
                        if x in tors_output[run_num]:
                            y_list.append(tors_output[run_num][x])
                            xys.append((x, tors_output[run_num][x]))
                    y_mins.append(min(y_list))

                ax.scatter(
                    [i[0] for i in xys],
                    [i[1] for i in xys],
                    marker="o",
                    c=cmap[tors][0],
                    s=40,
                    edgecolor="none",
                    zorder=2,
                )
                ax.plot(
                    xs,
                    y_mins,
                    # zs=t_angle,
                    lw=1,
                    c=cmap[tors][0],
                    alpha=1.0,
                    linestyle="--",
                    # label=f"C2 r0 {c1_opt}/CL r0 {c2_opt}",
                    label=f"{convert_tors(tors,num=False)}",
                    marker=cmap[tors][1],
                )

            ax.set_title(
                f"{clangle_str(num=Xc_map(tstr))} = {t_angle}",
                fontsize=16,
            )
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_ylabel(eb_str(), fontsize=16)
        ax.set_xlabel(r"target internal angle [$^\circ$]", fontsize=16)
        ax.legend(ncol=2, fontsize=16)
        ax.set_ylim(0, 5)
        ax.set_xlim(89, 181)
        # ax.set_yscale("log")

        fig.tight_layout()
        filename = f"ar_{tstr}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def selectivity_map(all_data, figure_output):
    logging.info("running selectivity_map")

    bite_angles = sorted(
        [
            float(i)
            for i in set(all_data["target_bite_angle"])
            if not math.isnan(i)
        ]
    )
    clangles = sorted([float(i) for i in set(all_data["clangle"])])

    properties = {
        "energy": {
            "col": "energy_per_bb",
            "cut": isomer_energy(),
            "dir": "<",
            "clbl": eb_str(),
        },
        # "pore": {"col": "pore", "cut": min_radius() * 2, "dir": "<"},
        # "min_b2b": {
        #     "col": "min_b2b",
        #     "cut": min_b2b_distance(),
        #     "dir": "<",
        # },
        # "sv_n_dist": {"col": "sv_n_dist", "cut": 0.9, "dir": ">"},
        # "sv_l_dist": {"col": "sv_l_dist", "cut": 0.9, "dir": ">"},
    }
    for prop in properties:
        pdict = properties[prop]

        count = 0
        topology_order = {}
        tset = set(all_data["topology"])
        for tstr in topology_labels(short="P"):
            if tstr in ("6P8",):
                continue
            if tstr in tset:
                topology_order[tstr] = count
                count += 1

        fig, axs = plt.subplots(
            ncols=len(clangles),
            nrows=2,
            sharex=True,
            sharey=True,
            figsize=(16, 10),
        )

        for r_axs, tors in zip(axs, ("ton", "toff")):
            tordata = all_data[all_data["torsions"] == tors]
            for clangle, ax in zip(clangles, r_axs):
                cdata = tordata[tordata["clangle"] == clangle]
                for tstr in topology_order:
                    xvalues = []
                    yvalues = []
                    cvalues = []
                    yvalue = topology_order[tstr]
                    tdata = cdata[cdata["topology"] == tstr]
                    for ba in bite_angles:
                        plotdata = tdata[
                            tdata["target_bite_angle"] == ba
                        ]
                        # total_count = len(plotdata)

                        property_list = list(plotdata[pdict["col"]])
                        if pdict["dir"] == "<":
                            under_cut = [
                                i
                                for i in property_list
                                if i < pdict["cut"]
                            ]
                            # order_string = "<"
                        elif pdict["dir"] == ">":
                            under_cut = [
                                i
                                for i in property_list
                                if i > pdict["cut"]
                            ]
                            # order_string = ">"

                        under_count = len(under_cut)
                        xvalue = ba
                        xvalues.append(xvalue)
                        yvalues.append(yvalue)
                        if under_count == 0:  # or total_count == 0:
                            cvalues.append("white")
                        else:
                            cvalues.append("k")
                            #     'under_count'
                            # )  # / total_count)

                    ax.scatter(
                        xvalues,
                        yvalues,
                        # c=under_count / total_count,
                        c=cvalues,
                        # vmin=0,
                        # vmax=vcount,
                        alpha=1.0,
                        edgecolor="none",
                        s=40,
                        marker="s",
                        # cmap="Blues",
                    )

                ax.set_title(
                    clangle,
                    fontsize=16,
                )
                ax.tick_params(axis="both", which="major", labelsize=16)

                ax.set_yticks(
                    [topology_order[i] for i in topology_order]
                )
                ax.set_yticklabels(
                    [convert_topo(i) for i in topology_order]
                )

            ax.set_xlabel("bite angle [deg]", fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"sel_{prop}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def draw_pie(colours, xpos, ypos, size, ax):
    """
    From:
    https://stackoverflow.com/questions/56337732/how-to-plot-scatter-
    pie-chart-using-matplotlib

    """

    num_points = len(colours)
    if num_points == 1:
        ax.scatter(
            xpos,
            ypos,
            c=colours[0],
            edgecolors="k",
            s=size,
        )
    else:
        ratios = [1 / num_points for i in range(num_points)]
        assert sum(ratios) <= 1, "sum of ratios needs to be < 1"

        markers = []
        previous = 0
        # calculate the points of the pie pieces
        for color, ratio in zip(colours, ratios):
            this = 2 * np.pi * ratio + previous
            x = (
                [0]
                + np.cos(np.linspace(previous, this, 100)).tolist()
                + [0]
            )
            y = (
                [0]
                + np.sin(np.linspace(previous, this, 100)).tolist()
                + [0]
            )
            xy = np.column_stack([x, y])
            previous = this
            markers.append(
                {
                    "marker": xy,
                    "s": np.abs(xy).max() ** 2 * np.array(size),
                    "facecolor": color,
                    "edgecolors": "k",
                }
            )

        # scatter each of the pie pieces to create pies
        for marker in markers:
            ax.scatter(xpos, ypos, **marker)


def selfsort_legend(all_data, figure_output):
    logging.info("running selfsort_legend")

    for cltitle in ("3C1", "4C1"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in cltypetopo_to_colormap():
            if i not in (cltitle, "unstable"):
                continue
            for j in cltypetopo_to_colormap()[i]:
                if i == "unstable":
                    label = "unstable"
                else:
                    label = convert_topo(j)
                ax.scatter(
                    None,
                    None,
                    c=cltypetopo_to_colormap()[i][j],
                    edgecolor="k",
                    s=400,
                    marker="o",
                    alpha=1.0,
                    label=label,
                )

        ax.legend(
            # ncol=5,
            fontsize=16,
        )

        fig.tight_layout()
        filename = f"ss_{cltitle}_legend.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def selfsort_map(all_data, figure_output):
    logging.info("running selfsort_map")

    cols_to_map = ["clangle", "c2angle"]
    cols_to_iter = [
        "torsions",
        "vdws",
        "cltitle",
    ]

    io1 = sorted(set(all_data[cols_to_iter[0]]))
    io2 = sorted(set(all_data[cols_to_iter[1]]))
    io3 = sorted(set(all_data[cols_to_iter[2]]))
    for tor, vdw, cltitle in itertools.product(io1, io2, io3):

        data = all_data[all_data[cols_to_iter[0]] == tor]
        data = data[data[cols_to_iter[1]] == vdw]
        data = data[data[cols_to_iter[2]] == cltitle]
        fig, ax = plt.subplots(figsize=(8, 5))

        uo1 = sorted(set(data[cols_to_map[0]]))
        uo2 = sorted(set(data[cols_to_map[1]]))

        for (i, cla), (j, ba) in itertools.product(
            enumerate(uo1), enumerate(uo2)
        ):
            plot_data = data[data[cols_to_map[0]] == cla]
            plot_data = plot_data[plot_data[cols_to_map[1]] == ba]

            xvalue = ba
            yvalue = cla

            if len(plot_data) == 0:
                continue

            energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in plot_data.iterrows()
            }

            mixed_energies = {
                i: energies[i]
                for i in energies
                if energies[i] < isomer_energy()
            }

            min_energy = min(energies.values())

            if min_energy > isomer_energy():
                colours = ["white"]
            # elif len(mixed_energies) == 0:
            #     topo_str = list(energies.keys())[
            #         list(energies.values()).index(min_energy)
            #     ]
            #     colours = [cltypetopo_to_colormap()[cltitle][topo_str]]
            #     dists = [1]

            else:
                colours = sorted(
                    [
                        cltypetopo_to_colormap()[cltitle][i]
                        for i in mixed_energies
                    ]
                )

            draw_pie(
                colours=colours,
                xpos=xvalue,
                ypos=yvalue,
                size=400,
                ax=ax,
            )

            # rect = ax.patch
            # rect.set_alpha(0)

            ax.set_title(convert_tors(tor, num=False), fontsize=16)
            ax.set_ylabel(clangle_str(num=int(cltitle[0])), fontsize=16)
            ax.set_xlabel(
                r"target internal angle [$^\circ$]", fontsize=16
            )
            ax.tick_params(axis="both", which="major", labelsize=16)

        # for i in cltypetopo_to_colormap():
        #     if i not in (cltitle, "unstable"):
        #         continue
        #     for j in cltypetopo_to_colormap()[i]:
        #         if i == "unstable":
        #             label = "unstable"
        #         else:
        #             label = convert_topo(j)
        #         ax.scatter(
        #             None,
        #             None,
        #             c=cltypetopo_to_colormap()[i][j],
        #             edgecolor="k",
        #             s=400,
        #             marker="o",
        #             alpha=1.0,
        #             label=label,
        #         )

        # fig.legend(
        #     bbox_to_anchor=(0, 1.02, 2, 0.2),
        #     loc="lower left",
        #     ncol=5,
        #     fontsize=16,
        # )

        fig.tight_layout()
        filename = f"ss_{cltitle}_{tor}_{vdw}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def kinetic_selfsort_map(all_data, figure_output):
    logging.info("running kinetic_selfsort_map")

    cols_to_map = ["clangle", "c2angle"]
    cols_to_iter = [
        "torsions",
        "vdws",
        "cltitle",
    ]

    io1 = sorted(set(all_data[cols_to_iter[0]]))
    io2 = sorted(set(all_data[cols_to_iter[1]]))
    io3 = sorted(set(all_data[cols_to_iter[2]]))
    for tor, vdw, cltitle in itertools.product(io1, io2, io3):

        data = all_data[all_data[cols_to_iter[0]] == tor]
        data = data[data[cols_to_iter[1]] == vdw]
        data = data[data[cols_to_iter[2]] == cltitle]
        fig, ax = plt.subplots(figsize=(8, 5))

        uo1 = sorted(set(data[cols_to_map[0]]))
        uo2 = sorted(set(data[cols_to_map[1]]))

        for (i, cla), (j, ba) in itertools.product(
            enumerate(uo1), enumerate(uo2)
        ):
            plot_data = data[data[cols_to_map[0]] == cla]
            plot_data = plot_data[plot_data[cols_to_map[1]] == ba]

            xvalue = ba
            yvalue = cla

            if len(plot_data) == 0:
                continue

            energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in plot_data.iterrows()
            }

            mixed_energies = {
                i: energies[i]
                for i in energies
                if energies[i] < isomer_energy()
            }
            if len(mixed_energies) == 0:
                colours = ["white"]
            else:
                stoichiometries = {
                    i: stoich_map(i) for i in mixed_energies
                }
                min_stoichiometry = min(stoichiometries.values())
                kinetic_energies = {
                    i: mixed_energies[i]
                    for i in stoichiometries
                    if stoichiometries[i] == min_stoichiometry
                }

                if len(kinetic_energies) > 1:
                    colours = sorted(
                        [
                            cltypetopo_to_colormap()[cltitle][i]
                            for i in kinetic_energies
                        ]
                    )
                else:
                    colours = [
                        cltypetopo_to_colormap()[cltitle][
                            list(kinetic_energies.keys())[0]
                        ]
                    ]

            draw_pie(
                colours=colours,
                xpos=xvalue,
                ypos=yvalue,
                size=400,
                ax=ax,
            )

            ax.set_title(convert_tors(tor, num=False), fontsize=16)
            ax.set_ylabel(clangle_str(num=int(cltitle[0])), fontsize=16)
            ax.set_xlabel(
                r"target internal angle [$^\circ$]",
                fontsize=16,
            )
            ax.tick_params(axis="both", which="major", labelsize=16)

        fig.tight_layout()
        filename = f"kss_{cltitle}_{tor}_{vdw}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def angle_map(all_data, figure_output):
    logging.info("running angle_map")

    color_map = topology_labels(short="P")

    vmax = isomer_energy() * 2

    for tstr in color_map:
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

        tdata = all_data[all_data["topology"] == tstr]
        for ax, tor in zip(flat_axs, tor_tests):
            pdata = tdata[tdata["torsions"] == tor]
            if tstr == "6P8":
                x = pdata["c3angle"]
                y = pdata["clangle"]
                ax.set_xlabel(clangle_str(num=3), fontsize=16)

            else:
                x = pdata["c2angle"]
                y = pdata["clangle"]
                ax.set_xlabel(
                    r"target internal angle [$^\circ$]",
                    fontsize=16,
                )
                ax.set_title(
                    f"{convert_tors(tor, num=False)}",
                    fontsize=16,
                )

            ax.scatter(
                x,
                y,
                c=pdata["energy_per_bb"],
                vmin=0,
                vmax=vmax,
                alpha=1.0,
                edgecolor="k",
                s=200,
                marker="s",
                cmap="Blues_r",
            )

            ax.tick_params(axis="both", which="major", labelsize=16)

            ax.set_ylabel(clangle_str(num=Xc_map(tstr)), fontsize=16)

        cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
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
        filename = f"am_{tstr}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def pd_4p82_figure(all_data, figure_output):
    logging.info("running angle_map")

    tstr = "4P82"

    vmax = 3

    fig, axs = plt.subplots(
        ncols=2,
        nrows=1,
        sharey=True,
        figsize=(16, 5),
    )

    tor_tests = ("ton", "toff")
    flat_axs = axs.flatten()

    tdata = all_data[all_data["topology"] == tstr]
    for ax, tor in zip(flat_axs, tor_tests):
        pdata = tdata[tdata["torsions"] == tor]

        x = pdata["c2angle"]
        y = pdata["clangle"]
        energies = pdata["energy_per_bb"]
        ax.set_xlabel(r"target internal angle [$^\circ$]", fontsize=16)
        ax.set_title(
            f"{convert_tors(tor, num=False)}",
            fontsize=16,
        )

        ax.scatter(
            x,
            y,
            c=[i - min(energies) for i in energies],
            vmin=0,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap="Blues_r",
        )

        ax.tick_params(axis="both", which="major", labelsize=16)

        ax.set_ylabel(clangle_str(num=4), fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
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
    filename = "4p82_test.pdf"
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def pdII_figure_bite_angle(all_data, figure_output):
    logging.info("running pdII_figure_bite_angle")

    color_map = ("2P4", "3P6", "4P8", "6P12", "12P24")

    fig, ax = plt.subplots(figsize=(8, 2.5))
    trim = all_data[all_data["vdws"] == "von"]
    trim = trim[trim["torsions"] == "ton"]
    trim = trim[trim["clangle"] == 90]

    all_bas = sorted(set(trim["target_bite_angle"]))
    all_ba_points = []
    tstr_points = {}
    for ba in all_bas:
        ba_data = trim[trim["target_bite_angle"] == ba]

        min_e_y = 1e24
        min_e_tstr = None
        for tstr in color_map:
            tdata = ba_data[ba_data["topology"] == tstr]
            ey = float(tdata["energy_per_bb"].iloc[0])
            if ey < min_e_y:
                min_e_y = ey
                min_e_tstr = tstr

        if min_e_tstr not in tstr_points:
            tstr_points[min_e_tstr] = []
        tstr_points[min_e_tstr].append((ba, min_e_y))

        all_ba_points.append((ba, min_e_y))

    ax.plot(
        [i[0] for i in all_ba_points],
        [i[1] for i in all_ba_points],
        alpha=1.0,
        lw=3,
        color="gray",
    )

    for tstr in color_map:
        ax.plot(
            [i[0] for i in tstr_points[tstr]],
            [i[1] for i in tstr_points[tstr]],
            alpha=1.0,
            # edgecolor="k",
            lw=3,
            marker="o",
            markeredgecolor="k",
            markersize=13,
            label=convert_topo(tstr),
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"target bite angle [$^\circ$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_ylim(0, 2)

    ax.legend(ncol=3, fontsize=16)
    fig.tight_layout()
    filename = "pdII_figure_bite_angle.pdf"
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
    write_out_mapping(all_data)

    bite_angle_relationship(all_data, figure_output)
    angle_map(low_e_data, figure_output)
    pdII_figure_bite_angle(low_e_data, figure_output)
    pd_4p82_figure(low_e_data, figure_output)
    selfsort_legend(low_e_data, figure_output)
    selfsort_map(low_e_data, figure_output)
    kinetic_selfsort_map(low_e_data, figure_output)
    selectivity_map(low_e_data, figure_output)
    nobiteangle_relationship(all_data, figure_output)


if __name__ == "__main__":
    main()