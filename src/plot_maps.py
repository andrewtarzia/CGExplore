#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot property, input maps.

Author: Andrew Tarzia

"""

import sys
import os
import logging
import itertools
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from env_set import cages
from analysis_utilities import (
    data_to_array,
    stoich_map,
    get_lowest_energy_data,
    isomer_energy,
    topology_labels,
    cltypetopo_to_colormap,
    write_out_mapping,
    convert_tors,
    convert_topo,
    convert_vdws,
)


def clangle_relationship(all_data, figure_output):
    logging.info("running clangle_relationship")

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
            # for c1_opt in sorted(set(clan_data["c2r0"])):
            for run_number in clan_data["run_number"]:
                plot_data = clan_data[
                    clan_data["run_number"] == run_number
                ]
                if len(plot_data) == 0:
                    continue
                # for c1_opt in sorted(set(clan_data["c3r0"])):
                #     test_data = clan_data[clan_data["c3r0"] == c1_opt]
                #     for c2_opt in sorted(set(test_data["clr0"])):
                #         plot_data = test_data[test_data["clr0"] == c2_opt]
                xs = list(plot_data["c3angle"])
                ys = list(plot_data["energy_per_bond"])
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

            # for x, y in zip(xs, y_medians):
            #     if y < isomer_energy():
            #         yx = [4.6]
            #         ax.scatter(
            #             [x],
            #             yx,
            #             marker="P",
            #             c="#086788",
            #             s=120,
            #             edgecolor="k",
            #             zorder=2,
            #         )

            ax.set_ylabel(r"$E_{bf}$", fontsize=16)
            ax.set_title(t_angle, fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("c3 angle", fontsize=16)
        ax.set_ylim(0, 5)
        # ax.set_yscale("log")

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
                    xs = list(plot_data["target_bite_angle"])
                    ys = list(plot_data["energy_per_bond"])
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

                # for x, y in zip(xs, y_medians):
                #     if y < isomer_energy():
                #         if tors == "ton":
                #             yx = [4.6]
                #         elif tors == "toff":
                #             yx = [4.6]
                #         ax.scatter(
                #             [x],
                #             yx,
                #             marker="P",
                #             c=cmap[tors][0],
                #             s=120,
                #             edgecolor="k",
                #             zorder=2,
                #         )
                # ax.axvline(
                #     x=xs[min(range(len(ys)), key=ys.__getitem__)],
                #     c=cmap[(c1_opt, c2_opt)][0],
                #     linestyle="--",
                #     alpha=0.5,
                # )

            ax.set_ylabel(r"$E_{bf}$", fontsize=16)
            ax.set_title(t_angle, fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c bite angle", fontsize=16)
        ax.legend(ncol=2, fontsize=16)
        ax.set_ylim(0, 5)
        ax.set_xlim(-1, 181)
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
            "col": "energy_per_bond",
            "cut": isomer_energy(),
            "dir": "<",
            "clbl": "$E_{bf}$",
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
                        s=60,
                        marker="o",
                        # cmap="Blues",
                    )

                ax.set_title(
                    # (
                    #     f"{convert_tors(tors, num=False)}, "
                    #     f"{convert_vdws(vdw)}, {clangle}"
                    # ),
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

        # cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        # cmap = mpl.cm.Blues
        # norm = mpl.colors.Normalize(vmin=0, vmax=vcount)
        # cbar = fig.colorbar(
        #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        #     cax=cbar_ax,
        #     orientation="vertical",
        # )
        # cbar.ax.tick_params(labelsize=16)
        # cbar.set_label(
        #     f"prop. {pdict['clbl']} {order_string} {pdict['cut']}",
        #     fontsize=16,
        # )

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"sel_{prop}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


# def draw_pie(colours, xpos, ypos, size, ax):
#     """
#     From:
#     https://stackoverflow.com/questions/56337732/how-to-plot-scatter-
#     pie-chart-using-matplotlib

#     """

#     num_points =

#     # for incremental pie slices
#     cumsum = np.cumsum(ratios)
#     cumsum = cumsum / cumsum[-1]
#     pie = [0] + cumsum.tolist()

#     for r1, r2, c in zip(pie[:-1], pie[1:], colours):
#         angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 100)
#         x = [0] + np.cos(angles).tolist()
#         y = [0] + np.sin(angles).tolist()

#         xy = np.column_stack([x, y])

#         ax.scatter([xpos], [ypos], marker=xy, s=size, c=[c])

#     return ax


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


def selfsort_map(all_data, figure_output):
    logging.info("running selfsort_map")

    cols_to_map = ["clangle", "target_bite_angle"]
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
                str(row["topology"]): float(row["energy_per_bond"])
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
            ax.set_ylabel("CL angle [deg]", fontsize=16)
            ax.set_xlabel("bite angle [deg]", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=16)

        for i in cltypetopo_to_colormap():
            if i not in (cltitle,):
                continue
            for j in cltypetopo_to_colormap()[i]:
                # if i == "mixed":
                #     string = f"mixed: {j}"
                # else:
                string = j
                ax.scatter(
                    None,
                    None,
                    c=cltypetopo_to_colormap()[i][j],
                    edgecolor="k",
                    s=400,
                    marker="o",
                    alpha=1.0,
                    label=convert_topo(string),
                )

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=5,
            fontsize=16,
        )

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

    cols_to_map = ["clangle", "target_bite_angle"]
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
                str(row["topology"]): float(row["energy_per_bond"])
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

            # rect = ax.patch
            # rect.set_alpha(0)

            ax.set_title(convert_tors(tor, num=False), fontsize=16)
            ax.set_ylabel("CL angle [deg]", fontsize=16)
            ax.set_xlabel("bite angle [deg]", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=16)

        for i in cltypetopo_to_colormap():
            if i not in (cltitle,):
                continue
            for j in cltypetopo_to_colormap()[i]:
                # if i == "mixed":
                #     string = f"mixed: {j}"
                # else:
                string = j
                ax.scatter(
                    None,
                    None,
                    c=cltypetopo_to_colormap()[i][j],
                    edgecolor="k",
                    s=400,
                    marker="o",
                    alpha=1.0,
                    label=convert_topo(string),
                )

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=5,
            fontsize=16,
        )

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

    raise SystemExit(
        "try to make this contain all sizes? like error in the colours?"
    )
    trim = all_data[all_data["clr0"] == 2]
    trim = trim[trim["c2r0"] == 5]
    vmax = isomer_energy() * 2

    for tstr in color_map:
        fig, axs = plt.subplots(
            ncols=2,
            nrows=2,
            sharey=True,
            figsize=(16, 10),
        )
        flat_axs = axs.flatten()
        tdata = trim[trim["topology"] == tstr]
        for ax, (tor, vdw) in zip(
            flat_axs,
            itertools.product(("ton", "toff"), ("von", "voff")),
        ):
            pdata = tdata[tdata["torsions"] == tor]
            pdata = pdata[pdata["vdws"] == vdw]
            ax.scatter(
                pdata["target_bite_angle"],
                pdata["clangle"],
                c=pdata["energy_per_bond"],
                vmin=0,
                vmax=vmax,
                alpha=1.0,
                edgecolor="k",
                s=200,
                marker="s",
                cmap="Blues_r",
            )

            ax.set_title(
                f"{convert_tors(tor, num=False)}; {convert_vdws(vdw)}",
                fontsize=16,
            )
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("bite angle [deg]", fontsize=16)
            ax.set_ylabel("cl angle [deg]", fontsize=16)

        cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        cmap = mpl.cm.Blues_r
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("$E_{pb}$ [kJmol-1]", fontsize=16)

        fig.tight_layout()
        filename = f"am_{tstr}.pdf"
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

    kinetic_selfsort_map(low_e_data, figure_output)
    selectivity_map(low_e_data, figure_output)
    selfsort_map(low_e_data, figure_output)
    clangle_relationship(all_data, figure_output)
    bite_angle_relationship(all_data, figure_output)
    raise SystemExit()
    angle_map(low_e_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
