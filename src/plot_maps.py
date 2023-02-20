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
import matplotlib.gridspec as grid_spec

from env_set import cages
from analysis_utilities import (
    data_to_array,
    max_energy,
    isomer_energy,
    topology_labels,
    cltypetopo_to_colormap,
    write_out_mapping,
    torsion_to_colormap,
    convert_tors,
    convert_topo,
    convert_vdws,
)


def bite_angle_relationship(all_data, figure_output):
    logging.info("running bite_angle_relationship")

    color_map = topo_to_colormap()
    for torsion in ("ton", "toff"):
        tor_data = all_data[all_data["torsions"] == torsion]
        for tstr in color_map:
            filt_data = tor_data[tor_data["topology"] == tstr]
            for t_angle in set(list(filt_data["clangle"])):
                clan_data = filt_data[filt_data["clangle"] == t_angle]
                fig, ax = plt.subplots(figsize=(8, 5))
                for c1_opt in sorted(set(clan_data["c2sigma"])):
                    test_data = clan_data[
                        clan_data["c2sigma"] == c1_opt
                    ]
                    for c2_opt in sorted(set(test_data["clsigma"])):
                        plot_data = test_data[
                            test_data["clsigma"] == c2_opt
                        ]
                        xs = list(plot_data["target_bite_angle"])
                        ys = list(plot_data["energy"])
                        xs, ys = zip(*sorted(zip(xs, ys)))
                        ax.plot(
                            xs,
                            ys,
                            lw=3,
                            alpha=1.0,
                            label=f"{c1_opt}.{c2_opt}",
                            marker="o",
                        )

                    ax.tick_params(
                        axis="both", which="major", labelsize=16
                    )
                    ax.set_xlabel("target 2c bite angle", fontsize=16)
                    ax.set_ylabel("energy", fontsize=16)
                    ax.legend()
                    ax.set_ylim(0, 2 * max_energy())

                fig.tight_layout()
                filename = f"ar_{torsion}_{t_angle}_{tstr}.pdf"
                fig.savefig(
                    os.path.join(figure_output, filename),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()


def energy_map(all_data, figure_output):
    logging.info("running energy_map")

    cols_to_map = [
        "clsigma",
        "c2sigma",
        "torsions",
    ]
    cols_to_iter = [
        "clangle",
        "topology",
        # "target_bite_angle",
    ]
    properties = {
        "energy_per_bond": {
            "ylbl": "energy per bond [kJmol-1]",
            "yline": max_energy(),
            "ylim": (0, max_energy() * 2),
        },
        # "pore": {
        #     "ylbl": "pore",
        #     "yline": None,
        #     "ylim": (0, 50),
        # },
        # "min_b2b": {
        #     "ylbl": "min b2b",
        #     "yline": None,
        #     "ylim": (0, 3),
        # },
    }

    von_data = all_data[all_data["vdws"] == "von"]
    io1 = sorted(set(von_data[cols_to_iter[0]]))
    io2 = sorted(set(von_data[cols_to_iter[1]]))
    for cla in io1:
        for to in io2:
            if to in ("6P8",):
                continue
            filt_data = von_data[von_data[cols_to_iter[0]] == cla]
            filt_data = filt_data[filt_data[cols_to_iter[1]] == to]
            if len(filt_data) == 0:
                continue
            uo1 = sorted(set(filt_data[cols_to_map[0]]))
            uo2 = sorted(set(filt_data[cols_to_map[1]]))
            uo3 = sorted(set(filt_data[cols_to_map[2]]))

            gs = grid_spec.GridSpec(len(uo1), len(uo2))
            for prop in properties:
                ybl = properties[prop]["ylbl"]
                yline = properties[prop]["yline"]
                ylim = properties[prop]["ylim"]

                fig = plt.figure(figsize=(16, 10))
                ax_objs = []
                for i, o1 in enumerate(uo1):
                    for j, o2 in enumerate(uo2):
                        ax_objs.append(
                            fig.add_subplot(gs[i : i + 1, j : j + 1])
                        )
                        for k, o3 in enumerate(uo3):
                            plot_data = filt_data[
                                filt_data[cols_to_map[0]] == o1
                            ]
                            plot_data = plot_data[
                                plot_data[cols_to_map[1]] == o2
                            ]
                            plot_data = plot_data[
                                plot_data[cols_to_map[2]] == o3
                            ]
                            if len(plot_data) == 0:
                                continue
                            ax = ax_objs[-1]
                            rect = ax.patch
                            rect.set_alpha(0)

                            xs = list(plot_data["target_bite_angle"])
                            ys = list(plot_data[prop])
                            xs, ys = zip(*sorted(zip(xs, ys)))
                            torlbl = convert_tors(
                                o3,
                                num=False,
                            )
                            ax.plot(
                                xs,
                                ys,
                                c=torsion_to_colormap()[o3],
                                lw=3,
                                alpha=1.0,
                                label=f"{torlbl}",
                                marker="o",
                            )
                            # ax.set_title(" min e value")
                            ax.set_title(
                                f"CLsig{o1}/C2sig{o2}",
                                fontsize=16,
                            )
                            if i == 0 and j == 0:
                                ax.legend(fontsize=16)
                            if i == 1 and j == 0:
                                ax.set_ylabel(ybl, fontsize=16)
                            if i == 1 and j == 1:
                                ax.set_xlabel(
                                    "bite angle [deg]",
                                    fontsize=16,
                                )
                            ax.tick_params(
                                axis="both",
                                which="both",
                                bottom=False,
                                top=False,
                                left=False,
                                right=False,
                                labelsize=16,
                            )
                            if i == 1:
                                ax.tick_params(
                                    axis="y",
                                    which="major",
                                    labelsize=16,
                                )
                            else:
                                ax.set_xticklabels([])
                            if j == 0:
                                ax.tick_params(
                                    axis="x",
                                    which="major",
                                    labelsize=16,
                                )
                            else:
                                ax.set_yticklabels([])

                            if yline is not None:
                                ax.axhline(
                                    y=yline,
                                    c="k",
                                    lw=2,
                                    linestyle="--",
                                )

                for i, ax in enumerate(ax_objs):
                    ax.set_ylim(ylim)
                    # spines = ["top", "right", "left", "bottom"]
                    # for s in spines:
                    #     ax.spines[s].set_visible(False)

                fig.suptitle(f"{to}, {cla}", fontsize=16)
                fig.tight_layout()
                filename = f"em_{cla}_{to}_{prop}.pdf"
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

    vcount = 1
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

        vdata = all_data[all_data["vdws"] == "von"]
        count = 0
        topology_order = {}
        tset = set(all_data["topology"])
        for tstr in topology_labels(short="P"):
            if tstr in ("6P8",):
                continue
            if tstr in tset:
                topology_order[tstr] = count
                count += 1

        for clangle in clangles:

            fig, axs = plt.subplots(
                nrows=1,
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(16, 5),
            )

            cdata = vdata[vdata["clangle"] == clangle]
            for tors, ax in zip(("ton", "toff"), axs):
                tordata = cdata[cdata["torsions"] == tors]
                for tstr in topology_order:
                    xvalues = []
                    yvalues = []
                    cvalues = []
                    yvalue = topology_order[tstr]
                    tdata = tordata[tordata["topology"] == tstr]
                    for ba in bite_angles:
                        plotdata = tdata[
                            tdata["target_bite_angle"] == ba
                        ]
                        total_count = len(plotdata)

                        property_list = list(plotdata[pdict["col"]])
                        if pdict["dir"] == "<":
                            under_cut = [
                                i
                                for i in property_list
                                if i < pdict["cut"]
                            ]
                            order_string = "<"
                        elif pdict["dir"] == ">":
                            under_cut = [
                                i
                                for i in property_list
                                if i > pdict["cut"]
                            ]
                            order_string = ">"
                        # print(under_cut)
                        under_count = len(under_cut)
                        xvalue = ba
                        xvalues.append(xvalue)
                        yvalues.append(yvalue)
                        if under_count == 0 or total_count == 0:
                            cvalues.append(0)
                        else:
                            cvalues.append(under_count / total_count)

                    ax.scatter(
                        xvalues,
                        yvalues,
                        # c=under_count / total_count,
                        c=cvalues,
                        vmin=0,
                        vmax=vcount,
                        alpha=1.0,
                        edgecolor="none",
                        s=80,
                        marker="s",
                        cmap="Blues",
                    )

                ax.set_title(
                    (f"{convert_tors(tors, num=False)}, {clangle}"),
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

            cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
            cmap = mpl.cm.Blues
            norm = mpl.colors.Normalize(vmin=0, vmax=vcount)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation="vertical",
            )
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(
                f"prop. {pdict['clbl']} {order_string} {pdict['cut']}",
                fontsize=16,
            )

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    figure_output, f"sel_{prop}_{int(clangle)}.pdf"
                ),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def draw_pie(dists, colours, xpos, ypos, size, ax):
    """
    From:
    https://stackoverflow.com/questions/56337732/how-to-plot-scatter-
    pie-chart-using-matplotlib

    """

    # for incremental pie slices
    cumsum = np.cumsum(dists)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()

    for r1, r2, c in zip(pie[:-1], pie[1:], colours):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])

        ax.scatter([xpos], [ypos], marker=xy, s=size, c=[c])

    return ax


def selfsort_map(all_data, figure_output):
    logging.info("running selfsort_map")

    trim = all_data[all_data["clsigma"] == 5]
    trim = trim[trim["c2sigma"] == 5]
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

        data = trim[trim[cols_to_iter[0]] == tor]
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

            if min_energy > max_energy():
                colours = ["k"]
                dists = [1]
            elif len(mixed_energies) == 0:
                topo_str = list(energies.keys())[
                    list(energies.values()).index(min_energy)
                ]
                colours = [cltypetopo_to_colormap()[cltitle][topo_str]]
                dists = [1]

            else:
                colours = [
                    cltypetopo_to_colormap()[cltitle][i]
                    for i in mixed_energies
                ]
                dists = [1 for i in range(len(colours))]

            draw_pie(
                dists=dists,
                colours=colours,
                xpos=xvalue,
                ypos=yvalue,
                size=400,
                ax=ax,
            )

            rect = ax.patch
            rect.set_alpha(0)

            ax.set_ylabel("CL angle [deg]", fontsize=16)
            ax.set_xlabel("bite angle [deg]", fontsize=16)
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelsize=16,
            )
            ax.tick_params(
                axis="y",
                which="major",
                labelsize=16,
            )

            ax.tick_params(
                axis="x",
                which="major",
                labelsize=16,
            )

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
                    edgecolor="none",
                    s=300,
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


def angle_map(all_data, figure_output):
    logging.info("running angle_map")

    color_map = topology_labels(short="P")

    trim = all_data[all_data["clsigma"] == 5]
    trim = trim[trim["c2sigma"] == 5]
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
    logging.info(f"there are {len(all_data)} collected data")
    write_out_mapping(all_data)

    selfsort_map(all_data, figure_output)
    selectivity_map(all_data, figure_output)
    angle_map(all_data, figure_output)
    raise SystemExit()
    bite_angle_relationship(all_data, figure_output)
    energy_map(all_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
