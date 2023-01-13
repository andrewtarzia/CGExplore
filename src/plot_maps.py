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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from env_set import cages
from analysis_utilities import (
    data_to_array,
    topo_to_colormap,
    max_energy,
    min_radius,
    isomer_energy,
    min_b2b_distance,
    topology_labels,
    stoich_map,
    cltypetopo_to_colormap,
    write_out_mapping,
    torsion_to_colormap,
    convert_tors,
    convert_topo,
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
            "ylbl": "energy per bond [eV]",
            "yline": max_energy(),
            "ylim": (0, max_energy() * 2),
        },
        "pore": {
            "ylbl": "pore",
            "yline": None,
            "ylim": (0, 50),
        },
        "min_b2b": {
            "ylbl": "min b2b",
            "yline": None,
            "ylim": (0, 3),
        },
    }

    io1 = sorted(set(all_data[cols_to_iter[0]]))
    io2 = sorted(set(all_data[cols_to_iter[1]]))
    for cla in io1:
        for to in io2:
            filt_data = all_data[all_data[cols_to_iter[0]] == cla]
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
                            if i == 2 and j == 1:
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
                            if i == 2:
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
                    spines = ["top", "right", "left", "bottom"]
                    for s in spines:
                        ax.spines[s].set_visible(False)

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
        [float(i) for i in set(all_data["target_bite_angle"])]
    )
    clangles = sorted([float(i) for i in set(all_data["clangle"])])

    vcount = 10
    properties = {
        "energy": {
            "col": "energy_per_bond",
            "cut": isomer_energy(),
            "dir": "<",
        },
        "energy_max": {
            "col": "energy_per_bond",
            "cut": max_energy(),
            "dir": "<",
        },
        "pore": {"col": "pore", "cut": min_radius() * 2, "dir": "<"},
        "min_b2b": {
            "col": "min_b2b",
            "cut": min_b2b_distance(),
            "dir": "<",
        },
        "sv_n_dist": {"col": "sv_n_dist", "cut": 0.9, "dir": ">"},
        "sv_l_dist": {"col": "sv_l_dist", "cut": 0.9, "dir": ">"},
    }
    for prop in properties:
        pdict = properties[prop]

        for clangle in clangles:
            fig, axs = plt.subplots(
                nrows=1,
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(16, 5),
            )
            cdata = all_data[all_data["clangle"] == clangle]
            count = 0
            topology_order = {}
            tset = set(cdata["topology"])
            for tstr in topology_labels(short=False):
                if tstr in tset:
                    topology_order[tstr] = count
                    count += 1

            for tors, ax in zip(("ton", "toff"), axs):
                tordata = cdata[cdata["torsions"] == tors]
                for tstr in topology_order:
                    yvalue = topology_order[tstr]
                    tdata = tordata[tordata["topology"] == tstr]
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
                        ax.scatter(
                            xvalue,
                            yvalue,
                            # c=under_count / total_count,
                            c=under_count,
                            vmin=0,
                            vmax=vcount,
                            alpha=1.0,
                            edgecolor="none",
                            s=80,
                            marker="s",
                            cmap="Blues",
                        )

                ax.set_title(
                    (f"{convert_tors(tors, num=False)}; " f"{clangle}"),
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
                f"count {pdict['col']} {order_string} {pdict['cut']}",
                fontsize=16,
            )

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    figure_output, f"sel_{prop}_{clangle}.pdf"
                ),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def selfsort_map(all_data, figure_output):
    logging.info("running selfsort_map")

    cols_to_map = ["clsigma", "clangle"]
    cols_to_iter = [
        "torsions",
        "cltitle",
        "c2sigma",
        "target_bite_angle",
    ]

    io1 = sorted(set(all_data[cols_to_iter[0]]))
    io2 = sorted(set(all_data[cols_to_iter[1]]))
    for tor in io1:
        for cltitle in io2:
            tors_data = all_data[all_data[cols_to_iter[0]] == tor]
            tors_data = tors_data[tors_data[cols_to_iter[1]] == cltitle]
            fig = plt.figure(figsize=(16, 8))
            uo1 = sorted(set(tors_data[cols_to_map[0]]))
            uo2 = sorted(set(tors_data[cols_to_map[1]]))
            print(tor, cltitle, uo1, uo2)

            gs = grid_spec.GridSpec(len(uo1), len(uo2))
            ax_objs = []
            for (i, o1), (j, o2) in itertools.product(
                enumerate(uo1), enumerate(uo2)
            ):
                ax_objs.append(
                    fig.add_subplot(gs[i : i + 1, j : j + 1])
                )
                ax = ax_objs[-1]
                plot_data = tors_data[tors_data[cols_to_map[0]] == o1]
                plot_data = plot_data[plot_data[cols_to_map[1]] == o2]
                io3 = sorted(set(plot_data[cols_to_iter[2]]))
                io4 = sorted(set(plot_data[cols_to_iter[3]]))
                ylabel_map = {}
                for cla, ba in itertools.product(io3, io4):
                    xvalue = ba
                    yvalue = io3.index(cla)
                    yname = cla
                    ylabel_map[yname] = yvalue

                    filt_data = plot_data[
                        plot_data[cols_to_iter[2]] == cla
                    ]
                    filt_data = filt_data[
                        filt_data[cols_to_iter[3]] == ba
                    ]
                    if len(filt_data) == 0:
                        continue

                    energies = {
                        str(row["topology"]): float(row["energy"])
                        / stoich_map(str(row["topology"]))
                        for i, row in filt_data.iterrows()
                    }
                    dists = {
                        str(row["topology"]): float(row["pore"])
                        for i, row in filt_data.iterrows()
                    }
                    b2bs = {
                        str(row["topology"]): float(row["min_b2b"])
                        for i, row in filt_data.iterrows()
                    }
                    # svs = {
                    #     str(row["topology"]): (
                    #         float(row["sv_n_dist"]),
                    #         float(row["sv_l_dist"]),
                    #     )
                    #     for i, row in filt_data.iterrows()
                    # }
                    num_mixed = len(
                        tuple(
                            i
                            for i in list(energies.values())
                            if i < isomer_energy()
                        )
                    )
                    min_energy = min(energies.values())
                    min_dist = None
                    # min_svs = None
                    min_b2b = None
                    if min_energy > max_energy():
                        topo_str = "unstable"
                        colour = "gray"
                    elif num_mixed > 1:
                        topo_str = "mixed"
                        colour = "white"
                    else:
                        topo_str = list(energies.keys())[
                            list(energies.values()).index(min_energy)
                        ]
                        min_dist = dists[topo_str]
                        # min_svs = svs[topo_str]
                        min_b2b = b2bs[topo_str]
                        colour = cltypetopo_to_colormap()[cltitle][
                            topo_str
                        ]
                        # print(energies, topo_str, min_dist, min_svs)
                        # print(min_b2b)

                    rect = ax.patch
                    rect.set_alpha(0)
                    ax.scatter(
                        xvalue,
                        yvalue,
                        c=colour,
                        alpha=1.0,
                        marker="s",
                        edgecolor="k",
                        s=200,
                    )

                    if min_dist is not None and min_dist < min_radius():
                        ax.scatter(
                            xvalue,
                            yvalue,
                            c="k",
                            alpha=1.0,
                            marker="X",
                            edgecolor="none",
                            s=80,
                        )

                    if (
                        min_b2b is not None
                        and min_b2b < min_b2b_distance()
                    ):
                        ax.scatter(
                            xvalue,
                            yvalue,
                            c="k",
                            alpha=1.0,
                            marker="D",
                            edgecolor="none",
                            s=80,
                        )

                    ax.set_title(
                        f"{cols_to_map[0]}:{o1}; {cols_to_map[1]}:{o2}",
                        fontsize=16,
                    )
                    if i == 1 and j == 0:
                        # ax.set_ylabel("CL angle [deg]", fontsize=16)
                        ax.set_ylabel("C2 sigma [A]", fontsize=16)
                    if i == 2 and j == 1:
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
                    if i == 2:
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
                        ax.set_yticks(list(ylabel_map.values()))
                        ax.set_yticklabels(list(ylabel_map.keys()))
                    else:
                        ax.set_yticklabels([])

                for i, ax in enumerate(ax_objs):
                    spines = ["top", "right", "left", "bottom"]
                    for s in spines:
                        ax.spines[s].set_visible(False)
                    ax.set_ylim(-0.5, 2.5)

            for i in cltypetopo_to_colormap():
                if i not in (cltitle, "mixed"):
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
                        s=300,
                        marker="s",
                        alpha=1.0,
                        label=convert_topo(string),
                    )
            ax.scatter(
                None,
                None,
                c="gray",
                edgecolor="k",
                s=300,
                marker="s",
                alpha=1.0,
                label="unstable",
            )
            ax.scatter(
                None,
                None,
                c="k",
                edgecolor="k",
                s=300,
                marker="s",
                alpha=1.0,
                label=f"min distance < {min_radius()}A",
            )

            fig.legend(
                bbox_to_anchor=(0, 1.02, 2, 0.2),
                loc="lower left",
                ncol=4,
                fontsize=16,
            )

            fig.tight_layout()
            filename = f"ss_{cltitle}_{tor}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def angle_map(all_data, figure_output):
    logging.info("running angle_map")

    color_map = topo_to_colormap()

    trim = all_data[all_data["clsigma"] == 2]
    trim = trim[trim["c2sigma"] == 5]
    vmax = max_energy() * 3

    for tstr in color_map:
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 5))
        tdata = trim[trim["topology"] == tstr]
        for ax, tor in zip(axs, ("ton", "toff")):
            pdata = tdata[tdata["torsions"] == tor]
            ax.scatter(
                pdata["target_bite_angle"],
                pdata["clangle"],
                c=pdata["energy"],
                vmin=0,
                vmax=vmax,
                alpha=1.0,
                edgecolor="k",
                s=200,
                marker="s",
                cmap="Blues",
            )

            ax.set_title(convert_tors(tor, num=False), fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("bite angle [deg]", fontsize=16)
            ax.set_ylabel("cl angle [deg]", fontsize=16)

        cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        cmap = mpl.cm.Blues
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("energy [eV]", fontsize=16)

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

    selectivity_map(opt_data, figure_output)
    energy_map(opt_data, figure_output)
    angle_map(opt_data, figure_output)
    selfsort_map(opt_data, figure_output)
    bite_angle_relationship(opt_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
