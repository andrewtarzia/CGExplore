#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting.

Author: Andrew Tarzia

"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt


plt.set_loglevel("WARNING")


def scatter(
    topo_to_c,
    results,
    ylabel,
    output_dir,
    filename,
):
    print("add properties to this")

    fig, ax = plt.subplots(figsize=(8, 5))
    switched_data = {}
    for biteangle in results:
        da = results[biteangle]
        for topo_s in da:
            if ylabel == "energy (eV)":
                ys = da[topo_s]["fin_energy"]
            elif ylabel == "OH-6":
                ys = da[topo_s]["oh6"]
                ax.axhline(y=0, lw=2, c="k")
            elif ylabel == "CU-8":
                ys = da[topo_s]["cu8"]
                ax.axhline(y=0, lw=2, c="k")

            if topo_s not in switched_data:
                switched_data[topo_s] = []
            switched_data[topo_s].append((biteangle, ys))

    for topo_s in topo_to_c:
        ax.plot(
            [i[0] for i in switched_data[topo_s]],
            [i[1] for i in switched_data[topo_s]],
            c=topo_to_c[topo_s][1],
            # marker=topo_to_c[topo_s][0],
            # edgecolor="k",
            # s=80,
            lw=3,
            alpha=1.0,
            label=convert_topo_names(topo_s),
        )

    ax.legend(fontsize=16, ncol=3)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("ff_str", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(0, 10)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def ey_vs_property(
    results,
    output_dir,
    filename,
):
    print("make this a property of interest -- pore?")
    return

    _to_plot = {
        "d2": ("o", "k"),
        "th2": ("X", "r"),
        "s62": ("D", "gold"),
        "d32": ("o", "skyblue"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for topo_s in _to_plot:
        x_vals = []
        y_vals = []
        for aniso in results:
            da = results[aniso]
            x_vals.append(da[topo_s]["cu8"])
            y_vals.append(da[topo_s]["fin_energy"])

        ax.scatter(
            x_vals,
            y_vals,
            c=_to_plot[topo_s][1],
            marker=_to_plot[topo_s][0],
            edgecolor="k",
            s=100,
            alpha=1.0,
            label=convert_topo_names(topo_s),
        )

    ax.legend(fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("CU-8", fontsize=16)
    ax.set_ylabel("energy (eV)", fontsize=16)
    ax.set_xlim(0, 2)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def geom_distributions(
    results,
    output_dir,
    filename,
):

    target_trio = "BNN"

    # Collect all values for each bond and angle type.
    distance_by_type = {}
    angle_by_type = {}
    for ff_str in results:
        da = results[ff_str]
        for topo_str in da:
            dists = da[topo_str]["distances"]
            angles = da[topo_str]["angles"]

            try:
                for d in dists:
                    dd = merge_bond_types(d)
                    if dd in distance_by_type:
                        distance_by_type[dd].extend(dists[d])
                    else:
                        distance_by_type[dd] = dists[d]

                for a in angles:
                    if a == target_trio:
                        aa = f"{a}{ff_str}"
                    else:
                        aa = merge_angle_types(a)
                    if aa in angle_by_type:
                        angle_by_type[aa].extend(angles[a])
                    else:
                        angle_by_type[aa] = angles[a]
            except KeyError as e:
                print(e)
                logging.info(f"expected bond types: {dists.keys()}")
                logging.info(f"expected bond types: {angles.keys()}")

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 8),
    )
    # Plot distributions of each bond type.
    for btype in distance_by_type:
        data = distance_by_type[btype]
        axs[0].hist(
            x=data,
            bins=50,
            range=(0, 4.4),
            density=True,
            histtype="step",
            # color='',
            label=btype,
            lw=3,
        )
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel(r"distance [$\mathrm{\AA}}$]", fontsize=16)
    axs[0].set_ylabel("frequency", fontsize=16)
    axs[0].legend(fontsize=16, ncol=1)

    # Plot distributions of each variable bond type.
    for atype in angle_by_type:
        if target_trio not in atype:
            continue
        data = angle_by_type[atype]
        biteangle = atype.replace(target_trio, "")
        axs[1].scatter(
            x=[biteangle for i in data],
            y=data,
            color="gray",
            s=30,
            alpha=0.3,
            rasterized=True,
        )
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("bite angle [degrees]", fontsize=16)
    axs[1].set_ylabel(f"{target_trio} [degrees]", fontsize=16)

    # Plot distributions of each angle type.
    for atype in angle_by_type:
        if target_trio in atype:
            continue
        data = angle_by_type[atype]
        axs[2].hist(
            x=data,
            bins=50,
            range=(0, 182),
            density=True,
            histtype="step",
            # color='',
            label=atype,
            lw=3,
        )
    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_xlabel("angle [degrees]", fontsize=16)
    axs[2].set_ylabel("frequency", fontsize=16)
    axs[2].legend(fontsize=16, ncol=1)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def heatmap(
    topo_to_c,
    results,
    output_dir,
    filename,
    vmin,
    vmax,
    clabel,
):
    print("add a property here too.")
    fig, ax = plt.subplots(figsize=(8, 8))
    maps = np.zeros((len(topo_to_c), len(results)))
    for j, biteangle in enumerate(results):
        da = results[biteangle]
        for i, topo_s in enumerate(topo_to_c):
            if clabel == "energy (eV)":
                maps[i][j] = da[topo_s]["fin_energy"]
            elif clabel == "CU-8":
                maps[i][j] = da[topo_s]["cu8"]
            elif clabel == "OH-6":
                maps[i][j] = da[topo_s]["oh6"]

    im = ax.imshow(maps, vmin=vmin, vmax=vmax, cmap="Purples_r")
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.4)
    cbar.ax.set_ylabel(clabel, rotation=-90, va="bottom", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    # ax.set_xticks(np.arange(maps.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(maps.shape[0] + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Scatter points for lowest/highest energy for each symm.
    if clabel == "energy (eV)":
        # Min of each row.
        index_min = np.argmin(maps, axis=1)
        ax.scatter(
            x=index_min,
            y=[topo_to_c[topo_s][2] for topo_s in topo_to_c],
            c="white",
            marker="P",
            edgecolors="k",
            s=80,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("ff_str", fontsize=16)
    ax.set_ylabel("topology", fontsize=16)
    # Show all ticks and label them with the respective lists.
    ax.set_xticks([i for i in range(len(results))])
    ax.set_xticklabels([a for a in results])
    ax.set_yticks([topo_to_c[topo_s][2] for topo_s in topo_to_c])
    ax.set_yticklabels(
        [convert_topo_names(topo_s) for topo_s in topo_to_c]
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def ey_vs_shape(
    results,
    topo_to_c,
    output_dir,
    filename,
):

    fig, ax = plt.subplots(figsize=(8, 5))

    for topo_s in topo_to_c:
        x_vals = []
        y_vals = []
        for res in results:
            da = results[res]
            x_vals.append(da[topo_s]["oh6"])
            y_vals.append(da[topo_s]["fin_energy"])

        ax.scatter(
            x_vals,
            y_vals,
            c=topo_to_c[topo_s][1],
            marker=topo_to_c[topo_s][0],
            edgecolor="k",
            s=40,
            alpha=1.0,
            label=convert_topo_names(topo_s),
        )

    ax.legend(fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("OH-6", fontsize=16)
    ax.set_ylabel("energy (eV)", fontsize=16)
    # ax.set_xlim(0, 2)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def convergence(
    results,
    output_dir,
    filename,
):

    # Pick examples to plot.
    _to_plot = (
        "FourPlusSix_ba70",
        "FourPlusSix2_ba130",
        "FourPlusSix_ba20",
        "FourPlusSix2_ba180",
        "FourPlusSix_ba100",
        "FourPlusSix2_ba40",
    )

    fig, axs = plt.subplots(
        nrows=len(_to_plot),
        ncols=1,
        sharex=True,
        figsize=(8, 10),
    )

    xmax, xwin = (501, 50)

    for name, ax in zip(_to_plot, axs):
        topo_s, ff_str = name.split("_")
        da = results[ff_str]
        traj = da[topo_s]["traj"]
        traj_x = [i for i in traj]
        traj_e = [traj[i]["energy"] for i in traj]
        traj_g = [traj[i]["gnorm"] for i in traj]

        color = "tab:red"
        ax.plot(
            traj_x,
            traj_e,
            lw=5,
            # marker='o',
            # markersize=12,
            color=color,
        )
        ax.tick_params(axis="y", labelcolor=color, labelsize=16)
        ax.set_yscale("log")

        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()
        color = "tab:blue"
        ax2.plot(
            traj_x,
            traj_g,
            lw=5,
            # marker='X',
            # markersize=12,
            color=color,
        )
        ax2.tick_params(axis="y", labelcolor=color, labelsize=16)
        ax2.set_yscale("log")

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.text(
            x=xmax * 0.6,
            y=1100,
            s=f"{convert_topo_names(topo_s)}, ff-str={ff_str}",
            fontsize=16,
        )
        if name == _to_plot[0]:
            ax.set_ylabel("energy [eV]", fontsize=16)
            ax2.set_ylabel("Gnorm", fontsize=16)
        ax.axhline(y=0.01, c="k", lw=2, linestyle="--")

    ax.set_xlabel("step", fontsize=16)
    ax.set_xticks(range(0, xmax, xwin))
    ax.set_xticklabels([str(i) for i in range(0, xmax, xwin)])

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
