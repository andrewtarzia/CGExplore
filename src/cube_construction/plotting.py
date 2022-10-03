#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting.

Author: Andrew Tarzia

"""

import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


plt.set_loglevel("WARNING")


def merge_bond_types(s):

    translation = {
        "CC": "face",
        "BB": "face",
        "CZn": "face-metal",
        "BZn": "face-metal",
        "FeFe": "metal",
        "FeZn": "metal",
        "ZnZn": "metal",
    }

    return translation[s]


def merge_angle_types(s):

    translation = {
        "BCC": "face",
        "BBC": "face",
        "BCZn": "face-metal",
        "BBZn": "face-metal",
        "CCZn": "face-metal",
        "BFeZn": "face-metal",
        "CFeZn": "face-metal",
        "FeZnZn": "metal",
        "FeFeFe": "metal",
        "ZnZnZn": "metal",
    }

    return translation[s]


def convert_symm_names(symm_name):

    new_names = {
        "d2": r"D$_2$",
        "th1": r"T$_{h, 1}$",
        "th2": r"T$_{h, 2}$",
        "td": r"T$_{\Delta}$",
        "tl": r"T$_{\Lambda}$",
        "s41": r"S$_{4, 1}$",
        "s42": r"S$_{4, 2}$",
        "s61": r"S$_{6, 1}$",
        "s62": r"S$_{6, 2}$",
        "d31": r"D$_{3, 1}$",
        "d32": r"D$_{3, 2}$",
        "d31n": r"D$_{3, 1n}$",
        "d32n": r"D$_{3, 2n}$",
        "c2v": r"C$_{2h}$",
        "c2h": r"C$_{2v}$",
        "d3c3": r"knot",
    }

    return new_names[symm_name]


def scatter(
    symm_to_c,
    results,
    ylabel,
    output_dir,
    filename,
    flex,
):

    fig, ax = plt.subplots(figsize=(8, 5))
    for aniso in results:
        da = results[aniso]
        for symm in da:
            if ylabel == "energy (eV)":
                ys = da[symm]["fin_energy"]
            elif ylabel == "CU-8":
                ys = da[symm]["cu8"]
                ax.axhline(y=0, lw=2, c="k")

            ax.scatter(
                aniso,
                ys,
                c=symm_to_c[symm][1],
                marker=symm_to_c[symm][0],
                edgecolor="k",
                s=80,
                alpha=0.5,
            )

    legend_elements = []
    for s in symm_to_c:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="w",
                marker=symm_to_c[s][0],
                label=convert_symm_names(s),
                markerfacecolor=symm_to_c[s][1],
                markersize=10,
                markeredgecolor="k",
            )
        )

    ax.legend(handles=legend_elements, fontsize=16, ncol=3)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("anisotropy", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"flex: {flex}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def ey_vs_shape(
    results,
    output_dir,
    filename,
):

    _to_plot = {
        "d2": ("o", "k"),
        "th2": ("X", "r"),
        "s62": ("D", "gold"),
        "d32": ("o", "skyblue"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for symm in _to_plot:
        x_vals = []
        y_vals = []
        for aniso in results:
            da = results[aniso]
            x_vals.append(da[symm]["cu8"])
            y_vals.append(da[symm]["fin_energy"])

        ax.scatter(
            x_vals,
            y_vals,
            c=_to_plot[symm][1],
            marker=_to_plot[symm][0],
            edgecolor="k",
            s=100,
            alpha=1.0,
            label=convert_symm_names(symm),
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


def comp_scatter(
    symm_to_c,
    symm_set,
    results,
    ylabel,
    output_dir,
    filename,
    flex,
    ylim,
):

    fig, ax = plt.subplots(figsize=(8, 5))
    for aniso in results:
        da = results[aniso]
        for symm in da:
            if symm not in symm_set:
                continue
            if ylabel == "energy (eV)":
                ys = da[symm]["fin_energy"]
            elif ylabel == "CU-8":
                ys = da[symm]["cu8"]
                ax.axhline(y=0, lw=2, c="k")

            ax.scatter(
                aniso,
                ys,
                c=symm_to_c[symm][1],
                marker=symm_to_c[symm][0],
                edgecolor="k",
                s=120,
            )

    legend_elements = []
    for s in symm_to_c:
        if s not in symm_set:
            continue
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="w",
                marker=symm_to_c[s][0],
                label=convert_symm_names(s),
                markerfacecolor=symm_to_c[s][1],
                markersize=12,
                markeredgecolor="k",
            )
        )

    ax.legend(handles=legend_elements, fontsize=16, ncol=2)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("anisotropy", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"flex: {flex}", fontsize=16)
    ax.set_ylim(ylim)

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

    # Collect all values for each bond and angle type.
    distance_by_type = {}
    angle_by_type = {}
    for aniso in results:
        da = results[aniso]
        for symm in da:
            dists = da[symm]["distances"]
            angles = da[symm]["angles"]
            for d in dists:
                if d == "BC":
                    dd = f"{d}{aniso}"
                else:
                    dd = merge_bond_types(d)
                if dd in distance_by_type:
                    distance_by_type[dd].extend(dists[d])
                else:
                    distance_by_type[dd] = dists[d]
            for a in angles:
                aa = merge_angle_types(a)
                if aa in angle_by_type:
                    angle_by_type[aa].extend(angles[a])
                else:
                    angle_by_type[aa] = angles[a]

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 8),
    )
    # Plot distributions of each bond type.
    for btype in distance_by_type:
        if "BC" in btype:
            continue
        data = distance_by_type[btype]
        axs[0].hist(
            x=data,
            bins=50,
            range=(3.6, 4.4),
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
    for btype in distance_by_type:
        if "BC" not in btype:
            continue
        data = distance_by_type[btype]
        aniso = float(btype.replace("BC", ""))
        axs[1].scatter(
            x=[aniso for i in data],
            y=data,
            color="gray",
            s=30,
            alpha=0.3,
            rasterized=True,
        )
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("anisotropy", fontsize=16)
    axs[1].set_ylabel(r"F1-F2 distance [$\mathrm{\AA}}$]", fontsize=16)

    # Plot distributions of each angle type.
    for atype in angle_by_type:
        data = angle_by_type[atype]
        axs[2].hist(
            x=data,
            bins=50,
            range=(20, 182),
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
    symm_to_c,
    results,
    output_dir,
    filename,
    vmin,
    vmax,
    clabel,
    flex,
):

    fig, ax = plt.subplots(figsize=(8, 8))
    maps = np.zeros((len(symm_to_c), len(results)))
    for j, aniso in enumerate(results):
        da = results[aniso]
        for i, symm in enumerate(symm_to_c):
            if clabel == "energy (eV)":
                maps[i][j] = da[symm]["fin_energy"]
            elif clabel == "CU-8":
                maps[i][j] = da[symm]["cu8"]

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
            y=[symm_to_c[symm][2] for symm in symm_to_c],
            c="white",
            marker="P",
            edgecolors="k",
            s=80,
        )
        # # Max of each row.
        # index_max = np.argmax(maps, axis=1)
        # ax.scatter(
        #     x=index_max,
        #     y=[symm_to_c[symm][2] for symm in symm_to_c],
        #     c='white',
        #     edgecolors='k',
        #     marker='X',
        #     s=40,
        # )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("anisotropy", fontsize=16)
    ax.set_ylabel("symmetry", fontsize=16)
    # Show all ticks and label them with the respective lists.
    ax.set_xticks([i for i in range(len(results))])
    ax.set_xticklabels([a for a in results])
    ax.set_yticks([symm_to_c[symm][2] for symm in symm_to_c])
    ax.set_yticklabels([convert_symm_names(symm) for symm in symm_to_c])

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_title(f"flex: {flex}", fontsize=16)

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
        "td_2.0",
        # 'c2v_1.65',
        "c2v_1.5",
        # 'th2_1.6',
        "th2_1.1",
        # 's61_1.85',
        "s61_1.3",
        "s42_1.05",
    )

    fig, axs = plt.subplots(
        nrows=len(_to_plot),
        ncols=1,
        sharex=True,
        figsize=(8, 10),
    )

    for name, ax in zip(_to_plot, axs):
        symm, aniso = name.split("_")
        da = results[float(aniso)]
        traj = da[symm]["traj"]
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
            x=340,
            y=1100,
            s=f"{convert_symm_names(symm)}, aniso={aniso}",
            fontsize=16,
        )
        if name == "th2_1.1":
            ax.set_ylabel("energy [eV]", fontsize=16)
            ax2.set_ylabel("Gnorm", fontsize=16)

    ax.set_xlabel("step", fontsize=16)
    ax.set_xticks(range(0, 501, 50))
    ax.set_xticklabels([str(i) for i in range(0, 501, 50)])

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
