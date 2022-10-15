#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting.

Author: Andrew Tarzia

"""

import os
import numpy as np
import matplotlib.pyplot as plt


plt.set_loglevel("WARNING")


def merge_bond_types(s):

    translation = {
        "NPd": "N-metal",
        "BC": "ligand",
        "CN": "ligand-N",
    }

    return translation[s]


def merge_angle_types(s):

    translation = {
        "BCC": "ligand",
        "NNPd": "N-metal-N",
        "BCN": "ligand-N",
        "CNPd": "ligand-N-metal",
    }

    return translation[s]


def convert_topo_names(topo_str):

    new_names = {
        "m2l4": "M2L4",
        "m3l6": "M3L6",
        "m4l8": "M4L8",
        "m6l12": "M6L12",
        "m12l24": "M12L24",
        "m24l48": "M24L48",
    }

    return new_names[topo_str]


def scatter(
    topo_to_c,
    results,
    ylabel,
    ylim,
    output_dir,
    filename,
):

    fig, ax = plt.subplots(figsize=(8, 5))
    switched_data = {}
    for bastr in results:
        biteangle = int(bastr.replace("ba", ""))
        da = results[bastr]
        for topo_s in da:
            if ylabel == "energy (eV)":
                ys = da[topo_s]["fin_energy"]
            elif ylabel == "CU-8":
                ys = da[topo_s]["cu8"]
                ax.axhline(y=0, lw=2, c="k")
            elif ylabel == "max diameter [A]":
                ys = da[topo_s]["max_opt_diam"]
            elif ylabel == "pore volume [A^3]":
                ys = da[topo_s]["opt_pore_data"]["pore_volume"]
            elif ylabel == "max. pore radius [A]":
                ys = da[topo_s]["opt_pore_data"]["pore_max_rad"]

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
    ax.set_xlabel("bite angle [degrees]", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def ey_vs_property(
    topo_to_c,
    results,
    output_dir,
    xprop,
    xlim,
    xtitle,
    filename,
    ylim,
):

    fig, ax = plt.subplots(figsize=(8, 5))
    for topo_s in topo_to_c:
        x_vals = []
        y_vals = []
        for biteangle in results:
            da = results[biteangle]
            try:
                x_vals.append(da[topo_s][xprop])
            except KeyError:
                x_vals.append(da[topo_s]["opt_pore_data"][xprop])
            y_vals.append(da[topo_s]["fin_energy"])

        ax.scatter(
            x_vals,
            y_vals,
            c=topo_to_c[topo_s][1],
            edgecolor="k",
            s=100,
            alpha=1.0,
            label=convert_topo_names(topo_s),
        )

    ax.legend(fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xtitle, fontsize=16)
    ax.set_ylabel("energy (eV)", fontsize=16)
    ax.set_xlim(xlim)
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

    target_type = "BCN"

    # Collect all values for each bond and angle type.
    distance_by_type = {}
    angle_by_type = {}
    for ffstr in results:
        da = results[ffstr]
        for topo_str in da:
            dists = da[topo_str]["distances"]
            angles = da[topo_str]["angles"]
            for d in dists:
                dd = merge_bond_types(d)
                if dd in distance_by_type:
                    distance_by_type[dd].extend(dists[d])
                else:
                    distance_by_type[dd] = dists[d]

            for a in angles:
                if a == target_type:
                    aa = f"{a}{ffstr}"
                else:
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
        if target_type not in atype:
            continue
        data = angle_by_type[atype]
        biteangle = float(atype.replace(f"{target_type}ba", ""))
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
    axs[1].set_ylabel(f"{target_type} [degrees]", fontsize=16)

    # Plot distributions of each angle type.
    for atype in angle_by_type:
        if target_type in atype:
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
    print("add a property to heatmap too.")
    fig, ax = plt.subplots(figsize=(8, 8))
    maps = np.zeros((len(topo_to_c), len(results)))
    for j, biteangle in enumerate(results):
        da = results[biteangle]
        for i, topo_s in enumerate(topo_to_c):
            if clabel == "energy (eV)":
                maps[i][j] = da[topo_s]["fin_energy"]
            elif clabel == "CU-8":
                maps[i][j] = da[topo_s]["cu8"]

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
    ax.set_xlabel("bite angle [degrees]", fontsize=16)
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


def convergence(
    results,
    output_dir,
    filename,
):

    # Pick examples to plot.
    _to_plot = (
        "m2l4_20",
        "m3l6_70",
        "m6l12_70",
        "m12l24_120",
        "m24l48_80",
    )

    fig, axs = plt.subplots(
        nrows=len(_to_plot),
        ncols=1,
        sharex=True,
        figsize=(8, 10),
    )

    xmax, xwin = (501, 50)

    for name, ax in zip(_to_plot, axs):
        topo_s, biteangle = name.split("_")
        da = results[f"ba{biteangle}"]
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
            s=f"{convert_topo_names(topo_s)}, bite-angle={biteangle}",
            fontsize=16,
        )
        if name == _to_plot[0]:
            ax.set_ylabel("energy [eV]", fontsize=16)
            ax2.set_ylabel("Gnorm", fontsize=16)

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


def md_output_plot(
    topo_to_c,
    results,
    output_dir,
    filename,
    y_column,
):

    topo_to_ax = {j: i for i, j in enumerate(topo_to_c)}
    ba_to_ax = {j: i for i, j in enumerate(results)}
    ycol_to_label = {
        "E": "energy [eV]",
        "T": "temperature [K]",
        "KE": "kinetic energy [eV]",
        "rmsd": "RMSD [A]",
        "num_windows": "num. windows",
        "blob_max_diam": "max blob diam [A]",
        "pore_max_rad": "max pore radius [A]",
        "pore_mean_rad": "mean pore radius [A]",
        "pore_volume": "pore volume [A^3]",
        "asphericity": "pore asphericity",
        "shape": "pore shape",
    }
    ycol_to_lim = {
        "E": (0, 10),
        "T": None,
        "KE": (0, 4),
        "rmsd": (0, 2),
        "num_windows": (0, 30),
        "blob_max_diam": (0, 80),
        "pore_max_rad": (0, 20),
        "pore_mean_rad": (0, 20),
        "pore_volume": (0, 20000),
        "asphericity": (0, 20),
        "shape": (0, 0.1),
    }

    fig, axs = plt.subplots(
        nrows=len(topo_to_c),
        ncols=len(results),
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )

    for ba in results:
        badict = results[ba]
        for topo_str in badict:
            ax = axs[topo_to_ax[topo_str]][ba_to_ax[ba]]
            tdict = badict[topo_str]
            tdata = tdict["mdtraj"]
            xs = []
            ys = []
            for step in tdata:
                sdict = tdata[step]
                xs.append(sdict["time"])
                ys.append(sdict[y_column])

            if topo_str == "m2l4":
                ax.set_title(ba, fontsize=16)

            ax.scatter(
                xs,
                ys,
                s=40,
                # marker='o',
                # markersize=12,
                color=topo_to_c[topo_str][1],
                rasterized=True,
            )
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_ylim(ycol_to_lim[y_column])

    axs[-1][0].set_xlabel("time []", fontsize=16)
    axs[-1][0].set_ylabel(ycol_to_label[y_column], fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
