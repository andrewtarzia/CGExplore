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
import logging
import pandas as pd
import matplotlib.pyplot as plt

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
    topo_to_colormap,
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
        "clsigma": {
            "measure": "bonds",
            "xlabel": "CL bond length [A]",
            "column": "clsigma",
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
        "c2sigma": {
            "measure": "bonds",
            "xlabel": "C2 backbone length [A]",
            "column": "c2sigma",
            "label_options": ("Ba_Ag",),
        },
        "c2bonder": {
            "measure": "bonds",
            "xlabel": "C2 bonder length [A]",
            "column": None,
            "label_options": ("Pb_Ba",),
        },
    }

    tcmap = topo_to_colormap()
    tcpos = {tstr: i for i, tstr in enumerate(tcmap)}

    for comp in comparisons:
        cdict = comparisons[comp]
        color_map = topo_to_colormap()
        column = cdict["column"]

        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 5))
        for ax, tors in zip(axs, ("ton", "toff")):
            tor_frame = all_data[all_data["torsions"] == tors]
            for i, tstr in enumerate(color_map):
                topo_frame = tor_frame[tor_frame["topology"] == tstr]

                values = []
                # energies = []
                for i, row in topo_frame.iterrows():
                    name = str(row["index"])
                    # energy = float(row["energy_per_bond"])
                    if column is None:
                        target = 0
                        target_oppos = None
                    else:
                        target = float(row[column])
                        if "4C1" in name and column == "clangle":
                            target_oppos = convert_pyramid_angle(target)
                        else:
                            target_oppos = None

                    # Ignore torsions for target bite-angles near 180.
                    if comp == "torsions":
                        tba = float(row["target_bite_angle"])
                        if tba == 180:
                            continue

                    obsdata = geom_data[name][cdict["measure"]]
                    for lbl in cdict["label_options"]:
                        if lbl in obsdata:
                            lbldata = obsdata[lbl]
                            for observed in lbldata:
                                if pd.isna(observed):
                                    continue

                                # For some comparisons, need to use
                                # matching rules.
                                if comp == "clsigma":
                                    new_target = (target + 1) / 2
                                    value = observed - new_target
                                elif comp == "c2sigma":
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

                ax.scatter(
                    [xpos for i in values],
                    [i for i in values],
                    c="gray",
                    # c=[i for i in energies],
                    edgecolor="none",
                    s=20,
                    alpha=0.2,
                    rasterized=True,
                    # vmin=0,
                    # vmax=vmax,
                    # cmap="Blues",
                )
                if column is not None:
                    ax.axhline(y=0, lw=2, c="k", linestyle="--")

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
                    pc.set_facecolor("gray")
                    pc.set_edgecolor("none")
                    pc.set_alpha(0.3)

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_title(
                (
                    f'{cdict["xlabel"]}: '
                    f"{convert_tors(tors,num=False)}"
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

        # cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        # cmap = mpl.cm.Blues
        # norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        # cbar = fig.colorbar(
        #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        #     cax=cbar_ax,
        #     orientation="vertical",
        # )
        # cbar.ax.tick_params(labelsize=16)
        # cbar.set_label("energy per bond formed [eV]", fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"gd_{comp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def rmsd_distributions(all_data, calculation_dir, figure_output):
    logging.info("running rmsd_distributions")

    tcmap = topo_to_colormap()

    rmsd_file = calculation_dir / "all_rmsds.json"
    if os.path.exists(rmsd_file):
        with open(rmsd_file, "r") as f:
            data = json.load(f)
    else:
        data = {}
        for tstr in tcmap:
            tdata = {}
            for o1 in calculation_dir.glob(f"*{tstr}_*ton*_opted2.mol"):
                if o1.name[0] in ("2", "3", "4"):
                    continue

                o1 = str(o1)
                o2 = o1.replace("opted2", "opted1")
                o3 = o1.replace("opted2", "opted3")
                ooff = o1.replace("ton", "toff")
                mol1 = stk.BuildingBlock.init_from_file(o1)
                mol2 = stk.BuildingBlock.init_from_file(o2)

                moloff = stk.BuildingBlock.init_from_file(ooff)

                rmsd_calc = stko.RmsdCalculator(mol1)
                # try:
                rmsd1 = rmsd_calc.get_results(mol2).get_rmsd()
                rmsdooff = rmsd_calc.get_results(moloff).get_rmsd()
                if os.path.exists(o3):
                    mol3 = stk.BuildingBlock.init_from_file(o3)
                    rmsd3 = rmsd_calc.get_results(mol3).get_rmsd()
                else:
                    rmsd3 = None
                # except stko.DifferentMoleculeException:
                #     logging.info(f"fail for {o1}, {o2}")

                tdata[o1] = (rmsd1, rmsdooff, rmsd3)
            data[tstr] = tdata

        with open(rmsd_file, "w") as f:
            json.dump(data, f)

    tcpos = {tstr: i for i, tstr in enumerate(tcmap)}
    lentcpos = len(tcpos)
    tcmap.update({"all": "k"})
    tcpos.update({"all": lentcpos})

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(16, 5),
    )
    # flat_axs = axs.flatten()

    ax = axs[0]
    ax2 = axs[1]
    allydata1 = []
    allydata2 = []
    for tstr in tcmap:
        if tstr == "all":
            ydata1 = allydata1
            ydata2 = allydata2
        else:
            ydata1 = [i[0] for i in data[tstr].values()]
            ydata2 = [i[1] for i in data[tstr].values()]
            ydata3 = [
                i[2] for i in data[tstr].values() if i[2] is not None
            ]
            allydata1.extend(ydata1)
            allydata2.extend(ydata2)

        xpos = tcpos[tstr]

        ax.scatter(
            [xpos for i in ydata1],
            [i for i in ydata1],
            c="gray",
            edgecolor="none",
            s=30,
            alpha=0.2,
            rasterized=True,
        )

        parts = ax.violinplot(
            [i for i in ydata1],
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
            pc.set_facecolor("gray")
            pc.set_edgecolor("none")
            pc.set_alpha(0.3)

        ax2.scatter(
            [xpos for i in ydata2],
            [i for i in ydata2],
            c="gray",
            edgecolor="none",
            s=30,
            alpha=0.2,
            rasterized=True,
        )

        parts = ax2.violinplot(
            [i for i in ydata2],
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
            pc.set_facecolor("gray")
            pc.set_edgecolor("none")
            pc.set_alpha(0.3)

        if tstr != "all":
            ax.scatter(
                [xpos for i in ydata3],
                [i for i in ydata3],
                c="gold",
                edgecolor="none",
                s=20,
                alpha=1.0,
                rasterized=True,
            )

    ax.plot((-1, 12), (0, 0), c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("RMSD [A]", fontsize=16)
    ax.set_xlim(-0.5, 11.5)
    ax.set_title("opt1->opt2; opt2->opt3", fontsize=16)
    ax.set_xticks([tcpos[i] for i in tcpos])
    ax.set_xticklabels(
        [convert_topo(i) for i in tcpos],
        rotation=45,
    )

    ax2.plot((-1, 12), (0, 0), c="k")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    # ax2.set_ylabel("RMSD [A]", fontsize=16)
    ax2.set_xlim(-0.5, 11.5)
    ax2.set_title("ton->toff", fontsize=16)
    ax2.set_xticks([tcpos[i] for i in tcpos])
    ax2.set_xticklabels(
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
        "energy_per_bond": {
            "xtitle": "E_bf [kJmol-1]",
            "xlim": (0, 150),
        },
        "pore": {"xtitle": "min. distance [A]", "xlim": (0, 20)},
        "min_b2b_distance": {
            "xtitle": "min. b2b distance [A]",
            "xlim": (0, 1),
        },
        "HarmonicBondForce_kjmol": {
            "xtitle": "E_HB [kJmol-1]",
            "xlim": (0, 1000),
        },
        "HarmonicAngleForce_kjmol": {
            "xtitle": "E_HA [kJmol-1]",
            "xlim": (0, 1000),
        },
        "CustomNonbondedForce_kjmol": {
            "xtitle": "E_NB [kJmol-1]",
            "xlim": (0, 1000),
        },
        "PeriodicTorsionForce_kjmol": {
            "xtitle": "E_PT [kJmol-1]",
            "xlim": (0, 1000),
        },
        "radius_gyration": {
            "xtitle": "R_g [A]",
            "xlim": (0, 2000),
        },
        "max_diameter": {
            "xtitle": "max. diameter [A]",
            "xlim": (0, 2000),
        },
        "rg/md": {"xtitle": "R_g / max. diameter", "xlim": (0, 200)},
        "flexibility_measure": {
            "xtitle": "flexibility $f$",
            "xlim": (0, 1000),
        },
    }

    topologies = topology_labels(short="P")
    color_map = {
        ("ton", "von"): "plum",
        ("ton", "voff"): "teal",
        ("toff", "von"): "magenta",
        ("toff", "voff"): "skyblue",
    }

    for tp in to_plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        xtitle = to_plot[tp]["xtitle"]
        xlim = to_plot[tp]["xlim"]
        count = 0
        toptions = {}
        for tstr in topologies:
            for tor in ("ton", "toff"):
                for vdw in ("von", "voff"):
                    color = color_map[(tor, vdw)]
                    toptions[(tstr, tor, vdw)] = (count, color)
                    count += 1

        for i, topt in enumerate(toptions):
            topo_frame = all_data[all_data["topology"] == topt[0]]
            tor_frame = topo_frame[topo_frame["torsions"] == topt[1]]
            fin_frame = tor_frame[tor_frame["vdws"] == topt[2]]
            values = fin_frame[tp]
            if len(values) == 0:
                continue

            xpos = toptions[topt][0]
            col = toptions[topt][1]

            ax.scatter(
                [xpos for i in values],
                [i for i in values],
                c=col,
                edgecolor="none",
                s=30,
                alpha=0.2,
                rasterized=True,
            )

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
                pc.set_facecolor(col)
                pc.set_edgecolor("none")
                pc.set_alpha(0.3)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel(xtitle, fontsize=16)
        ax.set_ylim(xlim)
        xticks = {}
        for i in toptions:
            tstr = i[0]
            if tstr in xticks:
                xticks[tstr] = (toptions[i][0] + xticks[tstr]) / 2
            else:
                xticks[tstr] = toptions[i][0]

        ax.set_xticks([xticks[i] for i in xticks])
        ax.set_xticklabels(
            [convert_topo(i) for i in xticks],
            rotation=45,
        )

        for lblkey in color_map:
            tor, vdw = lblkey
            col = color_map[lblkey]
            ax.scatter(
                None,
                None,
                c=col,
                edgecolor="none",
                s=30,
                alpha=0.2,
                label=f"{convert_tors(tor)},{convert_vdws(vdw)}",
            )
        ax.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"sing_{tp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()
        raise SystemExit()


def shape_vector_distributions(all_data, figure_output):
    logging.info("running shape_vector_distributions")

    present_shape_values = target_shapes()
    num_cols = 5
    num_rows = 2

    torsion_dict = {
        "ton": "-",
        "toff": "--",
    }

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    for i, shape in enumerate(present_shape_values):
        # nv = int(shape.split("-")[1])
        # c = color_map[nv]
        ax = flat_axs[i]

        keys = tuple(all_data.columns)
        nshape = f"n_{shape}"
        lshape = f"l_{shape}"

        for tors in torsion_dict:
            tor_data = all_data[all_data["torsions"] == tors]
            if nshape in keys:
                filt_data = tor_data[tor_data[nshape].notna()]
                n_values = list(filt_data[nshape])
                ax.hist(
                    x=n_values,
                    bins=50,
                    density=False,
                    histtype="step",
                    color="#DD1C1A",
                    lw=3,
                    linestyle=torsion_dict[tors],
                    label=f"node: {convert_tors(tors, num=False)}",
                )

            if lshape in keys:
                filt_data = tor_data[tor_data[lshape].notna()]
                l_values = list(filt_data[lshape])
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

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(shape, fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        # ax.set_xlim(0, xmax)
        ax.set_yscale("log")

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
    tstrs = topo_to_colormap()
    ntstrs = mapshape_to_topology(mode="n")
    ltstrs = mapshape_to_topology(mode="l")

    torsion_dict = {
        "ton": "-",
        "toff": "--",
    }

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
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
        ax.set_ylabel("log(count)", fontsize=16)
        ax.set_title(tstr, fontsize=16)
        ax.set_yscale("log")
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
    tstrs = topo_to_colormap()
    ntstrs = mapshape_to_topology(mode="n")
    ltstrs = mapshape_to_topology(mode="l")

    torsion_dict = {
        "ton": "-",
        "toff": "--",
    }

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
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

        ax_xlbl = {}
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
        ax.set_ylabel("log(count)", fontsize=16)
        ax.set_title(tstr, fontsize=16)
        ax.set_yscale("log")
        if len(mtstrs[tstr]) == 2:
            ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_vectors_3.pdf"),
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
    write_out_mapping(opt_data)

    identity_distributions(all_data, figure_output)
    single_value_distributions(all_data, figure_output)
    raise SystemExit()
    shape_vector_distributions(all_data, figure_output)
    shape_vectors_2(all_data, figure_output)
    shape_vectors_3(all_data, figure_output)
    rmsd_distributions(all_data, calculation_output, figure_output)
    geom_distributions(all_data, geom_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
