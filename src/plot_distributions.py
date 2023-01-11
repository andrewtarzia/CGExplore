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
import matplotlib.pyplot as plt
from env_set import cages

from analysis_utilities import (
    write_out_mapping,
    data_to_array,
    convert_topo_to_label,
    convert_torsion_to_label,
    topology_labels,
    target_shapes,
    topo_to_colormap,
    max_energy,
)


def identity_distributions(all_data, figure_output):

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {i: 0 for i in topology_labels(short=True)}
    categories.update({"X opt.": 0})
    count1 = all_data["topology"].value_counts()
    countxopt = all_data["optimised"].value_counts()
    for tstr, count in count1.items():
        categories[convert_topo_to_label(tstr)] = count
    categories["X opt."] = countxopt[False]
    num_cages = len(all_data)

    ax.bar(
        categories.keys(),
        categories.values(),
        # color="#06AED5",
        color="#086788",
        # color="#DD1C1A",
        # color="#320E3B",
        edgecolor="k",
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
    comparisons = {
        "clangle": {
            "measure": "angles",
            "xlabel": "angle [deg]",
            "label_options": (
                "Pb_Pd_Pb",
                "Pb_C_Pb",
                # "C_Pb_Ba",
                # "Pd_Pb_Ba",
            ),
        },
        "clsigma": {
            "measure": "bonds",
            "xlabel": "length [A]",
            "label_options": ("Pd_Pb", "C_Pb"),
        },
        "c2angle": {
            "measure": "angles",
            "xlabel": "angle [deg]",
            "label_options": ("Ba_Ag_Ba", "Pb_Ba_Ag"),
        },
        "c2sigma": {
            "measure": "bonds",
            "xlabel": "length [A]",
            "label_options": ("Pb_Ba", "Ba_Ag"),
        },
        "torsions": {
            "measure": "dihedrals",
            "xlabel": "torsion [deg]",
            "label_options": ("Pb_Ba_Ba_Pb",),
        },
    }

    for comp in comparisons:
        cdict = comparisons[comp]
        color_map = topo_to_colormap()

        for tors in ("ton", "toff"):
            for i, tstr in enumerate(color_map):
                fig, ax = plt.subplots(figsize=(8, 5))
                topo_frame = all_data[all_data["topology"] == tstr]
                # c_lines = list(set(topo_frame[comp]))
                tor_frame = topo_frame[topo_frame["torsions"] == tors]
                tor_names = list(tor_frame["index"])
                tor_energies = list(tor_frame["energy"])

                values = []
                highe_values = []
                for i, j in zip(tor_names, tor_energies):
                    gdata = geom_data[i][cdict["measure"]]
                    for lbl in cdict["label_options"]:
                        if lbl in gdata:
                            lbldata = gdata[lbl]
                            if j < max_energy():
                                values.extend(lbldata)
                            else:
                                highe_values.extend(lbldata)

                ax.hist(
                    x=values,
                    bins=50,
                    density=False,
                    histtype="step",
                    color="r",
                    lw=2,
                    label=f"E < {max_energy()}",
                )
                ax.hist(
                    x=highe_values,
                    bins=50,
                    density=False,
                    histtype="step",
                    color="gray",
                    lw=2,
                    alpha=1.0,
                    label=f"E > {max_energy()}",
                    linestyle="-",
                )

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel(cdict["xlabel"], fontsize=16)
                ax.set_ylabel("count", fontsize=16)
                ax.set_title(tstr, fontsize=16)
                ax.legend(fontsize=16)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        figure_output, f"gd_{tors}_{comp}_{tstr}.pdf"
                    ),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()


def rmsd_distributions(all_data, calculation_dir, figure_output):

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
            bw_method=0.5,
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
            bw_method=0.5,
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
        [convert_topo_to_label(i) for i in tcpos],
        rotation=45,
    )

    ax2.plot((-1, 12), (0, 0), c="k")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    # ax2.set_ylabel("RMSD [A]", fontsize=16)
    ax2.set_xlim(-0.5, 11.5)
    ax2.set_title("ton->toff", fontsize=16)
    ax2.set_xticks([tcpos[i] for i in tcpos])
    ax2.set_xticklabels(
        [convert_topo_to_label(i) for i in tcpos],
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
    to_plot = {
        "energy": {"xtitle": "energy", "xlim": (0, 100)},
        "gnorm": {"xtitle": "gnorm", "xlim": (0, 0.0005)},
        "pore": {"xtitle": "min. distance [A]", "xlim": (0, 20)},
        "min_b2b": {"xtitle": "min. b2b distance [A]", "xlim": (0, 1)},
    }

    color_map = topo_to_colormap()
    color_map.update({"3C1": "k", "4C1": "k"})

    for tp in to_plot:
        fig, ax = plt.subplots(figsize=(16, 5))
        xtitle = to_plot[tp]["xtitle"]
        xlim = to_plot[tp]["xlim"]
        count = 0
        toptions = {}
        for tstr in color_map:
            for tor in ("ton", "toff"):
                color = {"ton": "gray", "toff": "r"}[tor]
                toptions[(tstr, tor)] = (count, color)
                count += 1

        for i, topt in enumerate(toptions):
            if topt[0] in ("3C1", "4C1"):
                topo_frame = all_data[all_data["cltitle"] == topt[0]]
            else:
                topo_frame = all_data[all_data["topology"] == topt[0]]
            fin_frame = topo_frame[topo_frame["torsions"] == topt[1]]
            values = fin_frame[tp]

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
                bw_method=0.5,
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
            [convert_topo_to_label(i) for i in xticks],
            rotation=45,
        )

        for tor, col in (("ton", "gray"), ("toff", "r")):
            ax.scatter(
                None,
                None,
                c=col,
                edgecolor="none",
                s=30,
                alpha=0.2,
                label=convert_torsion_to_label(tor),
            )
        ax.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"sing_{tp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def shape_vector_distributions(all_data, figure_output):
    present_shape_values = target_shapes()
    num_cols = 4
    num_rows = 3

    # color_map = shapevertices_to_colormap()

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
        if nshape in keys:
            filt_data = all_data[all_data[nshape].notna()]
            n_values = list(filt_data[nshape])
            ax.hist(
                x=n_values,
                bins=50,
                density=False,
                histtype="step",
                color="#DD1C1A",
                lw=3,
            )

        if lshape in keys:
            filt_data = all_data[all_data[lshape].notna()]
            l_values = list(filt_data[lshape])
            ax.hist(
                x=l_values,
                bins=50,
                density=False,
                histtype="step",
                color="k",
                lw=2,
                linestyle="--",
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(shape, fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        # ax.set_xlim(0, xmax)
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_vectors.pdf"),
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
    with open(calculation_output / "all_geom.json", "r") as f:
        geom_data = json.load(f)
    logging.info(f"there are {len(all_data)} collected data")
    opt_data = all_data[all_data["optimised"]]
    logging.info(f"there are {len(opt_data)} successfully opted")
    write_out_mapping(opt_data)

    identity_distributions(all_data, figure_output)
    rmsd_distributions(opt_data, calculation_output, figure_output)
    geom_distributions(opt_data, geom_data, figure_output)
    single_value_distributions(opt_data, figure_output)
    shape_vector_distributions(opt_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
