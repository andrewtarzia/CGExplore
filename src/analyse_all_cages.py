#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

Author: Andrew Tarzia

"""

import sys
import os
import stk
import stko
import json
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from sklearn.cluster import KMeans
import pandas as pd

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from generate_all_cages import (
    core_2c_beads,
    arm_2c_beads,
    beads_3c,
    beads_4c,
)
from env_set import cages


def topology_labels(short=False):
    if short:
        return (
            "2+3",
            "4+6",
            "4+6(2)",
            "6+9",
            "8+12",
            "2+4",
            "3+6",
            "4+8",
            "6+12",
        )
    else:
        return (
            "TwoPlusThree",
            "FourPlusSix",
            "FourPlusSix2",
            "SixPlusNine",
            "EightPlusTwelve",
            "TwoPlusFour",
            "ThreePlusSix",
            "FourPlusEight",
            "SixPlusTwelve",
        )


def convert_topo_to_label(topo_str):
    return {
        "TwoPlusThree": "2+3",
        "FourPlusSix": "4+6",
        "FourPlusSix2": "4+6(2)",
        "SixPlusNine": "6+9",
        "EightPlusTwelve": "8+12",
        "TwoPlusFour": "2+4",
        "ThreePlusSix": "3+6",
        "FourPlusEight": "4+8",
        "SixPlusTwelve": "6+12",
        "mixed": "mixed",
        "unstable": "unstable",
    }[topo_str]


def topo_to_colormap():
    return {
        "TwoPlusThree": "#06AED5",
        "FourPlusSix": "#086788",
        "FourPlusSix2": "#DD1C1A",
        "SixPlusNine": "#320E3B",
        "EightPlusTwelve": "#CE7B91",
        "TwoPlusFour": "#6969B3",
        "ThreePlusSix": "#B279A7",
        "FourPlusEight": "#C3423F",
        "SixPlusTwelve": "#9BC53D",
    }


def cltype_to_colormap():
    return {
        "3C1": "#06AED5",
        "4C1": "#086788",
    }


def cltypetopo_to_colormap():
    return {
        "3C1": {
            "TwoPlusThree": "#1f77b4",
            "FourPlusSix": "#ff7f0e",
            "FourPlusSix2": "#2ca02c",
            "SixPlusNine": "#d62728",
            "EightPlusTwelve": "#17becf",
        },
        "4C1": {
            "TwoPlusFour": "#aec7e8",
            "ThreePlusSix": "#ffbb78",
            "FourPlusEight": "#98df8a",
            "SixPlusTwelve": "#ff9896",
        },
        "mixed": {
            "2": "#7b4173",
            ">2": "#de9ed6",
        },
    }


def shapevertices_to_colormap():
    return {
        4: "#06AED5",
        5: "#086788",
        6: "#DD1C1A",
        8: "#320E3B",
        3: "#6969B3",
    }


def shapelabels_to_colormap():
    return {
        "3": "#F9A03F",
        "4": "#0B2027",
        "5": "#86626E",
        "6": "#CA1551",
        "8": "#345995",
        "b": "#7A8B99",
    }


def target_shapes():
    return (
        "CU-8",
        "JETBPY-8",
        "OP-8",
        "OC-6",
        "PPY-6",
        "HP-6",
        "TBPY-5",
        "PP-5",
        "T-4",
        "SP-4",
        "TP-3",
        "mvOC-3",
    )


def target_shapes_by_cltype(cltype):
    if cltype == "4C1":
        return ("OC-6b", "TP-3", "SP-4", "OC-6")
    elif cltype == "3C1":
        return ("TBPY-5", "T-4", "T-4b", "TPR-6", "CU-8")


def shapetarget_to_colormap():
    return {
        "CU-8": "#06AED5",
        "OC-6": "#086788",
        "TBPY-5": "#DD1C1A",
        "T-4": "#320E3B",
        "TPR-6": "#CE7B91",
    }


def map_cltype_to_shapetopology():
    return {
        "3C1": {
            "5": "TwoPlusThree",
            "4": "FourPlusSix",
            "b": "FourPlusSix2",
            "6": "SixPlusNine",
            "8": "EightPlusTwelve",
        },
        "4C1": {
            "b": "TwoPlusFour",
            "3": "ThreePlusSix",
            "4": "FourPlusEight",
            "6": "SixPlusTwelve",
        },
    }


def mapshape_to_topology(reverse=False):
    if reverse:
        return {
            "TwoPlusThree": "TBPY-5",
            "FourPlusSix": "T-4",
            "FourPlusSix2": "T-4b",
            "SixPlusNine": "TPR-6",
            "EightPlusTwelve": "CU-8",
            "TwoPlusFour": "OC-6b",
            "ThreePlusSix": "TP-3",
            "FourPlusEight": "SP-4",
            "SixPlusTwelve": "OC-6",
        }
    else:
        return {
            "TBPY-5": "TwoPlusThree",
            "T-4": "FourPlusSix",
            "T-4b": "FourPlusSix2",
            "TPR-6": "SixPlusNine",
            "CU-8": "EightPlusTwelve",
            "OC-6b": "TwoPlusFour",
            "TP-3": "ThreePlusSix",
            "SP-4": "FourPlusEight",
            "OC-6": "SixPlusTwelve",
        }


def collate_cage_vector_from_bb(data, test_bb):
    tbb_dict = {}
    for cage in data:
        if test_bb in cage:
            cage_dict = data[cage]
            for sv in cage_dict["shape_vector"]:
                if "FourPlusSix2" in cage:
                    svb = sv + "b"
                else:
                    svb = sv
                tbb_dict[svb] = cage_dict["shape_vector"][sv]
    return tbb_dict


def get_CGBead_from_string(string, bead_library):
    return tuple(i for i in bead_library if i.element_string == string)[
        0
    ]


def get_shape_vector(shape_dictionary):
    shape_vector = {}
    for i in target_shapes():
        if i in shape_dictionary:
            shape_vector[i] = shape_dictionary[i][0]
        if i + "b" in shape_dictionary:
            shape_vector[i + "b"] = shape_dictionary[i + "b"][0]

    return shape_vector


def get_present_beads(c2_bbname):
    wtopo = c2_bbname[3:]
    present_beads_names = []
    while len(wtopo) > 0:
        if len(wtopo) == 1:
            bead_name = wtopo[0]
            wtopo = ""
        elif wtopo[1].islower():
            bead_name = wtopo[:2]
            wtopo = wtopo[2:]
        else:
            bead_name = wtopo[0]
            wtopo = wtopo[1:]

        present_beads_names.append(bead_name)

    if len(present_beads_names) != int(c2_bbname[2]) + 1:
        raise ValueError(f"{present_beads_names} length != {c2_bbname}")
    return present_beads_names


def identity_distributions(all_data, figure_output):

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {i: 0 for i in topology_labels(short=True)}
    num_cages = len(all_data)
    for cage in all_data:
        topo_name = cage.split("_")[0]
        categories[convert_topo_to_label(topo_name)] += 1

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


def rmsd_distributions(all_data, calculation_dir, figure_output):

    rmsd_file = calculation_dir / "all_rmsds.json"

    if os.path.exists(rmsd_file):
        with open(rmsd_file, "r") as f:
            data = json.load(f)
    else:
        data = {}
        for o1 in calculation_dir.glob("*_opted1.mol"):
            if o1.name[0] in ("2", "3", "4"):
                continue

            o1 = str(o1)
            o2 = o1.replace("opted1", "opted2")
            o3 = o1.replace("opted1", "opted3")
            try:
                mol1 = stk.BuildingBlock.init_from_file(o1)
                mol2 = stk.BuildingBlock.init_from_file(o2)
                mol3 = stk.BuildingBlock.init_from_file(o3)
            except OSError:
                pass
            rmsd_calc = stko.RmsdCalculator(mol1)
            rmsd_calc2 = stko.RmsdCalculator(mol2)
            try:
                rmsd2 = rmsd_calc.get_results(mol2).get_rmsd()
                rmsd3 = rmsd_calc.get_results(mol3).get_rmsd()
                rmsd4 = rmsd_calc2.get_results(mol3).get_rmsd()

            except stko.DifferentMoleculeException:
                logging.info(f"fail for {o1}, {o2}, {o3}")

            data[o1] = (rmsd2, rmsd3, rmsd4)

        with open(rmsd_file, "w") as f:
            json.dump(data, f)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        [i[0] for i in data.values() if i[0] < 10],
        bins=50,
        alpha=0.5,
        edgecolor="k",
        label="opt1->MD",
    )
    ax.hist(
        [i[1] for i in data.values() if i[1] < 10],
        bins=50,
        alpha=0.5,
        edgecolor="k",
        label="opt1->opt2",
    )
    ax.hist(
        [i[2] for i in data.values() if i[2] < 10],
        bins=50,
        alpha=0.5,
        edgecolor="k",
        label="MD->opt2",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("RMSD [A]", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "all_rmsds.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def single_value_distributions(all_data, figure_output):
    to_plot = {
        "energy": {"xtitle": "energy"},
        "pore_diameter": {"xtitle": "pore diameter"},
        "pore_volume": {"xtitle": "pore volume"},
        "energy_barm": {"xtitle": "energy"},
        "pore_diameter_barm": {"xtitle": "pore diameter"},
        "pore_volume_barm": {"xtitle": "pore volume"},
    }

    color_map = topo_to_colormap()
    color_map_barm = cltype_to_colormap()

    for tp in to_plot:

        xtitle = to_plot[tp]["xtitle"]
        if "barm" in tp:
            cmapp = color_map_barm
            fig, axs = plt.subplots(
                ncols=2,
                nrows=1,
                figsize=(16, 5),
            )
            flat_axs = axs.flatten()
        else:
            cmapp = color_map
            fig, axs = plt.subplots(
                ncols=3,
                nrows=3,
                figsize=(16, 10),
            )
            flat_axs = axs.flatten()

        for i, topo in enumerate(cmapp):
            if "barm" in tp:
                values = [
                    all_data[i][tp.replace("_barm", "")]
                    for i in all_data
                    if topo in i.split("_")[1]
                ]
            else:
                values = [
                    all_data[i][tp]
                    for i in all_data
                    if i.split("_")[0] == topo
                ]

            flat_axs[i].hist(
                x=values,
                bins=50,
                density=False,
                histtype="step",
                color=cmapp[topo],
                lw=3,
                label=topo,
            )

            flat_axs[i].tick_params(
                axis="both",
                which="major",
                labelsize=16,
            )
            flat_axs[i].set_xlabel(xtitle, fontsize=16)
            # flat_axs[i].set_ylabel("count", fontsize=16)
            flat_axs[i].set_ylabel("log(count)", fontsize=16)
            flat_axs[i].set_title(topo, fontsize=16)
            flat_axs[i].set_yscale("log")
            # ax.legend(ncol=2, fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"sing_{tp}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def shape_vector_distributions(all_data, figure_output):
    present_shape_values = sorted(
        set([j for i in all_data.values() for j in i["shape_vector"]]),
        key=lambda i: int(i[-1]),
    )
    num_cols = 4
    num_rows = 8

    color_map = shapevertices_to_colormap()
    xmax = 50

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    for i, shape in enumerate(present_shape_values):
        nv = int(shape.split("-")[1])
        c = color_map[nv]
        ax = flat_axs[i]

        values = []
        for i in all_data:
            if shape in all_data[i]["shape_vector"]:
                values.append(all_data[i]["shape_vector"][shape])

        ax.hist(
            x=values,
            bins=50,
            density=False,
            histtype="step",
            color=c,
            lw=3,
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(shape, fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        ax.set_xlim(0, xmax)
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "shape_vectors.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def shape_vector_cluster(all_data, c2bb, c3bb, figure_output):
    fig, ax = plt.subplots(figsize=(8, 5))

    if c3bb is None and c2bb is None:
        raise ValueError("atleast one c2bb, or c3bb should be not None")
    elif c3bb is None:
        bb_data = {
            i: all_data[i]
            for i in all_data
            if all_data[i]["c2bb"] == c2bb
        }
        list_of_test_bbs = set([i["c3bb"] for i in bb_data.values()])
    elif c2bb is None:
        bb_data = {
            i: all_data[i]
            for i in all_data
            if all_data[i]["c3bb"] == c3bb
        }
        list_of_test_bbs = set([i["c2bb"] for i in bb_data.values()])
    logging.info(f"trimmed to {len(bb_data)} points")
    logging.info(f"with {len(list_of_test_bbs)} test building blocks")

    shape_array = {}
    target_row_names = set()
    color_map = shapetarget_to_colormap()
    min_of_each_shape_plot = {i: None for i in target_shapes()}
    for test_bb in list_of_test_bbs:
        tbb_dict = {}
        for cage in bb_data:
            if test_bb in cage:
                cage_dict = bb_data[cage]
                for sv in cage_dict["shape_vector"]:
                    if "FourPlusSix2" in cage:
                        svb = sv + "b"
                    else:
                        svb = sv
                    tbb_dict[svb] = cage_dict["shape_vector"][sv]
                    target_row_names.add(svb)
                    if sv in target_shapes():
                        if min_of_each_shape_plot[sv] is None:
                            min_of_each_shape_plot[sv] = (
                                test_bb,
                                cage_dict["shape_vector"][sv],
                            )
                        elif (
                            cage_dict["shape_vector"][sv]
                            < min_of_each_shape_plot[sv][1]
                        ):
                            min_of_each_shape_plot[sv] = (
                                test_bb,
                                cage_dict["shape_vector"][sv],
                            )

        shape_array[test_bb] = tbb_dict

    data_array = pd.DataFrame.from_dict(
        shape_array,
        orient="index",
    ).reset_index()

    # Separating out the features
    x = data_array.loc[:, target_row_names].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    pc_df = pd.DataFrame(
        data=pcs,
        columns=["pc1", "pc2"],
    )
    # pcindexed_df = pd.concat(
    #     [pc_df, data_array[["index"]]],
    #     axis=1,
    # )
    ax.scatter(
        pc_df["pc1"],
        pc_df["pc2"],
        c="grey",
        edgecolor="none",
        s=30,
        alpha=1.0,
    )

    # # Initialize the class object
    # kmeans = KMeans(n_clusters=len(target_row_names))

    # # predict the labels of clusters.
    # label = kmeans.fit_predict(pc_df)

    # # Getting unique labels
    # centroids = kmeans.cluster_centers_
    # u_labels = np.unique(label)

    # # plotting the results:
    # for i in u_labels:
    #     ax.scatter(
    #         pcs[label == i, 0],
    #         pcs[label == i, 1],
    #         label=i,
    #         s=20,
    #     )
    # ax.scatter(
    #     centroids[:, 0],
    #     centroids[:, 1],
    #     s=40,
    #     color="white",
    #     edgecolor="k",
    # )

    # cluster_map = pd.DataFrame()
    # cluster_map["data_index"] = pc_df.index.values
    # cluster_map["cluster"] = kmeans.labels_

    for shape in min_of_each_shape_plot:
        min_c3bb, sv = min_of_each_shape_plot[shape]
        logging.info(
            f"for {shape}, A(!) min bb pair: {c2bb}/{min_c3bb} "
            f"with value {sv}"
        )
        data_index = data_array.index[
            data_array["index"] == min_c3bb
        ].tolist()
        ax.scatter(
            pcs[data_index[0], 0],
            pcs[data_index[0], 1],
            c=color_map[shape],
            label=shape,
            marker="D",
            edgecolor="k",
            s=50,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("principal component 1", fontsize=16)
    ax.set_ylabel("principal component 2", fontsize=16)
    ax.set_title(f"PCA: {c2bb}/{c3bb}", fontsize=16)
    ax.legend(ncol=4, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, f"cluster_{c2bb}_{c3bb}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_1(bb_data, figure_output):
    t_map = map_cltype_to_shapetopology()
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    data_dict = {}
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]
        c2_bbname = bb_pair[0]
        present_beads_names = get_present_beads(c2_bbname)

        x = get_CGBead_from_string(
            present_beads_names[0], core_2c_beads()
        ).sigma
        # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
        y = (
            get_CGBead_from_string(
                present_beads_names[-1], arm_2c_beads()
            ).angle_centered
            - 90
        ) * 2

        min_energy = min(tuple(i[1] for i in b_dict.values()))
        min_e_dict = {
            i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
        }
        keys = list(min_e_dict.keys())
        min_energy_topo = keys[0][-1]

        if "3C1" in cl_bbname:
            if t_map["3C1"][min_energy_topo] not in data_dict:
                data_dict[t_map["3C1"][min_energy_topo]] = []
            data_dict[t_map["3C1"][min_energy_topo]].append((x, y))
        elif "4C1" in cl_bbname:
            if t_map["4C1"][min_energy_topo] not in data_dict:
                data_dict[t_map["4C1"][min_energy_topo]] = []
            data_dict[t_map["4C1"][min_energy_topo]].append((x, y))

    for ax, t_str in zip(flat_axs, topology_labels()):
        ax.scatter(
            [i[0] for i in data_dict[t_str]],
            [i[1] for i in data_dict[t_str]],
            c="gray",
            edgecolor="none",
            s=30,
            alpha=0.2,
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)
        ax.set_title(t_str, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_2(bb_data, figure_output):
    for i in ("shape", "energy", "se"):
        ncols = 2

        if ncols == 1:
            fig, ax = plt.subplots(
                figsize=(8, 5),
            )
            flat_axs = [ax]
        else:
            fig, axs = plt.subplots(
                nrows=1,
                ncols=ncols,
                figsize=(16, 5),
            )
            flat_axs = axs.flatten()

        sc_3c0_2c1 = []
        sc_4c0_2c1 = []
        for bb_pair in bb_data:
            bb_dict = bb_data[bb_pair]
            # c2_bbname = bb_pair[0]
            cl_bbname = bb_pair[1]

            min_energy = min(tuple(i[1] for i in bb_dict.values()))
            min_e_dict = {
                i: bb_dict[i]
                for i in bb_dict
                if bb_dict[i][1] == min_energy
            }
            min_shape_value = min(
                (i[0] for i in list(min_e_dict.values()))
            )
            if i == "energy":
                x = min_energy
                y = list(min_e_dict.values())[0][2]
            elif i == "shape":
                x = min_shape_value
                y = list(min_e_dict.values())[0][2]
            elif i == "se":
                x = min_shape_value
                y = min_energy

            if "3C1" in cl_bbname:
                sc_3c0_2c1.append((x, y))
            elif "4C1" in cl_bbname:
                sc_4c0_2c1.append((x, y))

        shape_coords = (
            # (f"3C1-2C1 ({len(sc_3c0_2c0)})", sc_3c0_2c0),
            (f"3C1-2C1 ({len(sc_3c0_2c1)})", sc_3c0_2c1),
            (f"4C1-2C1 ({len(sc_4c0_2c1)})", sc_4c0_2c1),
            # (f"3C1-2C3 ({len(sc_3c0_2c3)})", sc_3c0_2c3),
        )

        for ax, (title, coords) in zip(flat_axs, shape_coords):

            if len(coords) != 0:
                hb = ax.hexbin(
                    [i[0] for i in coords],
                    [i[1] for i in coords],
                    gridsize=20,
                    cmap="inferno",
                    bins="log",
                )
                cbar = fig.colorbar(hb, ax=ax, label="log10(N)")

            cbar.ax.tick_params(labelsize=16)
            ax.set_title(title, fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=16)
            if i == "energy":
                ax.set_xlabel("min. energy", fontsize=16)
                ax.set_ylabel("pore radius", fontsize=16)
            elif i == "shape":
                ax.set_xlabel("min. shape", fontsize=16)
                ax.set_ylabel("pore radius", fontsize=16)
            elif i == "se":
                ax.set_xlabel("min. shape", fontsize=16)
                ax.set_ylabel("min. energy", fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"ps_2_{i}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def phase_space_3(bb_data, figure_output):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    isomer_energy = 0.05
    max_energy = 10

    topologies = map_cltype_to_shapetopology()

    sc_3c0 = {topologies["3C1"][i]: 0 for i in topologies["3C1"]}
    sc_3c0["mixed"] = 0
    sc_3c0["unstable"] = 0
    sc_4c0 = {topologies["4C1"][i]: 0 for i in topologies["4C1"]}
    sc_4c0["mixed"] = 0
    sc_4c0["unstable"] = 0
    for bb_pair in bb_data:
        bb_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]
        if "3C1" in cl_bbname:
            bbtitle = "3C1"
        elif "4C1" in cl_bbname:
            bbtitle = "4C1"

        all_energies = set(
            bb_dict[i][1] / int(i.rstrip("b")[-1]) for i in bb_dict
        )
        num_mixed = len(
            tuple(i for i in all_energies if i < isomer_energy)
        )
        min_energy = min(tuple(i[1] for i in bb_dict.values()))
        if min_energy > max_energy:
            topo_str = "unstable"
        elif num_mixed > 1:
            topo_str = "mixed"
        else:
            min_e_dict = {
                i: bb_dict[i]
                for i in bb_dict
                if bb_dict[i][1] == min_energy
            }
            keys = list(min_e_dict.keys())
            topo_str = topologies[bbtitle][keys[0][-1]]

        if bbtitle == "3C1":
            sc_3c0[topo_str] += 1
        elif bbtitle == "4C1":
            sc_4c0[topo_str] += 1

    shape_coords = (
        ("3C1", sc_3c0),
        ("4C1", sc_4c0),
    )

    tot_pairs = 0
    for ax, (title, coords) in zip(flat_axs, shape_coords):
        for i in coords:
            dd = coords[i]
            tot_pairs += dd

        ax.bar(
            [convert_topo_to_label(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        for i, key in enumerate(coords):
            val = coords[key]
            ax.text(
                i,
                val + 20,
                val,
                fontsize=16,
                ha="center",
            )

        ax.set_title(f"{title}: {isomer_energy}eV", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)

    if tot_pairs != len(bb_data):
        raise ValueError(
            f"extracted {tot_pairs} pairs, but there should be "
            f"{len(bb_data)}!"
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_4(bb_data, figure_output):
    color_map = shapelabels_to_colormap()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    s_vectors_3c0 = {}
    s_vectors_4c0 = {}
    row_3c0 = set()
    row_4c0 = set()
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]

        min_energy = min(tuple(i[1] for i in b_dict.values()))
        min_e_dict = {
            i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
        }
        keys = list(min_e_dict.keys())
        min_energy_topo = keys[0][-1]

        shape_vector = get_shape_vector(b_dict)
        if "3C1" in cl_bbname:
            s_vectors_3c0[
                (bb_pair[0], bb_pair[1], min_energy_topo)
            ] = shape_vector
            for i in shape_vector:
                row_3c0.add(i)
        elif "4C1" in cl_bbname:
            s_vectors_4c0[
                (bb_pair[0], bb_pair[1], min_energy_topo)
            ] = shape_vector
            for i in shape_vector:
                row_4c0.add(i)

    shape_vector_dicts = (
        ("3C1", s_vectors_3c0, row_3c0),
        ("4C1", s_vectors_4c0, row_4c0),
    )
    for ax, (title, coords, row_names) in zip(
        flat_axs, shape_vector_dicts
    ):
        data_array = pd.DataFrame.from_dict(
            coords,
            orient="index",
        ).reset_index()

        # Separating out the features
        x = data_array.loc[:, row_names].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(x)
        pc_df = pd.DataFrame(
            data=pcs,
            columns=["pc1", "pc2"],
        )

        for t_final in color_map:
            ax.scatter(
                pc_df["pc1"][data_array["level_2"] == t_final],
                pc_df["pc2"][data_array["level_2"] == t_final],
                c=color_map[str(t_final)],
                edgecolor="none",
                s=30,
                alpha=1.0,
                label=t_final,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("principal component 1", fontsize=16)
        ax.set_ylabel("principal component 2", fontsize=16)
        ax.set_title(f"PCA: {title}", fontsize=16)
        ax.legend(ncol=2, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_4.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_5(bb_data, figure_output):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    target_individuals = mapshape_to_topology()

    shape_coordinates = {
        target_individuals[i]: [] for i in target_individuals
    }
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        if "4C1" in bb_pair[1]:
            shapes = target_shapes_by_cltype("4C1")
        elif "3C1" in bb_pair[1]:
            shapes = target_shapes_by_cltype("3C1")

        present_beads_names = get_present_beads(c2_bbname=bb_pair[0])
        for shape in shapes:
            topo_str = target_individuals[shape]
            try:
                shape_value = b_dict[shape][0]
            except KeyError:
                continue
            energy = b_dict[shape][1]
            x = shape_value
            y = (
                get_CGBead_from_string(
                    present_beads_names[-1], arm_2c_beads()
                ).angle_centered
                - 90
            ) * 2
            z = energy
            shape_coordinates[topo_str].append((x, y, z))

    for ax, topo_str in zip(flat_axs, shape_coordinates):
        coords = shape_coordinates[topo_str]
        shape_str = list(target_individuals.keys())[
            list(target_individuals.values()).index(topo_str)
        ]
        ax.set_title(topo_str, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(shape_str, fontsize=16)
        ax.set_ylabel("target bite angle", fontsize=16)

        ax.scatter(
            [i[0] for i in coords],
            [i[1] for i in coords],
            c=[i[2] for i in coords],
            vmin=0,
            vmax=30,
            alpha=0.4,
            edgecolor="none",
            s=30,
            cmap="inferno",
        )

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=0, vmax=30)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("energy (eV)", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_5.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_6(bb_data, figure_output):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    target_individuals = mapshape_to_topology()

    shape_coordinates = {
        target_individuals[i]: [] for i in target_individuals
    }
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        if "4C1" in bb_pair[1]:
            shapes = target_shapes_by_cltype("4C1")
        elif "3C1" in bb_pair[1]:
            shapes = target_shapes_by_cltype("3C1")

        for shape in shapes:
            topo_str = target_individuals[shape]
            try:
                shape_value = b_dict[shape][0]
            except KeyError:
                continue
            energy = b_dict[shape][1]
            pore_radius = b_dict[shape][2]
            x = shape_value
            y = pore_radius
            z = energy
            shape_coordinates[topo_str].append((x, y, z))

    for ax, topo_str in zip(flat_axs, shape_coordinates):
        coords = shape_coordinates[topo_str]
        shape_str = list(target_individuals.keys())[
            list(target_individuals.values()).index(topo_str)
        ]
        ax.set_title(topo_str, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(shape_str, fontsize=16)
        ax.set_ylabel("pore radius [A]", fontsize=16)

        ax.scatter(
            [i[0] for i in coords],
            [i[1] for i in coords],
            c=[i[2] for i in coords],
            vmin=0,
            vmax=30,
            alpha=0.4,
            edgecolor="none",
            s=30,
            cmap="inferno",
        )

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=0, vmax=30)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("energy (eV)", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_6.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_7(bb_data, figure_output):

    topologies = mapshape_to_topology(reverse=True)

    t_map = map_cltype_to_shapetopology()

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    data_dict = {}
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]
        c2_bbname = bb_pair[0]
        present_beads_names = get_present_beads(c2_bbname)

        x = get_CGBead_from_string(
            present_beads_names[0], core_2c_beads()
        ).sigma
        # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
        y = (
            get_CGBead_from_string(
                present_beads_names[-1], arm_2c_beads()
            ).angle_centered
            - 90
        ) * 2

        min_energy = min(tuple(i[1] for i in b_dict.values()))
        min_e_dict = {
            i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
        }
        keys = list(min_e_dict.keys())
        min_energy_topo = keys[0][-1]

        if "3C1" in cl_bbname:
            topology = t_map["3C1"][min_energy_topo]
        elif "4C1" in cl_bbname:
            topology = t_map["4C1"][min_energy_topo]

        target_shape = topologies[topology]
        s = min_e_dict[target_shape][0]

        if topology not in data_dict:
            data_dict[topology] = []
        data_dict[topology].append((x, y, s))

    for ax, t_str in zip(flat_axs, topologies):
        ax.scatter(
            [i[0] for i in data_dict[t_str]],
            [i[1] for i in data_dict[t_str]],
            c=[i[2] for i in data_dict[t_str]],
            edgecolor="none",
            s=30,
            alpha=1.0,
            vmin=0,
            vmax=20,
            cmap="viridis",
        )

        title = t_str + ": " + topologies[t_str].rstrip("b")

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)
        ax.set_title(title, fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=20)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("target shape", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_7.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_8(bb_data, figure_output):

    t_map = map_cltype_to_shapetopology()

    color_map = cltypetopo_to_colormap()
    max_energy = 10
    isomer_energy = 0.05

    data_dict = {}
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]
        c2_bbname = bb_pair[0]

        if "3C1" in cl_bbname:
            title = "3C1"
        elif "4C1" in cl_bbname:
            title = "4C1"

        present_beads_names = get_present_beads(cl_bbname)
        core_cl_bead = present_beads_names[0]
        plot_set = (title, core_cl_bead)

        present_beads_names = get_present_beads(c2_bbname)
        core_bead_s = present_beads_names[0]
        arm_bead_s = present_beads_names[-1]
        x = get_CGBead_from_string(core_bead_s, core_2c_beads()).sigma
        # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
        y = (
            get_CGBead_from_string(
                arm_bead_s, arm_2c_beads()
            ).angle_centered
            - 90
        ) * 2

        all_energies = set(
            b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
        )
        num_mixed = len(
            tuple(i for i in all_energies if i < isomer_energy)
        )
        if num_mixed > 1:
            if num_mixed == 2:
                s = color_map["mixed"]["2"]
            elif num_mixed > 2:
                s = color_map["mixed"][">2"]

        else:
            min_energy = min(tuple(i[1] for i in b_dict.values()))
            min_e_dict = {
                i: b_dict[i]
                for i in b_dict
                if b_dict[i][1] == min_energy
            }
            keys = list(min_e_dict.keys())
            min_energy_topo = keys[0][-1]
            topology = t_map[title][min_energy_topo]

            if min_energy > max_energy:
                s = "white"
            else:
                s = color_map[title][topology]

        if plot_set not in data_dict:
            data_dict[plot_set] = []
        data_dict[plot_set].append((x, y, s))

    for data_pair in data_dict:
        cltype, cl_bead = data_pair
        cbead = get_CGBead_from_string(cl_bead, beads_3c() + beads_4c())
        cl_bead_sigma = cbead.sigma
        cl_bead_angle = cbead.angle_centered

        figtitle_suffix = f"{cltype}_{cl_bead}"

        # fig, axs = plt.subplots(
        #     nrows=1,
        #     ncols=2,
        #     sharex=True,
        #     sharey=True,
        #     figsize=(16, 5),
        # )
        # flat_axs = axs.flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            [i[0] for i in data_dict[data_pair]],
            [i[1] for i in data_dict[data_pair]],
            c=[i[2] for i in data_dict[data_pair]],
            edgecolor="k",
            s=300,
            marker="s",
            alpha=1.0,
        )
        title = (
            f"{cltype} : {cl_bead_sigma} : {cl_bead_angle} : "
            f"{max_energy}eV : {isomer_energy}eV"
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)
        ax.set_title(title, fontsize=16)

        for i in color_map:
            if i not in (cltype, "mixed"):
                continue
            for j in color_map[i]:
                if i == "mixed":
                    string = f"mixed: {j}"
                else:
                    string = j
                ax.scatter(
                    None,
                    None,
                    c=color_map[i][j],
                    edgecolor="k",
                    s=300,
                    marker="s",
                    alpha=1.0,
                    label=string,
                )
        ax.scatter(
            None,
            None,
            c="white",
            edgecolor="k",
            s=300,
            marker="s",
            alpha=1.0,
            label="unstable",
        )

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=3,
            fontsize=16,
        )

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"ps_8_{figtitle_suffix}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def phase_space_9(bb_data, figure_output):

    t_map = map_cltype_to_shapetopology()
    color_map = cltypetopo_to_colormap()
    max_energy = 10
    isomer_energy = 0.05
    min_radius = 0.5

    data_dict = {}
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]
        c2_bbname = bb_pair[0]

        if "3C1" in cl_bbname:
            title = "3C1"
        elif "4C1" in cl_bbname:
            title = "4C1"

        present_beads_names = get_present_beads(cl_bbname)
        core_cl_bead = present_beads_names[0]
        plot_set = (title, core_cl_bead)

        present_beads_names = get_present_beads(c2_bbname)
        core_bead_s = present_beads_names[0]
        arm_bead_s = present_beads_names[-1]
        x = get_CGBead_from_string(core_bead_s, core_2c_beads()).sigma
        # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
        y = (
            get_CGBead_from_string(
                arm_bead_s, arm_2c_beads()
            ).angle_centered
            - 90
        ) * 2

        all_energies = set(
            b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
        )
        num_mixed = len(
            tuple(i for i in all_energies if i < isomer_energy)
        )
        if num_mixed > 1:
            if num_mixed == 2:
                s = color_map["mixed"]["2"]
            elif num_mixed > 2:
                s = color_map["mixed"][">2"]

        else:
            min_energy = min(tuple(i[1] for i in b_dict.values()))
            min_e_dict = {
                i: b_dict[i]
                for i in b_dict
                if b_dict[i][1] == min_energy
            }
            keys = list(min_e_dict.keys())
            min_energy_topo = keys[0][-1]
            min_e_pore_rad = min_e_dict[keys[0]][2]
            topology = t_map[title][min_energy_topo]
            if min_e_pore_rad < min_radius:
                s = "k"
            elif min_energy > max_energy:
                s = "white"
            else:
                s = color_map[title][topology]

        if plot_set not in data_dict:
            data_dict[plot_set] = []
        data_dict[plot_set].append((x, y, s))

    for data_pair in data_dict:
        cltype, cl_bead = data_pair
        cbead = get_CGBead_from_string(cl_bead, beads_3c() + beads_4c())
        cl_bead_sigma = cbead.sigma
        cl_bead_angle = cbead.angle_centered

        figtitle_suffix = f"{cltype}_{cl_bead}"

        # fig, axs = plt.subplots(
        #     nrows=1,
        #     ncols=2,
        #     sharex=True,
        #     sharey=True,
        #     figsize=(16, 5),
        # )
        # flat_axs = axs.flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            [i[0] for i in data_dict[data_pair]],
            [i[1] for i in data_dict[data_pair]],
            c=[i[2] for i in data_dict[data_pair]],
            edgecolor="k",
            s=300,
            marker="s",
            alpha=1.0,
        )
        title = (
            f"{cltype} : {cl_bead_sigma} : {cl_bead_angle} : "
            f"{max_energy}eV : {isomer_energy}eV : {min_radius}A"
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)
        ax.set_title(title, fontsize=16)

        for i in color_map:
            if i not in (cltype, "mixed"):
                continue
            for j in color_map[i]:
                if i == "mixed":
                    string = f"mixed: {j}"
                else:
                    string = j
                ax.scatter(
                    None,
                    None,
                    c=color_map[i][j],
                    edgecolor="k",
                    s=300,
                    marker="s",
                    alpha=1.0,
                    label=string,
                )
        ax.scatter(
            None,
            None,
            c="white",
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
            label=f"pore rad < {min_radius}A",
        )

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=3,
            fontsize=16,
        )

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"ps_9_{figtitle_suffix}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def phase_space_10(bb_data, figure_output):

    t_map = map_cltype_to_shapetopology()
    bead_library = (
        arm_2c_beads() + core_2c_beads() + beads_3c() + beads_4c()
    )

    isomer_energy = 0.05
    max_energy = 10
    for t_cltopo in (3, 4):
        input_dict = {}
        data_dict = {}
        for bb_pair in bb_data:
            b_dict = bb_data[bb_pair]
            cl_bbname = bb_pair[1]
            c2_bbname = bb_pair[0]
            bb_string = f"{cl_bbname}_{c2_bbname}"

            present_c2_beads = get_present_beads(c2_bbname)
            present_cl_beads = get_present_beads(cl_bbname)

            cltopo = int(cl_bbname[0])
            if cltopo != t_cltopo:
                continue
            if cltopo == 4:
                clangle = 90
                cltitle = "4C1"
            elif cltopo == 3:
                clangle = get_CGBead_from_string(
                    present_cl_beads[0],
                    bead_library,
                ).angle_centered
                cltitle = "3C1"

            all_energies = set(
                b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
            )
            num_mixed = len(
                tuple(i for i in all_energies if i < isomer_energy)
            )
            if num_mixed > 1:
                if num_mixed == 2:
                    topology = "mixed (2)"
                elif num_mixed > 2:
                    topology = "mixed (>2)"
                min_e_pore_rad = None

            else:
                min_energy = min(tuple(i[1] for i in b_dict.values()))
                min_e_dict = {
                    i: b_dict[i]
                    for i in b_dict
                    if b_dict[i][1] == min_energy
                }
                keys = list(min_e_dict.keys())
                min_energy_topo = keys[0][-1]
                min_e_pore_rad = min_e_dict[keys[0]][2]
                topology = t_map[cltitle][min_energy_topo]

                if min_energy > max_energy:
                    topology = "unstable"

            row = {
                "cltopo": cltopo,
                "clsigma": get_CGBead_from_string(
                    present_cl_beads[0],
                    bead_library,
                ).sigma,
                "clangle": clangle,
                "c2sigma": get_CGBead_from_string(
                    present_c2_beads[0],
                    bead_library,
                ).sigma,
                "c2angle": (
                    get_CGBead_from_string(
                        present_c2_beads[1],
                        bead_library,
                    ).angle_centered
                    - 90
                )
                * 2,
                "pref_topology": topology,
                "pref_topology_pore": min_e_pore_rad,
            }

            input_dict[bb_string] = row
            data_dict[bb_string] = b_dict

        data_array = pd.DataFrame.from_dict(
            data_dict,
            orient="index",
        ).reset_index()
        print(data_array.head())

        input_array = pd.DataFrame.from_dict(
            input_dict,
            orient="index",
        ).reset_index()
        print(input_array.head())
        target_row_names = (
            "cltopo",
            "clsigma",
            "clangle",
            "c2sigma",
            "c2angle",
        )
        # Separating out the features
        x = input_array.loc[:, target_row_names].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(x)
        pc_df = pd.DataFrame(
            data=pcs,
            columns=["pc1", "pc2"],
        )

        properties = {
            "clangle": "cat",
            "c2sigma": "cat",
            "c2angle": "cat",
            "pref_topology": "cat",
            "pref_topology_pore": "cts",
        }

        for prop in properties:
            prop_type = properties[prop]
            if prop_type == "cat":
                fig, ax = plt.subplots(figsize=(8, 5))

                categories = {}
                for i, prop_set in enumerate(
                    sorted(set(input_array[prop]))
                ):
                    categories[prop_set] = len(
                        input_array[input_array[prop] == prop_set]
                    )

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
                ax.set_xlabel(prop, fontsize=16)
                ax.set_ylabel("count", fontsize=16)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        figure_output, f"dist_10_{t_cltopo}_{prop}.pdf"
                    ),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(
                    pc_df["pc1"],
                    pc_df["pc2"],
                    c="k",
                    edgecolor="none",
                    s=60,
                    alpha=1.0,
                )

                for i, prop_set in enumerate(
                    sorted(set(input_array[prop]))
                ):
                    ax.scatter(
                        pc_df["pc1"][input_array[prop] == prop_set],
                        pc_df["pc2"][input_array[prop] == prop_set],
                        # c=color_map[str(t_final)],
                        edgecolor="none",
                        s=20,
                        alpha=1.0,
                        label=f"{prop}: {prop_set}",
                    )

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel("pc1", fontsize=16)
                ax.set_ylabel("pc2", fontsize=16)
                ax.legend(fontsize=16)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        figure_output, f"ps_10_{t_cltopo}_{prop}.pdf"
                    ),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()
            elif prop_type == "cts":
                fig, ax = plt.subplots(figsize=(8, 5))

                ax.hist(
                    x=list(input_array[prop]),
                    bins=50,
                    density=False,
                    histtype="step",
                    lw=3,
                )

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel(prop, fontsize=16)
                ax.set_ylabel("count", fontsize=16)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        figure_output, f"dist_10_{t_cltopo}_{prop}.pdf"
                    ),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(
                    pc_df["pc1"][pd.notna(input_array[prop])],
                    pc_df["pc2"][pd.notna(input_array[prop])],
                    c="k",
                    edgecolor="none",
                    s=60,
                    alpha=1.0,
                )

                ax.scatter(
                    pc_df["pc1"][pd.notna(input_array[prop])],
                    pc_df["pc2"][pd.notna(input_array[prop])],
                    c=list(
                        input_array[pd.notna(input_array[prop])][prop]
                    ),
                    edgecolor="none",
                    s=20,
                    alpha=1.0,
                    vmin=0,
                    vmax=20,
                    cmap="viridis",
                )

                cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
                cmap = mpl.cm.viridis
                norm = mpl.colors.Normalize(vmin=0, vmax=20)
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax,
                    orientation="vertical",
                )
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(prop, fontsize=16)

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel("pc1", fontsize=16)
                ax.set_ylabel("pc2", fontsize=16)

                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        figure_output, f"ps_10_{t_cltopo}_{prop}.pdf"
                    ),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()


def parity_1(bb_data, figure_output):

    x_target = ("T-4", "4", "FourPlusSix")
    y_targets = (
        ("TBPY-5", "5", "TwoPlusThree"),
        ("T-4b", "b", "FourPlusSix2"),
        ("OC-6", "6", "SixPlusNine"),
        ("CU-8", "8", "EightPlusTwelve"),
    )

    fig, axs = plt.subplots(
        ncols=len(y_targets),
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    ax_datas = {i: [] for i in range(len(y_targets))}
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]

        for i, yt in enumerate(y_targets):
            if x_target[0] in b_dict and yt[0] in b_dict:
                x1 = b_dict[x_target[0]][1]
                y1 = b_dict[yt[0]][1]
                ax_datas[i].append((x1, y1))

    title = "energy [eV]"
    lim = (-0.1, 10)
    for ax, (yid, yinfo) in zip(flat_axs, enumerate(y_targets)):
        ax.scatter(
            [i[0] for i in ax_datas[yid]],
            [i[1] for i in ax_datas[yid]],
            c="gray",
            edgecolor="none",
            s=30,
            alpha=1.0,
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(x_target[2], fontsize=16)
        ax.set_ylabel(yinfo[2], fontsize=16)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "par_1.pdf"),
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

    all_data = {}
    bb_data = {}
    for j_file in calculation_output.glob("*_res.json"):
        with open(j_file, "r") as f:
            res_dict = json.load(f)
        name = str(j_file.name).replace("_res.json", "")
        _, clbb_name, c2bb_name = name.split("_")
        all_data[name] = {}
        all_data[name]["energy"] = res_dict["fin_energy"]
        all_data[name]["pore_diameter"] = (
            res_dict["opt_pore_data"]["pore_mean_rad"] * 2
        )
        all_data[name]["pore_volume"] = res_dict["opt_pore_data"][
            "pore_volume"
        ]
        all_data[name]["shape_vector"] = res_dict["shape_measures"]
        all_data[name]["clbb"] = clbb_name
        all_data[name]["c2bb"] = c2bb_name

        bb_pair = (c2bb_name, clbb_name)
        if bb_pair not in bb_data:
            bb_data[bb_pair] = {}
        for sv in res_dict["shape_measures"]:
            if "FourPlusSix2" in name:
                svb = sv + "b"
            elif "M2L4" in name:
                svb = sv + "b"
            elif "TwoPlusFour" in name:
                svb = sv + "b"
            else:
                svb = sv
            bb_data[bb_pair][svb] = (
                res_dict["shape_measures"][sv],
                res_dict["fin_energy"],
                res_dict["opt_pore_data"]["pore_mean_rad"],
            )

    logging.info(f"there are {len(all_data)} collected data")
    identity_distributions(all_data, figure_output)
    rmsd_distributions(all_data, calculation_output, figure_output)
    phase_space_10(bb_data, figure_output)
    single_value_distributions(all_data, figure_output)
    phase_space_3(bb_data, figure_output)

    raise SystemExit()

    shape_vector_distributions(all_data, figure_output)
    parity_1(bb_data, figure_output)
    phase_space_9(bb_data, figure_output)
    phase_space_1(bb_data, figure_output)
    phase_space_5(bb_data, figure_output)
    phase_space_6(bb_data, figure_output)
    phase_space_7(bb_data, figure_output)
    phase_space_8(bb_data, figure_output)
    phase_space_4(bb_data, figure_output)
    raise SystemExit()
    phase_space_2(bb_data, figure_output)
    raise SystemExit(
        "next I want map of target bite angle to actual bite angle "
    )
    raise SystemExit(
        "next I want PCA maps of shapes for all in each topology "
    )
    raise SystemExit(
        "next I want PCA maps of all shapes for each BB property "
    )
    shape_vector_cluster(
        all_data=all_data,
        c2bb="2C1Mn",
        c3bb=None,
        figure_output=figure_output,
    )
    shape_vector_cluster(
        all_data=all_data,
        c2bb="2C1MnMn",
        c3bb=None,
        figure_output=figure_output,
    )
    shape_vector_cluster(
        all_data=all_data,
        c2bb=None,
        c3bb="3C1Ho",
        figure_output=figure_output,
    )
    shape_vector_cluster(
        all_data=all_data,
        c2bb=None,
        c3bb="3C1C",
        figure_output=figure_output,
    )
    shape_vector_cluster(
        all_data=all_data,
        c2bb=None,
        c3bb="3C1HoMn",
        figure_output=figure_output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
