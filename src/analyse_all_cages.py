#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

Author: Andrew Tarzia

"""

import sys
import os
import json
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from sklearn.cluster import KMeans
import pandas as pd

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from generate_all_cages import core_2c_beads, arm_2c_beads
from env_set import cages


def identity_distributions(all_data, figure_output):

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {
        "2p3": 0,
        "4p6": 0,
        "4p62": 0,
        "6p9": 0,
        "8p12": 0,
        "m2l4": 0,
        "m3l6": 0,
        "m4l8": 0,
        "m6l12": 0,
        # "3C0": 0,
        # "4C0": 0,
        # "2C0": 0,
        # "2C1": 0,
        # "2C2": 0,
        # "2C3": 0,
    }
    num_cages = len(all_data)
    for cage in all_data:
        # cage_dict = all_data[cage]

        topo_name = cage.split("_")[0]
        # c2_bbname = cage_dict["c2bb"]
        # cl_bbname = cage_dict["clbb"]

        if topo_name == "TwoPlusThree":
            categories["2p3"] += 1
        if topo_name == "FourPlusSix":
            categories["4p6"] += 1
        if topo_name == "FourPlusSix2":
            categories["4p62"] += 1
        if topo_name == "SixPlusNine":
            categories["6p9"] += 1
        if topo_name == "EightPlusTwelve":
            categories["8p12"] += 1

        if topo_name == "M2L4":
            categories["m2l4"] += 1
        if topo_name == "M3L6":
            categories["m3l6"] += 1
        if topo_name == "M4L8":
            categories["m4l8"] += 1
        if topo_name == "M6L12":
            categories["m6l12"] += 1

    #     if "3C0" in cl_bbname:
    #         categories["3C0"] += 1
    #     if "4C0" in cl_bbname:
    #         categories["4C0"] += 1

    # if categories["3C0"] + categories["4C0"] != num_cages:
    #     raise ValueError(
    #         f'{categories["3C0"]} + {categories["4C0"]} != {num_cages}'
    #     )

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


def single_value_distributions(all_data, figure_output):
    to_plot = {
        "energy": {"xtitle": "energy"},
        "pore_diameter": {"xtitle": "pore diameter"},
        "pore_volume": {"xtitle": "pore volume"},
        "energy_barm": {"xtitle": "energy"},
        "pore_diameter_barm": {"xtitle": "pore diameter"},
        "pore_volume_barm": {"xtitle": "pore volume"},
    }

    color_map = {
        "TwoPlusThree": "#06AED5",
        "FourPlusSix": "#086788",
        "FourPlusSix2": "#DD1C1A",
        "SixPlusNine": "#320E3B",
        "EightPlusTwelve": "#CE7B91",
        "M2L4": "#6969B3",
        "M3L6": "#B279A7",
        "M4L8": "#C3423F",
        "M6L12": "#9BC53D",
    }
    color_map_barm = {
        "3C0": "#06AED5",
        "4C0": "#086788",
    }

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
            flat_axs[i].set_ylabel("count", fontsize=16)
            flat_axs[i].set_title(topo, fontsize=16)
            # ax.set_yscale("log")
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

    color_map = {
        4: "#06AED5",
        5: "#086788",
        6: "#DD1C1A",
        8: "#320E3B",
        3: "#6969B3",
    }
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
    target_individuals = ("CU-8", "OC-6", "TBPY-5", "T-4", "TPR-6")
    color_map = {
        "CU-8": "#06AED5",
        "OC-6": "#086788",
        "TBPY-5": "#DD1C1A",
        "T-4": "#320E3B",
        "TPR-6": "#CE7B91",
    }
    min_of_each_shape_plot = {i: None for i in target_individuals}
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
                    if sv in target_individuals:
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


def phase_space_1(bb_data, figure_output):

    topologies = (
        "TwoPlusThree",
        "FourPlusSix",
        "FourPlusSix2",
        "SixPlusNine",
        "EightPlusTwelve",
        "M2L4",
        "M3L6",
        "M4L8",
        "M6L12",
    )

    t_map = {
        "3C0": {
            "4": "FourPlusSix",
            "5": "TwoPlusThree",
            "6": "SixPlusNine",
            "8": "EightPlusTwelve",
            "b": "FourPlusSix2",
        },
        "4C0": {
            "3": "M3L6",
            "4": "M4L8",
            "6": "M6L12",
            "b": "M2L4",
        },
    }

    # color_map = {
    #     "3C0": {
    #         "TwoPlusThree": "#06AED5",
    #         "FourPlusSix": "#086788",
    #         "FourPlusSix2": "#DD1C1A",
    #         "SixPlusNine": "#320E3B",
    #         "EightPlusTwelve": "#CE7B91",
    #     },
    #     "4C0": {
    #         "M2L4": "#6969B3",
    #         "M3L6": "#B279A7",
    #         "M4L8": "#C3423F",
    #         "M6L12": "#9BC53D",
    #     },
    # }

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

        if "3C0" in cl_bbname:
            if t_map["3C0"][min_energy_topo] not in data_dict:
                data_dict[t_map["3C0"][min_energy_topo]] = []
            data_dict[t_map["3C0"][min_energy_topo]].append((x, y))
        elif "4C0" in cl_bbname:
            if t_map["4C0"][min_energy_topo] not in data_dict:
                data_dict[t_map["4C0"][min_energy_topo]] = []
            data_dict[t_map["4C0"][min_energy_topo]].append((x, y))

    for ax, t_str in zip(flat_axs, topologies):
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

            if "3C0" in cl_bbname:
                sc_3c0_2c1.append((x, y))
            elif "4C0" in cl_bbname:
                sc_4c0_2c1.append((x, y))

        shape_coords = (
            # (f"3C0-2C0 ({len(sc_3c0_2c0)})", sc_3c0_2c0),
            (f"3C0-2C1 ({len(sc_3c0_2c1)})", sc_3c0_2c1),
            (f"4C0-2C1 ({len(sc_4c0_2c1)})", sc_4c0_2c1),
            # (f"3C0-2C3 ({len(sc_3c0_2c3)})", sc_3c0_2c3),
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

    topologies = {
        "3C0": {
            "5": "2+3",
            "4": "4+6",
            "b": "4+6-2",
            "6": "6+9",
            "8": "8+12",
        },
        "4C0": {
            "b": "M2L4",
            "4": "M4L8",
            "3": "M3L6",
            "6": "M6L12",
        },
    }

    sc_3c0 = {topologies["3C0"][i]: 0 for i in topologies["3C0"]}
    sc_4c0 = {topologies["4C0"][i]: 0 for i in topologies["4C0"]}
    for bb_pair in bb_data:
        bb_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]

        min_energy = min(tuple(i[1] for i in bb_dict.values()))
        min_e_dict = {
            i: bb_dict[i]
            for i in bb_dict
            if bb_dict[i][1] == min_energy
        }
        keys = list(min_e_dict.keys())

        if "3C0" in cl_bbname:
            topo_str = topologies["3C0"][keys[0][-1]]
            sc_3c0[topo_str] += 1
        elif "4C0" in cl_bbname:
            topo_str = topologies["4C0"][keys[0][-1]]
            sc_4c0[topo_str] += 1

    shape_coords = (
        ("3C0", sc_3c0),
        ("4C0", sc_4c0),
    )

    tot_pairs = 0
    for ax, (title, coords) in zip(flat_axs, shape_coords):
        for i in coords:
            dd = coords[i]
            tot_pairs += dd

        ax.bar(
            coords.keys(),
            coords.values(),
            # color="#06AED5",
            color="#086788",
            # color="#DD1C1A",
            # color="#320E3B",
            edgecolor="k",
        )

        ax.set_title(title, fontsize=16)
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


def get_shape_vector(shape_dictionary):
    target_shapes = (
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
    shape_vector = {}
    for i in target_shapes:
        if i in shape_dictionary:
            shape_vector[i] = shape_dictionary[i][0]
        if i + "b" in shape_dictionary:
            shape_vector[i + "b"] = shape_dictionary[i + "b"][0]

    return shape_vector


def phase_space_4(bb_data, figure_output):
    color_map = {
        "3": "#F9A03F",
        "4": "#0B2027",
        "5": "#86626E",
        "6": "#CA1551",
        "8": "#345995",
        "b": "#7A8B99",
    }

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
        if "3C0" in cl_bbname:
            s_vectors_3c0[
                (bb_pair[0], bb_pair[1], min_energy_topo)
            ] = shape_vector
            for i in shape_vector:
                row_3c0.add(i)
        elif "4C0" in cl_bbname:
            s_vectors_4c0[
                (bb_pair[0], bb_pair[1], min_energy_topo)
            ] = shape_vector
            for i in shape_vector:
                row_4c0.add(i)

    shape_vector_dicts = (
        ("3C0", s_vectors_3c0, row_3c0),
        ("4C0", s_vectors_4c0, row_4c0),
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


def phase_space_5(bb_data, figure_output):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    target_individuals = {
        "TBPY-5": "TwoPlusThree",
        "T-4": "FourPlusSix",
        "T-4b": "FourPlusSix2",
        "TPR-6": "SixPlusNine",
        "CU-8": "EightPlusTwelve",
        "OC-6b": "M2L4",
        "TP-3": "M3L6",
        "SP-4": "M4L8",
        "OC-6": "M6L12",
    }

    # target_size = {"3C0": "Ir", "4C0": "Ne"}

    shape_coordinates = {
        target_individuals[i]: [] for i in target_individuals
    }
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        if "4C0" in bb_pair[1]:
            # if target_size["4C0"] not in bb_pair[1]:
            #     continue
            shapes = ("OC-6b", "TP-3", "SP-4", "OC-6")
        elif "3C0" in bb_pair[1]:
            # if target_size["3C0"] not in bb_pair[1]:
            #     continue
            shapes = ("TBPY-5", "T-4", "T-4b", "TPR-6", "CU-8")

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

    target_individuals = {
        "TBPY-5": "TwoPlusThree",
        "T-4": "FourPlusSix",
        "T-4b": "FourPlusSix2",
        "TPR-6": "SixPlusNine",
        "CU-8": "EightPlusTwelve",
        "OC-6b": "M2L4",
        "TP-3": "M3L6",
        "SP-4": "M4L8",
        "OC-6": "M6L12",
    }

    shape_coordinates = {
        target_individuals[i]: [] for i in target_individuals
    }
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        if "4C0" in bb_pair[1]:
            shapes = ("OC-6b", "TP-3", "SP-4", "OC-6")
        elif "3C0" in bb_pair[1]:
            shapes = ("TBPY-5", "T-4", "T-4b", "TPR-6", "CU-8")

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

    topologies = {
        "TwoPlusThree": "TBPY-5",
        "FourPlusSix": "T-4",
        "FourPlusSix2": "T-4b",
        "SixPlusNine": "TPR-6",
        "EightPlusTwelve": "CU-8",
        "M2L4": "OC-6b",
        "M3L6": "TP-3",
        "M4L8": "SP-4",
        "M6L12": "OC-6",
    }

    t_map = {
        "3C0": {
            "4": "FourPlusSix",
            "5": "TwoPlusThree",
            "6": "SixPlusNine",
            "8": "EightPlusTwelve",
            "b": "FourPlusSix2",
        },
        "4C0": {
            "3": "M3L6",
            "4": "M4L8",
            "6": "M6L12",
            "b": "M2L4",
        },
    }

    # color_map = {
    #     "3C0": {
    #         "TwoPlusThree": "#06AED5",
    #         "FourPlusSix": "#086788",
    #         "FourPlusSix2": "#DD1C1A",
    #         "SixPlusNine": "#320E3B",
    #         "EightPlusTwelve": "#CE7B91",
    #     },
    #     "4C0": {
    #         "M2L4": "#6969B3",
    #         "M3L6": "#B279A7",
    #         "M4L8": "#C3423F",
    #         "M6L12": "#9BC53D",
    #     },
    # }

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

        if "3C0" in cl_bbname:
            topology = t_map["3C0"][min_energy_topo]
        elif "4C0" in cl_bbname:
            topology = t_map["4C0"][min_energy_topo]

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

    t_map = {
        "3C0": {
            "4": "FourPlusSix",
            "5": "TwoPlusThree",
            "6": "SixPlusNine",
            "8": "EightPlusTwelve",
            "b": "FourPlusSix2",
        },
        "4C0": {
            "3": "M3L6",
            "4": "M4L8",
            "6": "M6L12",
            "b": "M2L4",
        },
    }

    color_map = {
        "3C0": {
            "TwoPlusThree": "#448947",
            "FourPlusSix": "#73A775",
            "FourPlusSix2": "#8AB58C",
            "SixPlusNine": "#B9D3BA",
            "EightPlusTwelve": "#E8F0E8",
        },
        "4C0": {
            "M2L4": "#0A5F92",
            "M3L6": "#4787AD",
            "M4L8": "#85AFC9",
            "M6L12": "#C2D7E4",
        },
        "mixed": {
            "2": "#894487",
            "3": "#B58AB4",
            "4": "#D3B9D2",
            ">4": "white",
        },
    }

    target_3or4_beads = ("Ho", "Pt")
    target_arm_sigma = 2.0
    max_energy = 10
    isomer_energy = 0.05

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    data_dict = {}
    for bb_pair in bb_data:
        b_dict = bb_data[bb_pair]
        cl_bbname = bb_pair[1]
        c2_bbname = bb_pair[0]

        if "3C0" in cl_bbname and target_3or4_beads[0] in cl_bbname:
            title = "3C0"

        elif "4C0" in cl_bbname and target_3or4_beads[1] in cl_bbname:
            title = "4C0"

        else:
            continue

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

        if (
            get_CGBead_from_string(arm_bead_s, arm_2c_beads()).sigma
            != target_arm_sigma
        ):
            continue

        all_energies = set(
            b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
        )
        num_mixed = len(
            tuple(i for i in all_energies if i < isomer_energy)
        )
        if num_mixed > 1:
            if num_mixed == 2:
                s = color_map["mixed"]["2"]
            elif num_mixed == 3:
                s = color_map["mixed"]["3"]
            elif num_mixed == 4:
                s = color_map["mixed"]["4"]
            else:
                s = color_map["mixed"][">4"]

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
                s = "gray"
            else:
                s = color_map[title][topology]

        if title not in data_dict:
            data_dict[title] = []
        data_dict[title].append((x, y, s))

    for ax, title in zip(flat_axs, t_map):
        ax.scatter(
            [i[0] for i in data_dict[title]],
            [i[1] for i in data_dict[title]],
            c=[i[2] for i in data_dict[title]],
            edgecolor="k",
            s=300,
            marker="s",
            alpha=1.0,
        )
        target_bead = ",".join(target_3or4_beads)
        title = (
            f"{title} : {target_bead} : {target_arm_sigma} : "
            f"{max_energy}eV : {isomer_energy}eV"
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)
        ax.set_title(title, fontsize=16)

    for i in color_map:
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
        c="gray",
        edgecolor="k",
        s=300,
        marker="s",
        alpha=1.0,
        label="unstable",
    )

    fig.legend(
        bbox_to_anchor=(0, 1.02, 2, 0.2),
        loc="lower left",
        ncol=5,
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_8.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_9(bb_data, figure_output):

    topologies = (
        ("4C0", "M2L4"),
        ("4C0", "M3L6"),
        ("4C0", "M4L8"),
        ("4C0", "M6L12"),
        ("3C0", "TwoPlusThree"),
        ("3C0", "FourPlusSix"),
        ("3C0", "FourPlusSix2"),
        ("3C0", "SixPlusNine"),
        ("3C0", "EightPlusTwelve"),
    )
    t_map = {
        "3C0": {
            "4": "FourPlusSix",
            "5": "TwoPlusThree",
            "6": "SixPlusNine",
            "8": "EightPlusTwelve",
            "b": "FourPlusSix2",
        },
        "4C0": {
            "3": "M3L6",
            "4": "M4L8",
            "6": "M6L12",
            "b": "M2L4",
        },
    }

    color_map = {
        "3C0": {
            "TwoPlusThree": "#448947",
            "FourPlusSix": "#73A775",
            "FourPlusSix2": "#8AB58C",
            "SixPlusNine": "#B9D3BA",
            "EightPlusTwelve": "#E8F0E8",
        },
        "4C0": {
            "M2L4": "#0A5F92",
            "M3L6": "#4787AD",
            "M4L8": "#85AFC9",
            "M6L12": "#C2D7E4",
        },
    }

    t_3or4_beads = ("Ho", "Pt")
    t_arm_sigma = 2.0
    max_energy = 10

    for title, tstring in topologies:

        fig, ax = plt.subplots(figsize=(8, 5))
        data_dict = {}
        for bb_pair in bb_data:
            b_dict = bb_data[bb_pair]
            cl_bbname = bb_pair[1]
            c2_bbname = bb_pair[0]
            if title not in data_dict:
                data_dict[title] = []

            if t_3or4_beads[0] in cl_bbname and title == "3C0":
                pass
            elif t_3or4_beads[1] in cl_bbname and title == "4C0":
                pass
            else:
                continue

            present_beads_names = get_present_beads(c2_bbname)
            core_bead_s = present_beads_names[0]
            arm_bead_s = present_beads_names[-1]
            x = get_CGBead_from_string(
                core_bead_s, core_2c_beads()
            ).sigma
            # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
            y = (
                get_CGBead_from_string(
                    arm_bead_s, arm_2c_beads()
                ).angle_centered
                - 90
            ) * 2

            if (
                get_CGBead_from_string(arm_bead_s, arm_2c_beads()).sigma
                != t_arm_sigma
            ):
                continue

            for i in b_dict:
                topology_ey = b_dict[i][1] / int(i.rstrip("b")[-1])
                if i[-1] in t_map[title]:
                    topology = t_map[title][i[-1]]
                    break

            if topology != tstring:
                continue

            if topology_ey > max_energy:
                s = "gray"
            else:
                s = color_map[title][topology]

            data_dict[title].append((x, y, s))

        ax.scatter(
            [i[0] for i in data_dict[title]],
            [i[1] for i in data_dict[title]],
            c=[i[2] for i in data_dict[title]],
            edgecolor="k",
            s=300,
            marker="s",
            alpha=1.0,
        )
        target_bead = ",".join(t_3or4_beads)
        title = (
            f"{title} : {target_bead} : {t_arm_sigma} : "
            f"{max_energy}eV"
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)
        ax.set_title(title, fontsize=16)

        for i in color_map:
            for j in color_map[i]:
                if j != tstring:
                    continue
                ax.scatter(
                    None,
                    None,
                    c=color_map[i][j],
                    edgecolor="k",
                    s=300,
                    marker="s",
                    alpha=1.0,
                    label=j,
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

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=5,
            fontsize=16,
        )

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"ps_9_{tstring}.pdf"),
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
            else:
                svb = sv
            bb_data[bb_pair][svb] = (
                res_dict["shape_measures"][sv],
                res_dict["fin_energy"],
                res_dict["opt_pore_data"]["pore_mean_rad"],
            )

    logging.info(f"there are {len(all_data)} collected data")
    phase_space_8(bb_data, figure_output)
    phase_space_9(bb_data, figure_output)

    parity_1(bb_data, figure_output)
    identity_distributions(all_data, figure_output)
    single_value_distributions(all_data, figure_output)
    shape_vector_distributions(all_data, figure_output)
    phase_space_1(bb_data, figure_output)
    phase_space_3(bb_data, figure_output)
    phase_space_4(bb_data, figure_output)
    phase_space_5(bb_data, figure_output)
    phase_space_6(bb_data, figure_output)
    phase_space_7(bb_data, figure_output)
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
        c2bb="2C0Mn",
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
        c3bb="3C0Ho",
        figure_output=figure_output,
    )
    shape_vector_cluster(
        all_data=all_data,
        c2bb=None,
        c3bb="3C0C",
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
