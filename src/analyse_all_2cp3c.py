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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from sklearn.cluster import KMeans
import pandas as pd

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from generate_all_2cp3c import core_2c_beads, arm_2c_beads
from env_set import cages


def identity_distributions(all_data, figure_output):

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {
        "2p3": 0,
        "4p6": 0,
        "4p62": 0,
        "6p9": 0,
        "8p12": 0,
        # "3C0": 0,
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
        # c3_bbname = cage_dict["c3bb"]

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

        # if "3C0" in c3_bbname:
        #     categories["3C0"] += 1
        # if "2C0" in c2_bbname:
        #     categories["2C0"] += 1
        # if "2C1" in c2_bbname:
        #     categories["2C1"] += 1
        # if "2C2" in c2_bbname:
        #     categories["2C2"] += 1
        # if "2C3" in c2_bbname:
        #     categories["2C3"] += 1

    # if categories["3C0"] != num_cages:
    #     raise ValueError(f'{categories["3C0"]} != {num_cages}')

    # if (
    #     sum(
    #         (
    #             categories["2C0"],
    #             categories["2C1"],
    #             categories["2C2"],
    #             categories["2C3"],
    #         )
    #     )
    #     != num_cages
    # ):
    #     raise ValueError(
    #         f'{categories["2C0"]} + {categories["2C1"]} + '
    #         f'{categories["2C2"]} + {categories["2C3"]} != {num_cages}'
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
    ax.set_yscale("log")

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
    }
    color_map_barm = {
        "2C0": "#06AED5",
        "2C1": "#086788",
        "2C2": "#DD1C1A",
        "2C3": "#320E3B",
    }

    for tp in to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        # zoom_values = [i for i in values if i < 1]
        xtitle = to_plot[tp]["xtitle"]
        if "barm" in tp:
            cmapp = color_map_barm
        else:
            cmapp = color_map

        for topo in cmapp:
            if "barm" in tp:
                values = [
                    all_data[i][tp.replace("_barm", "")]
                    for i in all_data
                    if topo in i.split("_")[2]
                ]
            else:
                values = [
                    all_data[i][tp]
                    for i in all_data
                    if i.split("_")[0] == topo
                ]
            ax.hist(
                x=values,
                bins=50,
                density=False,
                histtype="step",
                color=cmapp[topo],
                lw=3,
                label=topo,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(xtitle, fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        ax.set_yscale("log")
        ax.legend(fontsize=16)

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
    num_cols = 3
    num_rows = 9

    color_map = {
        4: "#06AED5",
        5: "#086788",
        6: "#DD1C1A",
        8: "#320E3B",
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
    pcindexed_df = pd.concat(
        [pc_df, data_array[["index"]]],
        axis=1,
    )
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

    target_individuals = (
        "TBPY-5",
        "T-4",
        "OC-6",
        "TPR-6",
        "CU-8",
    )

    fig, axs = plt.subplots(
        nrows=3,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    shape_coordinates = {i: [] for i in target_individuals}
    shape_coordinates["other"] = []
    for bb_pair in bb_data:
        c2_bbname = bb_pair[0]

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
            raise ValueError(
                f"{present_beads_names} length != {c2_bbname}"
            )

        x = get_CGBead_from_string(
            present_beads_names[0], core_2c_beads()
        ).sigma
        # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
        y = get_CGBead_from_string(
            present_beads_names[-1], arm_2c_beads()
        ).angle_centered

        bb_dict = bb_data[bb_pair]
        min_energy = min(tuple(i[1] for i in bb_dict.values()))
        min_e_dict = {
            i: bb_dict[i]
            for i in bb_dict
            if bb_dict[i][1] == min_energy
        }
        min_shape = min(min_e_dict, key=bb_dict.get)

        if min_shape.replace("b", "") not in target_individuals:
            shape_coordinates["other"].append((x, y))
        else:
            shape_coordinates[min_shape.replace("b", "")].append((x, y))

    for ax, shape in zip(flat_axs, shape_coordinates):
        coords = shape_coordinates[shape]

        ax.set_title(shape, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        # ax.set_xlabel("target 2c core angle", fontsize=16)
        ax.set_xlabel("target 2c core size", fontsize=16)
        # ax.set_ylabel("tritopic core size", fontsize=16)
        ax.set_ylabel("target 2c binder angle", fontsize=16)

        if len(coords) == 0:
            continue
        # hb = ax.hexbin(
        #     [i[0] for i in coords],
        #     [i[1] for i in coords],
        #     gridsize=20,
        #     cmap="inferno",
        #     bins="log",
        # )
        # fig.colorbar(hb, ax=ax, label="log10(N)")
        ax.scatter(
            [i[0] for i in coords],
            [i[1] for i in coords],
            c="r",
            alpha=0.2,
            s=60,
        )
        # ax.scatter(
        #     [i[0] for i in coords2],
        #     [i[1] for i in coords2],
        #     c="b",
        #     alpha=0.2,
        #     s=60,
        # )

    # fig.legend(ncol=4, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_2(bb_data, figure_output):
    for i in ("shape", "energy", "se"):
        ncols = 1

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
        for bb_pair in bb_data:
            bb_dict = bb_data[bb_pair]
            # c2_bbname = bb_pair[0]
            # c3_bbname = bb_pair[1]

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

            sc_3c0_2c1.append((x, y))

        shape_coords = (
            # (f"3C0-2C0 ({len(sc_3c0_2c0)})", sc_3c0_2c0),
            (f"3C0-2C1 ({len(sc_3c0_2c1)})", sc_3c0_2c1),
            # (f"3C0-2C2 ({len(sc_3c0_2c2)})", sc_3c0_2c2),
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
        ncols=4,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()
    raise SystemExit("here")

    topologies = {
        "5": "2p3",
        "4": "4p6",
        "b": "4p62",
        "6": "6p9",
        "8": "8p12",
    }

    sc_3c0_2c0 = {}
    sc_3c0_2c1 = {}
    sc_3c0_2c2 = {}
    sc_3c0_2c3 = {}
    for bb_pair in bb_data:
        bb_dict = bb_data[bb_pair]
        c2_bbname = bb_pair[0]

        min_energy = min(tuple(i[1] for i in bb_dict.values()))
        min_e_dict = {
            i: bb_dict[i]
            for i in bb_dict
            if bb_dict[i][1] == min_energy
        }
        keys = list(min_e_dict.keys())
        topo_str = topologies[keys[0][-1]]

        if "2C0" in c2_bbname:
            if topo_str not in sc_3c0_2c0:
                sc_3c0_2c0[topo_str] = 0
            sc_3c0_2c0[topo_str] += 1
        elif "2C1" in c2_bbname:
            if topo_str not in sc_3c0_2c1:
                sc_3c0_2c1[topo_str] = 0
            sc_3c0_2c1[topo_str] += 1
        elif "2C2" in c2_bbname:
            if topo_str not in sc_3c0_2c2:
                sc_3c0_2c2[topo_str] = 0
            sc_3c0_2c2[topo_str] += 1
        elif "2C3" in c2_bbname:
            if topo_str not in sc_3c0_2c3:
                sc_3c0_2c3[topo_str] = 0
            sc_3c0_2c3[topo_str] += 1

    shape_coords = (
        ("3C0-2C0", sc_3c0_2c0),
        ("3C0-2C1", sc_3c0_2c1),
        ("3C0-2C2", sc_3c0_2c2),
        ("3C0-2C3", sc_3c0_2c3),
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
        _, c3bb_name, c2bb_name = name.split("_")
        all_data[name] = {}
        all_data[name]["energy"] = res_dict["fin_energy"]
        all_data[name]["pore_diameter"] = (
            res_dict["opt_pore_data"]["pore_mean_rad"] * 2
        )
        all_data[name]["pore_volume"] = res_dict["opt_pore_data"][
            "pore_volume"
        ]
        all_data[name]["shape_vector"] = res_dict["shape_measures"]
        all_data[name]["c3bb"] = c3bb_name
        all_data[name]["c2bb"] = c2bb_name

        bb_pair = (c2bb_name, c3bb_name)
        if bb_pair not in bb_data:
            bb_data[bb_pair] = {}
        for sv in res_dict["shape_measures"]:
            if "FourPlusSix2" in name:
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
    single_value_distributions(all_data, figure_output)
    shape_vector_distributions(all_data, figure_output)
    phase_space_1(bb_data, figure_output)
    phase_space_2(bb_data, figure_output)
    phase_space_3(bb_data, figure_output)
    raise SystemExit()
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
