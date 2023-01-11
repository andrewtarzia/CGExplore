#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Utilities for analysis and plotting.

Author: Andrew Tarzia

"""

import sys
import os
import json
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from generate_all_cages import (
    core_2c_beads,
    arm_2c_beads,
    beads_3c,
    beads_4c,
)
from shape import known_shape_vectors


def isomer_energy():
    return 0.05


def max_energy():
    return 1.0


def min_radius():
    return 1.0


def min_b2b_distance():
    return 0.5


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
            "8+16",
            "12+24",
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
            "EightPlusSixteen",
            "M12L24",
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
        "EightPlusSixteen": "8+16",
        "M12L24": "12+24",
        "mixed": "mixed",
        "all": "all",
        "unstable": "unstable",
        "not": "not",
        "3C1": "3-c",
        "4C1": "4-c",
    }[topo_str]


def convert_torsion_to_label(topo_str, num=True):
    if num:
        return {
            "ton": "5 eV",
            "toff": "none",
        }[topo_str]
    else:
        return {
            "ton": "restricted",
            "toff": "not restricted",
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
        "EightPlusSixteen": "k",
        "M12L24": "k",
    }


def torsion_to_colormap():
    return {
        "ton": "k",
        "toff": "r",
    }


def stoich_map(tstr):
    return {
        "TwoPlusThree": 6,
        "FourPlusSix": 12,
        "FourPlusSix2": 12,
        "SixPlusNine": 18,
        "EightPlusTwelve": 24,
        "TwoPlusFour": 8,
        "ThreePlusSix": 12,
        "FourPlusEight": 16,
        "SixPlusTwelve": 24,
        "EightPlusSixteen": 32,
        "M12L24": 48,
    }[tstr]


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
            "TwoPlusFour": "#1f77b4",
            "ThreePlusSix": "#ff7f0e",
            "FourPlusEight": "#2ca02c",
            "SixPlusTwelve": "#d62728",
            "EightPlusSixteen": "k",
            "M12L24": "#17becf",
        },
        "mixed": {
            # "2": "#7b4173",
            # ">2": "#de9ed6",
            "mixed": "white",
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
            "12": "M12L24",
        },
    }


def mapshape_to_topology(mode, reverse=False):
    if mode == "n":
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
    elif mode == "l":
        if reverse:
            return {
                "TwoPlusThree": "TP-3",
                "FourPlusSix": "OC-6",
                "FourPlusSix2": "OC-6b",
                "TwoPlusFour": "SP-4",
            }
        else:
            return {
                "TP-3": "TwoPlusThree",
                "OC-6": "FourPlusSix",
                "OC-6b": "FourPlusSix2",
                "SP-4": "TwoPlusFour",
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
    return bead_library[string]


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
    break_strs = [i for i in wtopo if not i.isnumeric()]
    if len(break_strs) != 2:
        raise ValueError(f"Too many beads found in {c2_bbname}")

    broken_string = wtopo.split(break_strs[0])[1]
    bead1name, bead2name = broken_string.split(break_strs[1])
    bead1name = break_strs[0] + bead1name
    bead2name = break_strs[1] + bead2name
    present_beads_names = (bead1name, bead2name)
    return present_beads_names


def get_sv_dist(row, mode):
    maps = mapshape_to_topology(mode=mode, reverse=True)
    try:
        tshape = maps[str(row["topology"])]
    except KeyError:
        return None

    if tshape[-1] == "b":
        tshape = tshape[:-1]
    known_sv = known_shape_vectors()[tshape]
    current_sv = {i: float(row[f"{mode}_{i}"]) for i in known_sv}
    a = np.array([known_sv[i] for i in known_sv])
    b = np.array([current_sv[i] for i in known_sv])
    cosine_similarity = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    return cosine_similarity


def is_persistent(row):

    n_vector_distance = get_sv_dist(row, mode="n")
    l_vector_distance = get_sv_dist(row, mode="l")

    pore = float(row["pore"])
    if pore > 1:
        if n_vector_distance is None:
            return True
        elif n_vector_distance < 1:
            if l_vector_distance is None:
                return True
            elif l_vector_distance < 1:
                return True

    return False


def data_to_array(json_files, output_dir):
    output_csv = output_dir / "all_array.csv"
    geom_json = output_dir / "all_geom.json"

    if os.path.exists(output_csv):
        input_array = pd.read_csv(output_csv)
        input_array = input_array.loc[
            :, ~input_array.columns.str.contains("^Unnamed")
        ]
        return input_array

    input_dict = {}
    geom_data = {}
    for j_file in sorted(json_files):
        with open(j_file, "r") as f:
            res_dict = json.load(f)
        name = str(j_file.name).replace("_res.json", "")
        t_str, clbb_name, c2bb_name, torsions = name.split("_")
        cage_name = f"{t_str}_{clbb_name}_{c2bb_name}"
        optimised = res_dict["optimised"]
        if optimised:
            energy = res_dict["fin_energy"]
            energy_per_bond = res_dict["fin_energy"] / stoich_map(t_str)
            gnorm = res_dict["fin_gnorm"]
            min_distance = res_dict["opt_pore_data"]["min_distance"]
            min_b2b = res_dict["min_b2b_distance"]
            node_shape_vector = res_dict["node_shape_measures"]
            lig_shape_vector = res_dict["lig_shape_measures"]
            bond_data = res_dict["bond_data"]
            angle_data = res_dict["angle_data"]
            dihedral_data = res_dict["dihedral_data"]
            geom_data[name] = {
                "bonds": bond_data,
                "angles": angle_data,
                "dihedrals": dihedral_data,
            }

        else:
            energy = None
            min_distance = None
            node_shape_vector = None
            lig_shape_vector = None

        present_c2_beads = get_present_beads(c2bb_name)
        present_cl_beads = get_present_beads(clbb_name)

        cl_bead_libs = beads_3c().copy()
        cl_bead_libs.update(beads_4c())
        cltopo = int(clbb_name[0])
        clangle = get_CGBead_from_string(
            present_cl_beads[0],
            cl_bead_libs,
        ).angle_centered
        cltitle = "4C1" if cltopo == 4 else "3C1"

        row = {
            "cage_name": cage_name,
            "clbb_name": clbb_name,
            "c2bb_name": c2bb_name,
            "clbb_b1": present_cl_beads[0],
            "clbb_b2": present_cl_beads[1],
            "c2bb_b1": present_c2_beads[0],
            "c2bb_b2": present_c2_beads[1],
            "bbpair": clbb_name + c2bb_name,
            "cltopo": cltopo,
            "cltitle": cltitle,
            "clsigma": get_CGBead_from_string(
                present_cl_beads[0],
                cl_bead_libs,
            ).sigma,
            "clangle": clangle,
            "c2sigma": get_CGBead_from_string(
                present_c2_beads[0],
                core_2c_beads(),
            ).sigma,
            "c2angle": get_CGBead_from_string(
                present_c2_beads[1],
                arm_2c_beads(),
            ).angle_centered,
            "target_bite_angle": (
                get_CGBead_from_string(
                    present_c2_beads[1],
                    arm_2c_beads(),
                ).angle_centered
                - 90
            )
            * 2,
            "energy": energy,
            "energy_per_bond": energy_per_bond,
            "gnorm": gnorm,
            "pore": min_distance,
            "min_b2b": min_b2b,
            "topology": t_str,
            "torsions": torsions,
            "optimised": optimised,
        }
        if node_shape_vector is not None:
            for sv in node_shape_vector:
                row[f"n_{sv}"] = node_shape_vector[sv]
        if lig_shape_vector is not None:
            for sv in lig_shape_vector:
                row[f"l_{sv}"] = lig_shape_vector[sv]
        input_dict[name] = row

    input_array = pd.DataFrame.from_dict(
        input_dict,
        orient="index",
    ).reset_index()

    input_array["sv_n_dist"] = input_array.apply(
        lambda row: get_sv_dist(row, mode="n"),
        axis=1,
    )
    input_array["sv_l_dist"] = input_array.apply(
        lambda row: get_sv_dist(row, mode="l"),
        axis=1,
    )
    input_array["persistent"] = input_array.apply(
        lambda row: is_persistent(row),
        axis=1,
    )

    input_array.to_csv(output_csv, index=False)

    with open(geom_json, "w") as f:
        json.dump(geom_data, f, indent=4)

    return input_array


def shape_vector_cluster(all_data, c2bb, c3bb, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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


def write_out_mapping(all_data):
    bite_angle_map = {}
    clangle_map = {}
    clsigma_map = {}
    c2sigma_map = {}

    for t_angle in set(list(all_data["clangle"])):
        clan_data = all_data[all_data["clangle"] == t_angle]

        for clbb in set(sorted(clan_data["clbb_b1"])):
            clangle_map[clbb] = t_angle

        for c1_opt in sorted(set(clan_data["c2sigma"])):
            test_data = clan_data[clan_data["c2sigma"] == c1_opt]
            for c2_opt in sorted(set(test_data["clsigma"])):
                plot_data = test_data[test_data["clsigma"] == c2_opt]

                for clbb in set(sorted(plot_data["clbb_b1"])):
                    clsigma_map[clbb] = c2_opt

                c2sigma_map[plot_data.iloc[0]["c2bb_b1"]] = c1_opt
                for bid, ba in zip(
                    list(plot_data["c2bb_b2"]),
                    list(plot_data["target_bite_angle"]),
                ):
                    bite_angle_map[bid] = ba

    logging.info(f"\nclangles: {clangle_map}\n")
    logging.info(f"\nclsigmas: {clsigma_map}\n")
    logging.info(f"\nbite_angles: {bite_angle_map}\n")
    logging.info(f"\nc2sigmas: {c2sigma_map}\n")


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    raise SystemExit(
        "next I want map of target bite angle to actual bite angle "
    )
    raise SystemExit(
        "next I want PCA maps of shapes for all in each topology "
    )
    raise SystemExit(
        "next I want PCA maps of all shapes for each BB property "
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
