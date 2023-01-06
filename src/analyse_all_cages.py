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
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


from generate_all_cages import (
    core_2c_beads,
    arm_2c_beads,
    beads_3c,
    beads_4c,
)
from env_set import cages


def colour_by_energy(energy):
    # cmap = Color.interpolate(
    #     ["rebeccapurple", "lch(85% 100 85)"],
    #     space="lch",
    # )
    # energy_bins = np.linspace(0, energy_max, 100)
    if energy <= isomer_energy():
        colorcode = "#345995"
    elif energy <= max_energy():
        # colorcode = cmap(101).to_string(hex=True)
        colorcode = "#F9A03F"
    else:
        # idx = np.argmin(np.abs(energy_bins - energy))
        # colorcode = cmap(idx).to_string(hex=True)
        colorcode = "#CA1551"

    return colorcode


def isomer_energy():
    return 0.05


def max_energy():
    return 1.0


def min_radius():
    return 0.1


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
        "M12L24": "12+24",
        "mixed": "mixed",
        "unstable": "unstable",
        "not": "not",
        "3C1": "3-coordinate",
        "4C1": "4-coordinate",
    }[topo_str]


def convert_torsion_to_label(topo_str):
    return {
        "ton": "5 eV",
        "toff": "none",
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
            "c2angle": (
                get_CGBead_from_string(
                    present_c2_beads[1],
                    arm_2c_beads(),
                ).angle_centered
                - 90
            )
            * 2,
            "energy": energy,
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


def identity_distributions(all_data, figure_output):

    fig, ax = plt.subplots(figsize=(16, 5))

    categories = {i: 0 for i in topology_labels(short=True)}
    categories.update({"not opt.": 0})
    count1 = all_data["topology"].value_counts()
    count2 = all_data["optimised"].value_counts()
    for tstr, count in count1.items():
        categories[convert_topo_to_label(tstr)] = count
    categories["not opt."] = count2[False]
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
    print(all_data.head())
    print(all_data.columns)
    to_plot = {
        "energy": {"xtitle": "energy"},
        "pore": {"xtitle": "min. distance [A]"},
        "energy_barm": {"xtitle": "energy"},
        "pore_barm": {"xtitle": "min. distance [A]"},
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
                sharex=True,
                figsize=(16, 5),
            )
            flat_axs = axs.flatten()
        else:
            cmapp = color_map
            fig, axs = plt.subplots(
                ncols=3,
                nrows=3,
                sharex=True,
                figsize=(16, 10),
            )
            flat_axs = axs.flatten()

        for i, t_option in enumerate(cmapp):
            target_column = tp.replace("_barm", "")
            if "barm" in tp:
                topo_frame = all_data[all_data["cltitle"] == t_option]
                values = topo_frame[target_column]
            else:
                topo_frame = all_data[all_data["topology"] == t_option]
                values = topo_frame[target_column]

            flat_axs[i].hist(
                x=values,
                bins=50,
                density=False,
                histtype="step",
                color=cmapp[t_option],
                lw=3,
                label=t_option,
            )

            flat_axs[i].tick_params(
                axis="both",
                which="major",
                labelsize=16,
            )
            flat_axs[i].set_xlabel(xtitle, fontsize=16)
            # flat_axs[i].set_ylabel("count", fontsize=16)
            flat_axs[i].set_ylabel("log(count)", fontsize=16)
            flat_axs[i].set_title(t_option, fontsize=16)
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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


def phase_space_1(bb_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        cl_bbname = bb_triplet[1]
        c2_bbname = bb_triplet[0]
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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
        for bb_triplet in bb_data:
            bb_dict = bb_data[bb_triplet]
            # c2_bbname = bb_triplet[0]
            cl_bbname = bb_triplet[1]

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
                ax.set_ylabel("min distance", fontsize=16)
            elif i == "shape":
                ax.set_xlabel("min. shape", fontsize=16)
                ax.set_ylabel("min distance", fontsize=16)
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


def phase_space_3(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

    opt_data = all_data[all_data["optimised"]]
    groups = opt_data.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"].values()},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"].values()},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"].values()},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"].values()},
    }
    for gid, dfi in groups:
        bbtitle = gid[:3]
        for tors in ("ton", "toff"):
            fin_data = dfi[dfi["torsions"] == tors]
            energies = {
                str(row["topology"]): float(row["energy"])
                / stoich_map(str(row["topology"]))
                for i, row in fin_data.iterrows()
            }
            num_mixed = len(
                tuple(
                    i
                    for i in list(energies.values())
                    if i < isomer_energy()
                )
            )
            min_energy = min(energies.values())
            if min_energy > max_energy():
                topo_str = "unstable"
            elif num_mixed > 1:
                topo_str = "mixed"
            else:
                topo_str = list(energies.keys())[
                    list(energies.values()).index(min_energy)
                ]
            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        print(coords)
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

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_4(bb_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        cl_bbname = bb_triplet[1]

        min_energy = min(tuple(i[1] for i in b_dict.values()))
        min_e_dict = {
            i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
        }
        keys = list(min_e_dict.keys())
        min_energy_topo = keys[0][-1]

        shape_vector = get_shape_vector(b_dict)
        if "3C1" in cl_bbname:
            s_vectors_3c0[
                (bb_triplet[0], bb_triplet[1], min_energy_topo)
            ] = shape_vector
            for i in shape_vector:
                row_3c0.add(i)
        elif "4C1" in cl_bbname:
            s_vectors_4c0[
                (bb_triplet[0], bb_triplet[1], min_energy_topo)
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        if "4C1" in bb_triplet[1]:
            shapes = target_shapes_by_cltype("4C1")
        elif "3C1" in bb_triplet[1]:
            shapes = target_shapes_by_cltype("3C1")

        present_beads_names = get_present_beads(c2_bbname=bb_triplet[0])
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        if "4C1" in bb_triplet[1]:
            shapes = target_shapes_by_cltype("4C1")
        elif "3C1" in bb_triplet[1]:
            shapes = target_shapes_by_cltype("3C1")

        for shape in shapes:
            topo_str = target_individuals[shape]
            try:
                shape_value = b_dict[shape][0]
            except KeyError:
                continue
            energy = b_dict[shape][1]
            min_distance = b_dict[shape][2]
            x = shape_value
            y = min_distance
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
        ax.set_ylabel("min. distance [A]", fontsize=16)

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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
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
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        cl_bbname = bb_triplet[1]
        c2_bbname = bb_triplet[0]
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    t_map = map_cltype_to_shapetopology()

    color_map = cltypetopo_to_colormap()

    data_dict = {}
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        cl_bbname = bb_triplet[1]
        c2_bbname = bb_triplet[0]

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
            tuple(i for i in all_energies if i < isomer_energy())
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

            if min_energy > max_energy():
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
            f"{max_energy()}eV : {isomer_energy()}eV"
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    t_map = map_cltype_to_shapetopology()
    color_map = cltypetopo_to_colormap()

    data_dict = {}
    for bb_triplet in bb_data:
        b_dict = bb_data[bb_triplet]
        cl_bbname = bb_triplet[1]
        c2_bbname = bb_triplet[0]

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
            tuple(i for i in all_energies if i < isomer_energy())
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
            min_e_distance = min_e_dict[keys[0]][2]
            topology = t_map[title][min_energy_topo]
            if min_e_distance < min_radius:
                s = "k"
            elif min_energy > max_energy():
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
            f"{max_energy()}eV : {isomer_energy()}eV : {min_radius()}A"
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
            label=f"min distance < {min_radius}A",
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
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    t_map = map_cltype_to_shapetopology()
    for t_cltopo, torsions in itertools.product(
        (3, 4), ("ton", "toff")
    ):
        input_dict = {}
        data_dict = {}
        for bb_triplet in bb_data:
            b_dict = bb_data[bb_triplet]
            cl_bbname = bb_triplet[1]
            c2_bbname = bb_triplet[0]
            torsion = bb_triplet[2]
            if torsion != torsions:
                continue
            bb_string = f"{cl_bbname}_{c2_bbname}"

            present_c2_beads = get_present_beads(c2_bbname)
            present_cl_beads = get_present_beads(cl_bbname)

            cltopo = int(cl_bbname[0])
            if cltopo != t_cltopo:
                continue
            if cltopo == 4:
                clangle = get_CGBead_from_string(
                    present_cl_beads[0],
                    beads_4c(),
                ).angle_centered[0]
                cltitle = "4C1"
            elif cltopo == 3:
                clangle = get_CGBead_from_string(
                    present_cl_beads[0],
                    beads_3c(),
                ).angle_centered
                cltitle = "3C1"

            all_energies = set(
                b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
            )
            num_mixed = len(
                tuple(i for i in all_energies if i < isomer_energy())
            )
            if num_mixed > 1:
                if num_mixed == 2:
                    topology = "mixed (2)"
                elif num_mixed > 2:
                    topology = "mixed (>2)"
                min_e_distance = None

            else:
                min_energy = min(tuple(i[1] for i in b_dict.values()))
                min_e_dict = {
                    i: b_dict[i]
                    for i in b_dict
                    if b_dict[i][1] == min_energy
                }
                keys = list(min_e_dict.keys())
                min_energy_topo = keys[0][-1]
                min_e_distance = min_e_dict[keys[0]][2]
                topology = t_map[cltitle][min_energy_topo]

                if min_energy > max_energy():
                    topology = "unstable"

            cl_bead_libs = beads_3c().copy()
            cl_bead_libs.update(beads_4c())
            row = {
                "cltopo": cltopo,
                "clsigma": get_CGBead_from_string(
                    present_cl_beads[0],
                    cl_bead_libs,
                ).sigma,
                "clangle": clangle,
                "c2sigma": get_CGBead_from_string(
                    present_c2_beads[0],
                    core_2c_beads(),
                ).sigma,
                "c2angle": (
                    get_CGBead_from_string(
                        present_c2_beads[1],
                        arm_2c_beads(),
                    ).angle_centered
                    - 90
                )
                * 2,
                "pref_topology": topology,
                "pref_topology_pore": min_e_distance,
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
                        figure_output,
                        f"dist_10_{t_cltopo}_{torsions}_{prop}.pdf",
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
                        figure_output,
                        f"ps_10_{t_cltopo}_{torsions}_{prop}.pdf",
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
                        figure_output,
                        f"dist_10_{t_cltopo}_{torsions}_{prop}.pdf",
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
                        figure_output,
                        f"ps_10_{t_cltopo}_{torsions}_{prop}.pdf",
                    ),
                    dpi=720,
                    bbox_inches="tight",
                )
                plt.close()


def parity_1(all_data, figure_output):
    tcmap = topo_to_colormap()

    fig, ax = plt.subplots(figsize=(8, 5))
    opt_data = all_data[all_data["optimised"]]
    ton_data = opt_data[opt_data["torsions"] == "ton"]
    toff_data = opt_data[opt_data["torsions"] == "toff"]

    out_data = ton_data.merge(
        toff_data,
        on="cage_name",
    )

    max_xlim = 0
    max_ylim = 0
    for tstr in tcmap:
        t_data = out_data[out_data["topology_x"] == tstr]
        ydata = list(t_data["energy_x"])
        xdata = list(t_data["energy_y"])

        if len(xdata) > 0:
            ax.scatter(
                [i for i in xdata],
                [i for i in ydata],
                c=tcmap[tstr],
                edgecolor="none",
                s=30,
                alpha=1.0,
                label=convert_topo_to_label(tstr),
            )
            max_xlim = max((max_xlim, max(xdata)))
            max_ylim = max((max_ylim, max(ydata)))

    ax.plot((0, max_ylim), (0, max_ylim), c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("toff [eV]", fontsize=16)
    ax.set_ylabel("ton [eV]", fontsize=16)
    ax.set_xlim(0, max_xlim)
    ax.set_ylim(0, max_ylim)
    ax.legend(fontsize=16, ncol=3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "par_1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def not_opt_phase_space(all_data, figure_output):
    color_map = topo_to_colormap()
    tt_map = {j: i for i, j in enumerate(color_map)}
    fig, axs = plt.subplots(
        ncols=3,
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    unopt_data = all_data[~all_data["optimised"]]
    for tstr in tt_map:
        ax = flat_axs[tt_map[tstr]]
        topo_data = unopt_data[unopt_data["topology"] == tstr]

        toff_data = topo_data[topo_data["torsions"] == "toff"]
        c = "r"
        s = 100
        m = "s"
        x = toff_data["clangle"]
        y = toff_data["c2angle"]
        ax.scatter(
            x,
            y,
            c=c,
            marker=m,
            edgecolor="none",
            s=s,
            alpha=0.2,
        )

        ton_data = topo_data[topo_data["torsions"] == "ton"]
        c = "k"
        s = 50
        m = "o"
        x = ton_data["clangle"]
        y = ton_data["c2angle"]
        ax.scatter(
            x,
            y,
            c=c,
            marker=m,
            edgecolor="none",
            s=s,
            alpha=0.2,
        )

        count_str = f"{len(toff_data)}/{len(ton_data)}"
        ax.set_title(f"{tstr}: {count_str}", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        # ax.set_xlabel("target 2c core size", fontsize=16)
        ax.set_xlabel("target Lc angle", fontsize=16)
        ax.set_ylabel("target 2c bite angle", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_notopt.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def bite_angle_relationship(all_data, figure_output):
    color_map = topo_to_colormap()
    tt_map = {j: i for i, j in enumerate(color_map)}

    opt_data = all_data[all_data["optimised"]]
    clangle_data = opt_data[opt_data["clangle"] == 120]
    for torsion in ("ton", "toff"):
        if torsion == "ton":
            c = "gray"
            m = "s"
        elif torsion == "toff":
            c = "r"
            m = "o"

        tor_data = clangle_data[clangle_data["torsions"] == torsion]

        fig, axs = plt.subplots(
            ncols=3,
            nrows=3,
            sharex=True,
            sharey=True,
            figsize=(16, 10),
        )
        flat_axs = axs.flatten()
        for tstr in tt_map:
            filt_data = tor_data[tor_data["topology"] == tstr]
            ax = flat_axs[tt_map[tstr]]
            for c1_option in sorted(set(filt_data["c2sigma"])):
                test_data = filt_data[filt_data["c2sigma"] == c1_option]
                for c2_option in sorted(set(test_data["clsigma"])):
                    plot_data = test_data[
                        test_data["clsigma"] == c2_option
                    ]
                    xs = list(plot_data["c2angle"])
                    ys = list(plot_data["energy"])
                    xs, ys = zip(*sorted(zip(xs, ys)))
                    ax.plot(
                        xs,
                        ys,
                        c=c,
                        lw=2,
                        marker=m,
                        alpha=1.0,
                    )

                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.set_xlabel("target 2c bite angle", fontsize=16)
                ax.set_ylabel("energy", fontsize=16)

                title = f"{tstr} : 120"
                ax.set_title(title, fontsize=16)
        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"anglerelation_{torsion}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


def visualise_high_energy(all_data, figure_output):
    opt_data = all_data[all_data["optimised"]]
    high_energy = opt_data[opt_data["energy"] > 500]
    high_e_names = list(high_energy["index"])
    logging.info(
        f"there are {len(high_e_names)} high energy structures"
    )
    with open(figure_output / "high_energy_names.txt", "w") as f:
        f.write("_opted3.mol ".join(high_e_names))
        f.write("_opted3.mol")


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
        output_csv=calculation_output / "all_array.csv",
    )
    logging.info(f"there are {len(all_data)} collected data")

    phase_space_3(all_data, figure_output)
    identity_distributions(all_data, figure_output)
    single_value_distributions(all_data, figure_output)
    not_opt_phase_space(all_data, figure_output)
    bite_angle_relationship(all_data, figure_output)
    visualise_high_energy(all_data, figure_output)
    rmsd_distributions(all_data, calculation_output, figure_output)
    parity_1(all_data, figure_output)

    raise SystemExit()
    phase_space_10(all_data, figure_output)

    shape_vector_distributions(all_data, figure_output)
    phase_space_9(all_data, figure_output)
    phase_space_1(all_data, figure_output)
    phase_space_5(all_data, figure_output)
    phase_space_6(all_data, figure_output)
    phase_space_7(all_data, figure_output)
    phase_space_8(all_data, figure_output)
    phase_space_4(all_data, figure_output)
    raise SystemExit()
    phase_space_2(all_data, figure_output)
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
