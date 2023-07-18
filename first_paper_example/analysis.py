#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Utilities for analysis and plotting.

Author: Andrew Tarzia

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cgexplore.beads import get_CGBead_from_string
from cgexplore.shape import known_shape_vectors
from cgexplore.cage_construction.topologies import cage_topology_options

from bead_libraries import (
    core_2c_beads,
    arm_2c_beads,
    beads_3c,
    beads_4c,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def clangle_str(num=None):
    if num is None:
        return r"$x$C angle [$^\circ$]"
    else:
        return rf"{num}C angle [$^\circ$]"


def eb_str(no_unit=False):
    if no_unit:
        return r"$E_{\mathrm{b}}$"
    else:
        return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def pore_str():
    # return r"min(centroid-bead) [$\mathrm{\AA}$]"
    return r"pore size [$\mathrm{\AA}$]"


def rg_str():
    return r"$R_{\mathrm{g}}$ [$\mathrm{\AA}$]"


def shape_threshold():
    return 2


def isomer_energy():
    # kJ/mol.
    return 0.3


def min_radius():
    return 1.0


def min_b2b_distance():
    return 0.5


def topology_labels(short):
    if short == "+":
        return (
            "2+3",
            "4+6",
            "4+6(2)",
            "6+9",
            "8+12",
            "2+4",
            "3+6",
            "4+8",
            "4+8(2)",
            "6+12",
            "8+16",
            "12+24",
            "6+8",
        )
    elif short == "P":
        return (
            "2P3",
            "4P6",
            "4P62",
            "6P9",
            "8P12",
            "2P4",
            "3P6",
            "4P8",
            "4P82",
            "6P12",
            "8P16",
            "12P24",
            "6P8",
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
            "FourPlusEight2",
            "SixPlusTwelve",
            "EightPlusSixteen",
            "M12L24",
        )


def convert_prop(prop_str):
    raise NotImplementedError("check convert propr")
    return {
        "energy_per_bb": "E_b [eV]",
        "sv_n_dist": "node shape similarity",
        "sv_l_dist": "lig. shape similarity",
        "both_sv_n_dist": "shape similarity",
    }[prop_str]


def convert_outcome(topo_str):
    return {
        "mixed": "mixed",
        "mixed-2": "mixed (2)",
        "mixed-3": "mixed (3)",
        "mixed>3": "mixed (>3)",
        "all": "all",
        "unstable": "unstable",
        "not": "not",
    }[topo_str]


def convert_connectivity(topo_str):
    raise SystemExit("decide how to do this.")
    return {
        "3C1": "3-c",
        "4C1": "4-c",
    }[topo_str]


def convert_topo(topo_str):
    return {
        "2P3": r"Tri$^{2}$Di$^{3}$",
        "4P6": r"Tri$^{4}$Di$^{6}$",
        "4P62": r"Tri$^{4}_{2}$Di$^{6}$",
        "6P9": r"Tri$^{6}$Di$^{9}$",
        "8P12": r"Tri$^{8}$Di$^{12}$",
        "2P4": r"Tet$^{2}$Di$^{4}$",
        "3P6": r"Tet$^{3}_{3}$Di$^{6}$",
        "4P8": r"Tet$^{4}_{4}$Di$^{8}$",
        "4P82": r"Tet$^{4}_{2}$Di$^{8}$",
        "6P12": r"Tet$^{6}$Di$^{12}$",
        "8P16": r"Tet$^{8}$Di$^{16}$",
        "12P24": r"Tet$^{12}$Di$^{24}$",
        "6P8": r"Tet$^{6}$Tri$^{8}$",
    }[topo_str]


def convert_tors(topo_str, num=True):
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


def convert_vdws(vstr):
    return {
        "von": "excl. vol.",
        "voff": "no. NB",
    }[vstr]


def Xc_map(tstr):
    """
    Maps topology string to pyramid angle.

    """

    return {
        "2P3": 3,
        "4P6": 3,
        "4P62": 3,
        "6P9": 3,
        "8P12": 3,
        "2P4": 4,
        "3P6": 4,
        "4P8": 4,
        "4P82": 4,
        "6P12": 4,
        "8P16": 4,
        "12P24": 4,
        "6P8": 4,
    }[tstr]


def stoich_map(tstr):
    """
    Stoichiometry maps to the number of building blocks.

    """

    return {
        "2P3": 5,
        "4P6": 10,
        "4P62": 10,
        "6P9": 15,
        "8P12": 20,
        "2P4": 6,
        "3P6": 9,
        "4P8": 12,
        "4P82": 12,
        "6P12": 18,
        "8P16": 24,
        "12P24": 36,
        "6P8": 14,
    }[tstr]


def cltype_to_colormap():
    raise SystemExit("cltype_to_colormap, if this is used, fix")
    return {
        "3C1": "#06AED5",
        "4C1": "#086788",
    }


def cltypetopo_to_colormap():
    return {
        "3C1": {
            "2P3": "#1f77b4",
            "4P6": "#ff7f0e",
            "4P62": "#2ca02c",
            "6P9": "#d62728",
            "8P12": "#17becf",
        },
        "4C1": {
            "2P4": "#1f77b4",
            "3P6": "#ff7f0e",
            "4P8": "#2ca02c",
            "4P82": "#c057a1",
            "6P12": "#17becf",
            "8P16": "#75499c",
            "12P24": "#d62728",
        },
        "mixed": {
            # "2": "#7b4173",
            # ">2": "#de9ed6",
            # "mixed": "white",
            "mixed-2": "white",
            "mixed-3": "#8A8A8A",
            "mixed>3": "#434343",
        },
        "unstable": {
            "unstable": "white",
        },
    }


def shapevertices_to_colormap():
    raise SystemExit("shapevertices_to_colormap, if this is used, fix")
    return {
        4: "#06AED5",
        5: "#086788",
        6: "#DD1C1A",
        8: "#320E3B",
        3: "#6969B3",
    }


def shapelabels_to_colormap():
    raise SystemExit("shapelabels_to_colormap, if this is used, fix")
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
        # "TBPY-5",
        # "PP-5",
        "T-4",
        "SP-4",
        "TP-3",
        "mvOC-3",
    )


def target_shapes_by_cltype(cltype):
    raise SystemExit("target_shapes_by_cltype, if this is used, fix")
    if cltype == "4C1":
        return ("OC-6b", "TP-3", "SP-4", "OC-6")
    elif cltype == "3C1":
        return ("TBPY-5", "T-4", "T-4b", "TPR-6", "CU-8")


def shapetarget_to_colormap():
    raise SystemExit("shapetarget_to_colormap, if this is used, fix")
    return {
        "CU-8": "#06AED5",
        "OC-6": "#086788",
        "TBPY-5": "#DD1C1A",
        "T-4": "#320E3B",
        "TPR-6": "#CE7B91",
    }


def map_cltype_to_topology():
    return {
        "3C1": ("2P3", "4P6", "4P62", "6P9", "8P12"),
        "4C1": ("2P4", "3P6", "4P8", "4P82", "6P12", "8P16", "12P24"),
    }


def mapshape_to_topology(mode, from_shape=False):
    if from_shape:
        if mode == "n":
            return {
                "TP-3": ("3P6",),
                "mvOC-3": ("3P6",),
                "T-4": ("4P6", "4P62", "4P8", "4P82"),
                "SP-4": ("4P6", "4P62", "4P8", "4P82"),
                "OC-6": ("6P9", "6P12", "6P8"),
                "PPY-6": ("6P9", "6P12", "6P8"),
                "HP-6": ("6P9", "6P12", "6P8"),
                "CU-8": ("8P12", "8P16"),
                "JETBPY-8": ("8P12", "8P16"),
                "OP-8": ("8P12", "8P16"),
            }

        elif mode == "l":
            return {
                "TP-3": ("2P3",),
                "mvOC-3": ("2P3",),
                "T-4": ("2P4",),
                "SP-4": ("2P4",),
                "OC-6": ("4P6", "4P62"),
                "PPY-6": ("4P6", "4P62"),
                "HP-6": ("4P6", "4P62"),
                "CU-8": ("6P8",),
                "JETBPY-8": ("6P8",),
                "OP-8": ("6P8",),
            }
    else:
        if mode == "n":
            return {
                # "2P3": "TBPY-5",
                "4P6": "T-4",
                "4P62": "SP-4",
                "6P9": "TPR-6",
                "8P12": "CU-8",
                # "2P4": "OC-6b",
                "3P6": "TP-3",
                "4P8": "SP-4",
                "4P82": "T-4",
                "6P12": "OC-6",
                "8P16": "SAPR-8",
                # "12P24": "",
                "6P8": "OC-6",
            }

        elif mode == "l":
            return {
                "2P3": "TP-3",
                "4P6": "OC-6",
                "4P62": "OC-6",
                "2P4": "SP-4",
                "6P8": "CU-8",
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


def node_expected_topologies():
    return {
        "4P6": 4,
        "4P62": 4,
        "6P9": 6,
        "8P12": 8,
        "3P6": 3,
        "4P8": 4,
        "4P82": 4,
        "6P12": 6,
        "8P16": 8,
        "6P8": 6,
    }


def ligand_expected_topologies():
    return {
        "2P3": 3,
        "4P6": 6,
        "4P62": 6,
        "2P4": 4,
        "6P8": 8,
    }


def get_sv_dist(row, mode):
    maps = mapshape_to_topology(mode=mode)
    try:
        tshape = maps[str(row["topology"])]
    except KeyError:
        return None

    if tshape[-1] == "b":
        raise ValueError("I removed all uses of `shape`b label, check.")

    known_sv = known_shape_vectors()[tshape]
    current_sv = {i: float(row[f"{mode}_{i}"]) for i in known_sv}
    a = np.array([known_sv[i] for i in known_sv])
    b = np.array([current_sv[i] for i in known_sv])
    cosine_similarity = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return cosine_similarity


def is_persistent(row):
    raise NotImplementedError()

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
    json_files = sorted(json_files)
    len_jsons = len(json_files)
    count = 0
    for j_file in json_files:
        logging.info(f"arraying {j_file.name} ({count}/{len_jsons})")
        with open(j_file, "r") as f:
            res_dict = json.load(f)

        row = {}

        name = str(j_file.name).replace("_res.json", "")

        (
            t_str,
            clbb_name,
            c2bb_name,
            torsions,
            vdws,
            run_number,
        ) = name.split("_")

        row["cage_name"] = f"{t_str}_{clbb_name}_{c2bb_name}"
        row["clbb_name"] = clbb_name
        row["c2bb_name"] = c2bb_name
        row["topology"] = t_str
        row["torsions"] = torsions
        row["vdws"] = vdws
        row["run_number"] = run_number

        present_c2_beads = get_present_beads(c2bb_name)
        present_cl_beads = get_present_beads(clbb_name)
        row["clbb_b1"] = present_cl_beads[0]
        row["clbb_b2"] = present_cl_beads[1]
        row["c2bb_b1"] = present_c2_beads[0]
        row["c2bb_b2"] = present_c2_beads[1]

        cl_bead_libs = beads_3c().copy()
        cl_bead_libs.update(beads_4c())
        row["cltopo"] = int(clbb_name[0])
        clangle = get_CGBead_from_string(
            present_cl_beads[0],
            cl_bead_libs,
        ).angle_centered

        row["bbpair"] = clbb_name + c2bb_name
        row["optimised"] = res_dict["optimised"]
        # row["mdexploded"] = res_dict["mdexploded"]
        # row["mdfailed"] = res_dict["mdfailed"]

        if t_str in cage_topology_options(
            "2p3"
        ) or t_str in cage_topology_options("2p4"):
            cltitle = "3C1" if row["cltopo"] == 3 else "4C1"
            row["c2r0"] = get_CGBead_from_string(
                present_c2_beads[0],
                core_2c_beads(),
            ).bond_r
            row["c2angle"] = get_CGBead_from_string(
                present_c2_beads[1],
                arm_2c_beads(),
            ).angle_centered
            row["target_bite_angle"] = (
                get_CGBead_from_string(
                    present_c2_beads[1],
                    arm_2c_beads(),
                ).angle_centered
                - 90
            ) * 2

        elif t_str in cage_topology_options("3p4"):
            cltitle = "4C1"
            row["c3r0"] = get_CGBead_from_string(
                present_c2_beads[0],
                cl_bead_libs,
            ).bond_r
            row["c3angle"] = get_CGBead_from_string(
                present_c2_beads[0],
                cl_bead_libs,
            ).angle_centered

        row["cltitle"] = cltitle
        row["clr0"] = get_CGBead_from_string(
            present_cl_beads[0],
            cl_bead_libs,
        ).bond_r
        row["clangle"] = clangle

        if row["optimised"]:
            row["strain_energy"] = res_dict["fin_energy_kjmol"]
            row["energy_per_bb"] = res_dict[
                "fin_energy_kjmol"
            ] / stoich_map(t_str)
            for force_title in res_dict["fin_energy_decomp"]:
                if force_title in (
                    "CMMotionRemover_kJ/mol",
                    "tot_energy_kjmol",
                    "total energy_kJ/mol",
                ):
                    continue
                row[force_title] = res_dict["fin_energy_decomp"][
                    force_title
                ]

            row["pore"] = res_dict["opt_pore_data"]["min_distance"]
            row["min_b2b_distance"] = res_dict["min_b2b_distance"]
            row["radius_gyration"] = res_dict["radius_gyration"]
            row["max_diameter"] = res_dict["max_diameter"]
            row["rg_md"] = (
                res_dict["radius_gyration"] / res_dict["max_diameter"]
            )
            row["pore_md"] = (
                res_dict["opt_pore_data"]["min_distance"]
                / res_dict["max_diameter"]
            )
            row["pore_rg"] = (
                res_dict["opt_pore_data"]["min_distance"]
                / res_dict["radius_gyration"]
            )

            # trajectory_data = res_dict["trajectory"]
            # if trajectory_data is None:
            #     row["pore_dynamics"] = None
            #     row["structure_dynamics"] = None
            #     row["node_shape_dynamics"] = None
            #     row["lig_shape_dynamics"] = None
            # else:
            #     list_of_rgs = [
            #         rd["radius_gyration"]
            #         for rd in trajectory_data.values()
            #     ]
            #     list_of_pores = [
            #         rd["pore_data"]["min_distance"]
            #         for rd in trajectory_data.values()
            #     ]
            #     row["structure_dynamics"] = np.std(
            #         list_of_rgs
            #     ) / np.mean(list_of_rgs)
            #     row["pore_dynamics"] = np.std(list_of_pores) / np.mean(
            #         list_of_pores
            #     )

            #     list_of_node_sv_cosdists = []
            #     list_of_lig_sv_cosdists = []
            #     for rd in trajectory_data.values():
            #         rd_series = pd.Series(dtype="object")
            #         rd_series["topology"] = row["topology"]
            #         r_node_shape_vector = rd["node_shape_measures"]
            #         r_lig_shape_vector = rd["lig_shape_measures"]
            #         if r_node_shape_vector is not None:
            #             for sv in r_node_shape_vector:
            #                 rd_series[f"n_{sv}"] = r_node_shape_vector[
            #                     sv
            #                 ]
            #         if r_lig_shape_vector is not None:
            #             for sv in r_lig_shape_vector:
            #                 rd_series[f"l_{sv}"] = r_lig_shape_vector[
            #                     sv
            #                 ]
            #         list_of_node_sv_cosdists.append(
            #             get_sv_dist(rd_series, mode="n")
            #         )
            #         list_of_lig_sv_cosdists.append(
            #             get_sv_dist(rd_series, mode="l")
            #         )

            #     if list_of_node_sv_cosdists[0] is None:
            #         row["node_shape_dynamics"] = None
            #     else:
            #         row["node_shape_dynamics"] = np.std(
            #             list_of_node_sv_cosdists
            #         )
            #     if list_of_lig_sv_cosdists[0] is None:
            #         row["lig_shape_dynamics"] = None
            #     else:
            #         row["lig_shape_dynamics"] = np.std(
            #             list_of_lig_sv_cosdists
            #         )

            bond_data = res_dict["bond_data"]
            angle_data = res_dict["angle_data"]
            dihedral_data = res_dict["dihedral_data"]
            geom_data[name] = {
                "bonds": bond_data,
                "angles": angle_data,
                "dihedrals": dihedral_data,
            }

            node_shape_vector = res_dict["node_shape_measures"]
            lig_shape_vector = res_dict["lig_shape_measures"]
            if node_shape_vector is not None:
                for sv in node_shape_vector:
                    row[f"n_{sv}"] = node_shape_vector[sv]
            if lig_shape_vector is not None:
                for sv in lig_shape_vector:
                    row[f"l_{sv}"] = lig_shape_vector[sv]

        input_dict[name] = row
        count += 1

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
    clr0_map = {}
    c2r0_map = {}

    for t_angle in set(list(all_data["clangle"])):
        clan_data = all_data[all_data["clangle"] == t_angle]

        for clbb in set(sorted(clan_data["clbb_b1"])):
            clangle_map[clbb] = t_angle

        for c1_opt in sorted(set(clan_data["c2r0"])):
            test_data = clan_data[clan_data["c2r0"] == c1_opt]
            for c2_opt in sorted(set(test_data["clr0"])):
                plot_data = test_data[test_data["clr0"] == c2_opt]

                for clbb in set(sorted(plot_data["clbb_b1"])):
                    clr0_map[clbb] = c2_opt

                c2r0_map[plot_data.iloc[0]["c2bb_b1"]] = c1_opt
                for bid, ba in zip(
                    list(plot_data["c2bb_b2"]),
                    list(plot_data["target_bite_angle"]),
                ):
                    bite_angle_map[bid] = ba

    properties = [
        "energy_per_bb",
        "pore",
        "min_b2b_distance",
        "radius_gyration",
        "max_diameter",
        "rg_md",
        "pore_md",
        "pore_rg",
        "HarmonicBondForce_kJ/mol",
        "HarmonicAngleForce_kJ/mol",
        "CustomNonbondedForce_kJ/mol",
        "PeriodicTorsionForce_kJ/mol",
        # "structure_dynamics",
        # "pore_dynamics",
        # "node_shape_dynamics",
        # "lig_shape_dynamics",
        "sv_n_dist",
        "sv_l_dist",
    ]
    logging.info(f"\nclangles: {clangle_map}\n")
    logging.info(f"\nclr0s: {clr0_map}\n")
    logging.info(f"\nbite_angles: {bite_angle_map}\n")
    logging.info(f"\nc2r0s: {c2r0_map}\n")
    logging.info(f"available properties:\n {properties}\n")


def get_lowest_energy_data(all_data, output_dir):
    logging.info("defining low energy array")
    output_csv = output_dir / "lowe_array.csv"

    if os.path.exists(output_csv):
        input_array = pd.read_csv(output_csv)
        input_array = input_array.loc[
            :, ~input_array.columns.str.contains("^Unnamed")
        ]
        return input_array

    lowe_array = pd.DataFrame()
    grouped_data = all_data.groupby(["cage_name"])
    for system in set(all_data["cage_name"]):
        logging.info(f"checking {system}")
        rows = grouped_data.get_group(system)
        for tors in ("ton", "toff"):
            trows = rows[rows["torsions"] == tors]
            if len(trows) == 0:
                continue
            final_row = trows[
                trows["energy_per_bb"] == trows["energy_per_bb"].min()
            ]
            # Get lowest energy row, and add to new dict.
            lowe_array = pd.concat([lowe_array, final_row.iloc[[0]]])

    lowe_array.reset_index()
    lowe_array.to_csv(output_csv, index=False)

    return lowe_array