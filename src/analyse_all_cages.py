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
import numpy as np
import logging

# from coloraide import Color
import itertools
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from generate_all_cages import (
    core_2c_beads,
    arm_2c_beads,
    beads_3c,
    beads_4c,
)
from env_set import cages
from shape import known_shape_vectors
from visualisation import Pymol


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
        "all": "all",
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
    # categories.update({"X opt. [3]": 0, "X opt. [4]": 0})
    count1 = all_data["topology"].value_counts()
    # data_3c1 = all_data[all_data["cltitle"] == "3C1"]
    # count3c1 = data_3c1["optimised"].value_counts()
    # data_4c1 = all_data[all_data["cltitle"] == "4C1"]
    # count4c1 = data_4c1["optimised"].value_counts()
    for tstr, count in count1.items():
        categories[convert_topo_to_label(tstr)] = count
    # categories["X opt. [3]"] = count3c1[False]
    # categories["X opt. [4]"] = count4c1[False]
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
    tcmap.update({"all": "k"})
    tcpos.update({"all": 10})

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

    ax.plot((-1, 11), (0, 0), c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("RMSD [A]", fontsize=16)
    ax.set_xlim(-0.5, 10.5)
    ax.set_title("opt1->opt2; opt2->opt3", fontsize=16)
    ax.set_xticks([tcpos[i] for i in tcpos])
    ax.set_xticklabels(
        [convert_topo_to_label(i) for i in tcpos],
        rotation=45,
    )

    ax2.plot((-1, 11), (0, 0), c="k")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    # ax2.set_ylabel("RMSD [A]", fontsize=16)
    ax2.set_xlim(-0.5, 10.5)
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
        "energy_barm": {"xtitle": "energy", "xlim": (0, 100)},
        "pore_barm": {"xtitle": "min. distance [A]", "xlim": (0, 20)},
    }

    color_map = topo_to_colormap()
    color_map_barm = cltype_to_colormap()

    for tp in to_plot:

        xtitle = to_plot[tp]["xtitle"]
        xlim = to_plot[tp]["xlim"]
        if "barm" in tp:
            cmapp = color_map_barm
            fig, axs = plt.subplots(
                ncols=2,
                nrows=1,
                sharex=True,
                sharey=True,
                figsize=(16, 5),
            )
            flat_axs = axs.flatten()
        else:
            cmapp = color_map
            fig, axs = plt.subplots(
                ncols=2,
                nrows=5,
                sharex=True,
                sharey=True,
                figsize=(16, 12),
            )
            flat_axs = axs.flatten()

        for i, t_option in enumerate(cmapp):
            target_column = tp.replace("_barm", "")
            if "barm" in tp:
                topo_frame = all_data[all_data["cltitle"] == t_option]
                toff_frame = topo_frame[
                    topo_frame["torsions"] == "toff"
                ]
                ton_frame = topo_frame[topo_frame["torsions"] == "ton"]
                ton_values = ton_frame[target_column]
                toff_values = toff_frame[target_column]
            else:
                topo_frame = all_data[all_data["topology"] == t_option]
                toff_frame = topo_frame[
                    topo_frame["torsions"] == "toff"
                ]
                ton_frame = topo_frame[topo_frame["torsions"] == "ton"]
                ton_values = ton_frame[target_column]
                toff_values = toff_frame[target_column]

            xbins = np.linspace(xlim[0], xlim[1], 100)
            flat_axs[i].hist(
                x=ton_values,
                bins=xbins,
                density=False,
                histtype="step",
                color="k",
                lw=2,
                label=convert_torsion_to_label("ton"),
            )
            flat_axs[i].hist(
                x=toff_values,
                bins=xbins,
                density=False,
                histtype="step",
                color="r",
                lw=2,
                linestyle="--",
                label=convert_torsion_to_label("toff"),
            )

            flat_axs[i].tick_params(
                axis="both",
                which="major",
                labelsize=16,
            )
            flat_axs[i].set_xlabel(xtitle, fontsize=16)
            flat_axs[i].set_ylabel("count", fontsize=16)
            # flat_axs[i].set_ylabel("log(count)", fontsize=16)
            flat_axs[i].set_title(
                convert_topo_to_label(t_option),
                fontsize=16,
            )
            flat_axs[i].set_xlim(xlim)
            # flat_axs[i].set_yscale("log")
            if i == 0:
                flat_axs[i].legend(fontsize=16)

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


def phase_space_2(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    axmap = (
        {
            "ax": flat_axs[0],
            "tor": "toff",
            "x": "pore",
            "y": "energy",
        },
        {
            "ax": flat_axs[2],
            "tor": "ton",
            "x": "pore",
            "y": "energy",
        },
        {
            "ax": flat_axs[1],
            "tor": "toff",
            "x": "sv_n_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[3],
            "tor": "ton",
            "x": "sv_n_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[4],
            "tor": "toff",
            "x": "min_b2b",
            "y": "energy",
        },
        {
            "ax": flat_axs[6],
            "tor": "ton",
            "x": "min_b2b",
            "y": "energy",
        },
        {
            "ax": flat_axs[5],
            "tor": "toff",
            "x": "sv_l_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[7],
            "tor": "ton",
            "x": "sv_l_dist",
            "y": "energy",
        },
    )
    for axd in axmap:
        ax = axd["ax"]
        tdata = all_data[all_data["torsions"] == axd["tor"]]
        edata = tdata[tdata["energy"] < max_energy() * 50]
        xvalues = edata[axd["x"]]
        yvalues = edata[axd["y"]]
        hb = ax.hexbin(
            xvalues,
            yvalues,
            gridsize=20,
            cmap="inferno",
            bins="log",
            vmax=len(all_data),
        )
        cbar = fig.colorbar(hb, ax=ax, label="log10(N)")
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"{axd['x']}", fontsize=16)
        ax.set_ylabel(f"{axd['y']}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_2.pdf"),
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

        if "mixed" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["mixed"] = 0
        if "mixed" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["mixed"] = 0
        if "unstable" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["unstable"] = 0
        if "unstable" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["unstable"] = 0

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
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
            if val < 20:
                move = 20
            else:
                move = -20
            ax.text(
                i,
                val + move,
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


def phase_space_11(all_data, figure_output):
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
            for tstr in topologies[bbtitle].values():
                t_data = fin_data[fin_data["topology"] == tstr]
                if len(t_data) != 1:
                    continue
                if t_data.iloc[0]["persistent"]:
                    topo_str = tstr
                else:
                    topo_str = "not"

                if topo_str not in data[(bbtitle, tors)]:
                    data[(bbtitle, tors)][topo_str] = 0
                data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
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
            if val < 20:
                move = 20
            else:
                move = -20
            ax.text(
                i,
                val + move,
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
        os.path.join(figure_output, "ps_11.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_12(all_data, figure_output):
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
            per_data = fin_data[fin_data["persistent"]]
            present_topologies = list(per_data["topology"])
            if len(present_topologies) == 1:
                topo_str = present_topologies[0]
            else:
                topo_str = "mixed"

            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
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
            if val < 30:
                move = 20
            else:
                move = -20
            ax.text(
                i,
                val + move,
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
        os.path.join(figure_output, "ps_12.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_13(all_data, figure_output):

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    color_map = topo_to_colormap()
    for ax, tstr in zip(flat_axs, color_map):
        t_data = all_data[all_data["topology"] == tstr]
        n_values = list(t_data["sv_n_dist"])
        ax.hist(
            x=n_values,
            bins=50,
            density=False,
            histtype="step",
            color="#DD1C1A",
            lw=3,
        )

        filt_data = t_data[t_data["sv_l_dist"].notna()]
        if len(filt_data) > 0:
            l_values = list(t_data["sv_l_dist"])
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
        ax.set_xlabel("cosine similarity", fontsize=16)
        ax.set_ylabel("log(count)", fontsize=16)
        ax.set_title(tstr, fontsize=16)
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_13.pdf"),
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


def phase_space_5(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 8),
    )
    flat_axs = axs.flatten()
    cmap = topo_to_colormap()
    for ax, tstr in zip(flat_axs, cmap):
        tdata = all_data[all_data["topology"] == tstr]
        tondata = tdata[tdata["torsions"] == "ton"]
        toffdata = tdata[tdata["torsions"] == "toff"]
        x1 = tondata["c2angle"]
        x2 = toffdata["c2angle"]
        y1 = tondata["sv_n_dist"]
        y2 = toffdata["sv_n_dist"]
        z1 = tondata["energy"]
        z2 = toffdata["energy"]

        ax.set_title(tstr, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("c2angle", fontsize=16)
        ax.set_ylabel("sv_n_dist", fontsize=16)

        ax.scatter(
            x1,
            y1,
            c=z1,
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
    raise SystemExit()


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
    for torsion in ("ton", "toff"):
        tor_data = all_data[all_data["torsions"] == torsion]
        for tstr in color_map:
            filt_data = tor_data[tor_data["topology"] == tstr]
            for t_angle in set(list(filt_data["clangle"])):
                clan_data = filt_data[filt_data["clangle"] == t_angle]
                fig, ax = plt.subplots(figsize=(8, 5))
                for c1_opt in sorted(set(clan_data["c2sigma"])):
                    test_data = clan_data[
                        clan_data["c2sigma"] == c1_opt
                    ]
                    for c2_opt in sorted(set(test_data["clsigma"])):
                        plot_data = test_data[
                            test_data["clsigma"] == c2_opt
                        ]
                        xs = list(plot_data["c2angle"])
                        ys = list(plot_data["energy"])
                        xs, ys = zip(*sorted(zip(xs, ys)))
                        ax.plot(
                            xs,
                            ys,
                            lw=3,
                            alpha=1.0,
                            label=f"{c1_opt}.{c2_opt}",
                            marker="o",
                        )

                    ax.tick_params(
                        axis="both", which="major", labelsize=16
                    )
                    ax.set_xlabel("target 2c bite angle", fontsize=16)
                    ax.set_ylabel("energy", fontsize=16)
                    ax.legend()
                    ax.set_ylim(0, 2 * max_energy())

                fig.tight_layout()
                filename = f"ar_{torsion}_{t_angle}_{tstr}.pdf"
                fig.savefig(
                    os.path.join(figure_output, filename),
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
                    list(plot_data["c2angle"]),
                ):
                    bite_angle_map[bid] = ba

    logging.info(f"\nclangles: {clangle_map}\n")
    logging.info(f"\nclsigmas: {clsigma_map}\n")
    logging.info(f"\nbite_angles: {bite_angle_map}\n")
    logging.info(f"\nc2sigmas: {c2sigma_map}\n")


def visualise_bite_angle(all_data, figure_output):
    struct_output = cages() / "structures"
    topologies = (
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

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    trim = all_data[all_data["clsigma"] == 2]
    trim = trim[trim["c2sigma"] == 5]

    for tstr in topologies:
        tdata = trim[trim["topology"] == tstr]
        clangles = set(tdata["clangle"])

        for clangle in clangles:
            fig, axs = plt.subplots(
                ncols=19,
                nrows=2,
                figsize=(16, 5),
            )
            cdata = tdata[tdata["clangle"] == clangle]

            for i, tors in enumerate(("ton", "toff")):
                flat_axs = axs[i].flatten()

                show = cdata[cdata["torsions"] == tors]
                names_energies = [
                    (
                        str(row["index"]),
                        float(row["energy"]),
                        float(row["c2angle"]),
                    )
                    for idx, row in show.iterrows()
                ]
                names_energies = sorted(
                    names_energies, key=lambda tup: tup[2]
                )

                for cage_data, ax in zip(names_energies, flat_axs):
                    name, energy, ba = cage_data
                    structure_file = struct_output / f"{name}_optc.mol"
                    structure_colour = colour_by_energy(energy)
                    png_file = figure_output / f"{name}_f.png"
                    if not os.path.exists(png_file):
                        viz = Pymol(
                            output_dir=figure_output,
                            file_prefix=f"{name}_f",
                            settings=settings,
                        )
                        viz.visualise(
                            [structure_file], [structure_colour]
                        )

                    img = mpimg.imread(png_file)
                    ax.imshow(img)
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(f"{ba}", fontsize=16)

            ax.plot(
                [None, None],
                [None, None],
                c=colour_by_energy(max_energy() + 1),
                lw=3,
                label=f"energy > {max_energy()}eV",
            )
            ax.plot(
                [None, None],
                [None, None],
                c=colour_by_energy(isomer_energy() + 0.1),
                lw=3,
                label=f"energy <= {max_energy()}eV",
            )
            ax.plot(
                [None, None],
                [None, None],
                c=colour_by_energy(0),
                lw=3,
                label=f"energy <= {isomer_energy()}eV",
            )

            fig.legend(
                bbox_to_anchor=(0, 1.02, 2, 0.2),
                loc="lower left",
                ncol=3,
                fontsize=16,
            )
            fig.tight_layout()
            filename = f"vba_{tstr}_{clangle}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def visualise_self_sort(all_data, figure_output):
    struct_output = cages() / "structures"
    topology_dict = cltypetopo_to_colormap()

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    trim = all_data[all_data["clsigma"] == 2]
    trim = trim[trim["c2sigma"] == 5]

    bbpairs = set(trim["bbpair"])

    for bbpair in bbpairs:
        fig, axs = plt.subplots(
            ncols=5,
            nrows=2,
            figsize=(16, 5),
        )
        bdata = trim[trim["bbpair"] == bbpair]

        for i, tors in enumerate(("ton", "toff")):
            flat_axs = axs[i].flatten()

            show = bdata[bdata["torsions"] == tors]
            ctitle = "4C1" if "4C1" in bbpair else "3C1"
            c2angle = float(list(show["c2angle"])[0])
            names_energies = []
            for tstr in topology_dict[ctitle]:
                row = show[show["topology"] == tstr].iloc[0]
                names_energies.append(
                    (str(row["index"]), float(row["energy"])),
                )

            for j, cage_data in enumerate(names_energies):
                ax = flat_axs[j]
                name, energy = cage_data
                structure_file = struct_output / f"{name}_optc.mol"
                structure_colour = colour_by_energy(energy)
                png_file = figure_output / f"{name}_f.png"
                if not os.path.exists(png_file):
                    viz = Pymol(
                        output_dir=figure_output,
                        file_prefix=f"{name}_f",
                        settings=settings,
                    )
                    viz.visualise([structure_file], [structure_colour])

                img = mpimg.imread(png_file)
                ax.imshow(img)
                ax.axis("off")
                if i == 0 and j == 0:
                    ax.set_title(f"ba: {c2angle}", fontsize=16)

        ax.plot(
            [None, None],
            [None, None],
            c=colour_by_energy(max_energy() + 1),
            lw=3,
            label=f"energy > {max_energy()}eV",
        )
        ax.plot(
            [None, None],
            [None, None],
            c=colour_by_energy(isomer_energy() + 0.1),
            lw=3,
            label=f"energy <= {max_energy()}eV",
        )
        ax.plot(
            [None, None],
            [None, None],
            c=colour_by_energy(0),
            lw=3,
            label=f"energy <= {isomer_energy()}eV",
        )

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=3,
            fontsize=16,
        )
        fig.tight_layout()
        filename = f"vss_{bbpair}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()

    raise SystemExit()


def visualise_high_energy(all_data, figure_output):
    struct_output = cages() / "structures"
    high_energy = all_data[all_data["energy"] > 500]
    high_e_names = list(high_energy["index"])
    high_e_energies = list(high_energy["energy"])
    logging.info(
        f"there are {len(high_e_names)} high energy structures"
    )
    with open(figure_output / "high_energy_names.txt", "w") as f:
        f.write("_opted3.mol ".join(high_e_names))
        f.write("_opted3.mol")

    cage_name = high_e_names[0]
    structure_files = []
    structure_colours = []
    for i, cage_name in enumerate(high_e_names):
        structure_files.append(struct_output / f"{cage_name}_optc.mol")
        structure_colours.append(colour_by_energy(high_e_energies[i]))
    viz = Pymol(output_dir=figure_output, file_prefix="highe")
    viz.visualise(structure_files, structure_colours)


def energy_map(all_data, figure_output):

    cols_to_map = [
        "clsigma",
        "c2sigma",
        "torsions",
    ]
    cols_to_iter = [
        "clangle",
        "topology",
        # "c2angle",
    ]

    io1 = sorted(set(all_data[cols_to_iter[0]]))
    io2 = sorted(set(all_data[cols_to_iter[1]]))
    for cla in io1:
        for to in io2:
            filt_data = all_data[all_data[cols_to_iter[0]] == cla]
            filt_data = filt_data[filt_data[cols_to_iter[1]] == to]
            if len(filt_data) == 0:
                continue
            uo1 = sorted(set(filt_data[cols_to_map[0]]))
            uo2 = sorted(set(filt_data[cols_to_map[1]]))
            uo3 = sorted(set(filt_data[cols_to_map[2]]))

            gs = grid_spec.GridSpec(len(uo1), len(uo2))
            fig = plt.figure(figsize=(16, 10))
            ax_objs = []
            max_y = 0
            for i, o1 in enumerate(uo1):
                for j, o2 in enumerate(uo2):
                    ax_objs.append(
                        fig.add_subplot(gs[i : i + 1, j : j + 1])
                    )
                    for k, o3 in enumerate(uo3):
                        plot_data = filt_data[
                            filt_data[cols_to_map[0]] == o1
                        ]
                        plot_data = plot_data[
                            plot_data[cols_to_map[1]] == o2
                        ]
                        plot_data = plot_data[
                            plot_data[cols_to_map[2]] == o3
                        ]
                        if len(plot_data) == 0:
                            continue
                        ax = ax_objs[-1]
                        rect = ax.patch
                        rect.set_alpha(0)

                        xs = list(plot_data["c2angle"])
                        ys = list(plot_data["energy"])
                        xs, ys = zip(*sorted(zip(xs, ys)))
                        ax.plot(
                            xs,
                            ys,
                            c=torsion_to_colormap()[o3],
                            lw=3,
                            alpha=1.0,
                            label=f"{convert_torsion_to_label(o3)}",
                            marker="o",
                        )
                        # ax.set_title(" min e value")
                        ax.set_title(f"{o1}.{o2}", fontsize=16)
                        if i == 0 and j == 0:
                            ax.legend(fontsize=16)
                        if i == 1 and j == 0:
                            ax.set_ylabel("energy [eV]", fontsize=16)
                        if i == 2 and j == 1:
                            ax.set_xlabel(
                                "bite angle [deg]",
                                fontsize=16,
                            )
                        ax.tick_params(
                            axis="both",
                            which="both",
                            bottom=False,
                            top=False,
                            left=False,
                            right=False,
                            labelsize=16,
                        )
                        if i == 2:
                            ax.tick_params(
                                axis="y",
                                which="major",
                                labelsize=16,
                            )
                        else:
                            ax.set_xticklabels([])
                        if j == 0:
                            ax.tick_params(
                                axis="x",
                                which="major",
                                labelsize=16,
                            )
                        else:
                            ax.set_yticklabels([])

                        ax.axhline(y=max_energy())
                        max_y = max([max_y, max(ys)])

            for i, ax in enumerate(ax_objs):
                ax.set_ylim(0, max_y)
                spines = ["top", "right", "left", "bottom"]
                for s in spines:
                    ax.spines[s].set_visible(False)

            fig.tight_layout()
            filename = f"em_{cla}_{to}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()
    raise SystemExit()


def size_parities(all_data, figure_output):

    properties = {
        "energy": (0, 1),
        "gnorm": (0, 0.001),
        "pore": (0, 20),
        "min_b2b": (0, 1),
        "sv_n_dist": (0, 1),
        "sv_l_dist": (0, 1),
    }
    comps = {
        "1->5": (1, 5, "k"),
        "1->10": (1, 10, "skyblue"),
        "5->10": (5, 10, "gold"),
    }
    comp_cols = ("clsigma", "c2sigma")

    for tstr in sorted(set(all_data["topology"])):
        tdata = all_data[all_data["topology"] == tstr]
        for tors in ("ton", "toff"):
            tor_data = tdata[tdata["torsions"] == tors]

            fig, axs = plt.subplots(
                ncols=3,
                nrows=4,
                figsize=(16, 12),
            )
            flat_axs = axs.flatten()

            for prop, ax1, ax2 in zip(
                properties, flat_axs[:6], flat_axs[6:]
            ):
                ptup = properties[prop]
                for comp in comps:
                    ctup = comps[comp]
                    xdata = tor_data[tor_data[comp_cols[0]] == ctup[0]]
                    ydata = tor_data[tor_data[comp_cols[0]] == ctup[1]]
                    ax1.scatter(
                        xdata[prop],
                        ydata[prop],
                        c=ctup[2],
                        edgecolor="none",
                        s=30,
                        alpha=1.0,
                        # label=comp,
                    )

                    xdata = tor_data[tor_data[comp_cols[1]] == ctup[0]]
                    ydata = tor_data[tor_data[comp_cols[1]] == ctup[1]]
                    ax2.scatter(
                        xdata[prop],
                        ydata[prop],
                        c=ctup[2],
                        edgecolor="none",
                        s=30,
                        alpha=1.0,
                        # label=comp,
                    )

                    ax1.tick_params(
                        axis="both",
                        which="major",
                        labelsize=16,
                    )
                    ax1.set_xlabel(f"{prop}: smaller", fontsize=16)
                    ax1.set_ylabel(f"{prop}: larger", fontsize=16)
                    ax1.set_title(f"{comp_cols[0]}", fontsize=16)
                    ax1.set_xlim(ptup[0], ptup[1])
                    ax1.set_ylim(ptup[0], ptup[1])
                    # ax1.legend(fontsize=16, ncol=3)

                    ax2.tick_params(
                        axis="both",
                        which="major",
                        labelsize=16,
                    )
                    ax2.set_xlabel(f"{prop}: smaller", fontsize=16)
                    ax2.set_ylabel(f"{prop}: larger", fontsize=16)
                    ax2.set_title(f"{comp_cols[1]}", fontsize=16)
                    ax2.set_xlim(ptup[0], ptup[1])
                    ax2.set_ylim(ptup[0], ptup[1])
                    # ax2.legend(fontsize=16, ncol=3)

                    ax1.plot(
                        (ptup[0], ptup[1]),
                        (ptup[0], ptup[1]),
                        c="k",
                    )
                    ax2.plot(
                        (ptup[0], ptup[1]),
                        (ptup[0], ptup[1]),
                        c="k",
                    )

            for comp in comps:
                ctup = comps[comp]
                ax1.scatter(
                    None,
                    None,
                    c=ctup[2],
                    edgecolor="none",
                    s=30,
                    alpha=1.0,
                    label=comp,
                )
            fig.legend(
                bbox_to_anchor=(0, 1.02, 2, 0.2),
                loc="lower left",
                ncol=3,
                fontsize=16,
            )
            fig.tight_layout()
            filename = f"sp_{tstr}_{tors}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def selfsort_map(all_data, figure_output):
    cols_to_map = ["clsigma", "clangle"]
    cols_to_iter = ["torsions", "cltitle", "c2sigma", "c2angle"]

    io1 = sorted(set(all_data[cols_to_iter[0]]))
    io2 = sorted(set(all_data[cols_to_iter[1]]))
    for tor in io1:
        for cltitle in io2:
            tors_data = all_data[all_data[cols_to_iter[0]] == tor]
            tors_data = tors_data[tors_data[cols_to_iter[1]] == cltitle]
            fig = plt.figure(figsize=(16, 8))
            uo1 = sorted(set(tors_data[cols_to_map[0]]))
            uo2 = sorted(set(tors_data[cols_to_map[1]]))
            print(tor, cltitle, uo1, uo2)

            gs = grid_spec.GridSpec(len(uo1), len(uo2))
            ax_objs = []
            for (i, o1), (j, o2) in itertools.product(
                enumerate(uo1), enumerate(uo2)
            ):
                ax_objs.append(
                    fig.add_subplot(gs[i : i + 1, j : j + 1])
                )
                ax = ax_objs[-1]
                plot_data = tors_data[tors_data[cols_to_map[0]] == o1]
                plot_data = plot_data[plot_data[cols_to_map[1]] == o2]
                io3 = sorted(set(plot_data[cols_to_iter[2]]))
                io4 = sorted(set(plot_data[cols_to_iter[3]]))
                ylabel_map = {}
                for cla, ba in itertools.product(io3, io4):
                    xvalue = ba
                    yvalue = io3.index(cla)
                    yname = cla
                    ylabel_map[yname] = yvalue

                    filt_data = plot_data[
                        plot_data[cols_to_iter[2]] == cla
                    ]
                    filt_data = filt_data[
                        filt_data[cols_to_iter[3]] == ba
                    ]
                    if len(filt_data) == 0:
                        continue

                    energies = {
                        str(row["topology"]): float(row["energy"])
                        / stoich_map(str(row["topology"]))
                        for i, row in filt_data.iterrows()
                    }
                    dists = {
                        str(row["topology"]): float(row["pore"])
                        for i, row in filt_data.iterrows()
                    }
                    b2bs = {
                        str(row["topology"]): float(row["min_b2b"])
                        for i, row in filt_data.iterrows()
                    }
                    svs = {
                        str(row["topology"]): (
                            float(row["sv_n_dist"]),
                            float(row["sv_l_dist"]),
                        )
                        for i, row in filt_data.iterrows()
                    }
                    num_mixed = len(
                        tuple(
                            i
                            for i in list(energies.values())
                            if i < isomer_energy()
                        )
                    )
                    min_energy = min(energies.values())
                    min_dist = None
                    min_svs = None
                    min_b2b = None
                    if min_energy > max_energy():
                        topo_str = "unstable"
                        colour = "gray"
                    elif num_mixed > 1:
                        topo_str = "mixed"
                        colour = "white"
                    else:
                        topo_str = list(energies.keys())[
                            list(energies.values()).index(min_energy)
                        ]
                        min_dist = dists[topo_str]
                        min_svs = svs[topo_str]
                        min_b2b = b2bs[topo_str]
                        colour = cltypetopo_to_colormap()[cltitle][
                            topo_str
                        ]
                        # print(energies, topo_str, min_dist, min_svs)
                        # print(min_b2b)

                    rect = ax.patch
                    rect.set_alpha(0)
                    ax.scatter(
                        xvalue,
                        yvalue,
                        c=colour,
                        alpha=1.0,
                        marker="s",
                        edgecolor="k",
                        s=200,
                    )

                    if min_dist is not None and min_dist < min_radius():
                        ax.scatter(
                            xvalue,
                            yvalue,
                            c="k",
                            alpha=1.0,
                            marker="X",
                            edgecolor="none",
                            s=80,
                        )

                    if (
                        min_b2b is not None
                        and min_b2b < min_b2b_distance()
                    ):
                        ax.scatter(
                            xvalue,
                            yvalue,
                            c="k",
                            alpha=1.0,
                            marker="D",
                            edgecolor="none",
                            s=80,
                        )

                    ax.set_title(
                        f"{cols_to_map[0]}:{o1}; {cols_to_map[1]}:{o2}",
                        fontsize=16,
                    )
                    if i == 1 and j == 0:
                        # ax.set_ylabel("CL angle [deg]", fontsize=16)
                        ax.set_ylabel("C2 sigma [A]", fontsize=16)
                    if i == 2 and j == 1:
                        ax.set_xlabel(
                            "bite angle [deg]",
                            fontsize=16,
                        )
                    ax.tick_params(
                        axis="both",
                        which="both",
                        bottom=False,
                        top=False,
                        left=False,
                        right=False,
                        labelsize=16,
                    )
                    if i == 2:
                        ax.tick_params(
                            axis="y",
                            which="major",
                            labelsize=16,
                        )
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.tick_params(
                            axis="x",
                            which="major",
                            labelsize=16,
                        )
                        ax.set_yticks(list(ylabel_map.values()))
                        ax.set_yticklabels(list(ylabel_map.keys()))
                    else:
                        ax.set_yticklabels([])

                for i, ax in enumerate(ax_objs):
                    spines = ["top", "right", "left", "bottom"]
                    for s in spines:
                        ax.spines[s].set_visible(False)
                    ax.set_ylim(-0.5, 2.5)

            for i in cltypetopo_to_colormap():
                if i not in (cltitle, "mixed"):
                    continue
                for j in cltypetopo_to_colormap()[i]:
                    # if i == "mixed":
                    #     string = f"mixed: {j}"
                    # else:
                    string = j
                    ax.scatter(
                        None,
                        None,
                        c=cltypetopo_to_colormap()[i][j],
                        edgecolor="k",
                        s=300,
                        marker="s",
                        alpha=1.0,
                        label=convert_topo_to_label(string),
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
            ax.scatter(
                None,
                None,
                c="k",
                edgecolor="k",
                s=300,
                marker="s",
                alpha=1.0,
                label=f"min distance < {min_radius()}A",
            )

            fig.legend(
                bbox_to_anchor=(0, 1.02, 2, 0.2),
                loc="lower left",
                ncol=4,
                fontsize=16,
            )

            fig.tight_layout()
            filename = f"ss_{cltitle}_{tor}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=720,
                bbox_inches="tight",
            )
            plt.close()


def parallel_plot(all_data, figure_output):
    return None
    print(all_data.columns)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax = parallel_coordinates(
        frame=all_data,
        class_column="topology",
        cols=["clangle", "clsigma", "c2sigma", "c2angle"],
        ax=ax,
        color=None,
        use_columns=False,
        xticks=None,
        colormap=None,
        axvlines=True,
        axvlines_kwds=None,
        sort_labels=False,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.set_xlabel("target 2c bite angle", fontsize=16)
    # ax.set_ylabel("energy", fontsize=16)
    # ax.legend()
    # ax.set_ylim(0, 2 * max_energy())

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "pp.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
    raise SystemExit()


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
    identity_distributions(all_data, figure_output)
    write_out_mapping(all_data)
    visualise_bite_angle(all_data, figure_output)
    visualise_self_sort(all_data, figure_output)
    visualise_high_energy(all_data, figure_output)
    phase_space_5(all_data, figure_output)
    phase_space_2(all_data, figure_output)
    single_value_distributions(all_data, figure_output)
    size_parities(all_data, figure_output)
    selfsort_map(all_data, figure_output)
    energy_map(all_data, figure_output)
    shape_vector_distributions(all_data, figure_output)
    phase_space_3(all_data, figure_output)
    phase_space_13(all_data, figure_output)
    rmsd_distributions(all_data, calculation_output, figure_output)
    geom_distributions(all_data, geom_data, figure_output)
    bite_angle_relationship(all_data, figure_output)

    raise SystemExit()

    phase_space_11(all_data, figure_output)
    phase_space_12(all_data, figure_output)
    not_opt_phase_space(all_data, figure_output)

    parity_1(all_data, figure_output)

    raise SystemExit()
    phase_space_10(all_data, figure_output)

    phase_space_9(all_data, figure_output)
    phase_space_1(all_data, figure_output)

    phase_space_6(all_data, figure_output)
    phase_space_7(all_data, figure_output)
    phase_space_8(all_data, figure_output)
    phase_space_4(all_data, figure_output)
    raise SystemExit()

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
    parallel_plot(all_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
