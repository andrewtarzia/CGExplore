#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Utilities for analysis and plotting.

Author: Andrew Tarzia

"""

import json
import logging
import os

import numpy as np
import openmm
import pandas as pd
from cgexplore.geom import GeomMeasure
from cgexplore.pore import PoreMeasure
from cgexplore.shape import (
    ShapeMeasure,
    get_shape_molecule_ligands,
    get_shape_molecule_nodes,
    known_shape_vectors,
)
from cgexplore.torsions import TargetTorsion
from env_set import shape_path
from topologies import cage_topology_options

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_paired_cage_name(cage_name):
    """
    Get new FF number from a cage number based on ton vs toff.

    """

    ff_num = int(cage_name.split("_")[-1].split("f")[1])
    new_ff_num = ff_num + 1
    new_name = cage_name.replace(f"f{ff_num}", f"f{new_ff_num}")
    return new_name


def analyse_cage(
    conformer,
    name,
    output_dir,
    force_field,
    node_element,
    ligand_element,
):
    output_file = os.path.join(output_dir, f"{name}_res.json")
    shape_molfile1 = os.path.join(output_dir, f"{name}_shape1.mol")
    shape_molfile2 = os.path.join(output_dir, f"{name}_shape2.mol")

    if not os.path.exists(output_file):
        logging.info(f"analysing {name}")

        energy_decomp = {}
        for component in conformer.energy_decomposition:
            component_tup = conformer.energy_decomposition[component]
            if component == "total energy":
                energy_decomp[f"{component}_{component_tup[1]}"] = float(
                    component_tup[0]
                )
            else:
                just_name = component.split("'")[1]
                key = f"{just_name}_{component_tup[1]}"
                value = float(component_tup[0])
                if key in energy_decomp:
                    energy_decomp[key] += value
                else:
                    energy_decomp[key] = value
        fin_energy = energy_decomp["total energy_kJ/mol"]
        try:
            assert (
                sum(
                    energy_decomp[i]
                    for i in energy_decomp
                    if "total energy" not in i
                )
                == fin_energy
            )
        except AssertionError:
            raise AssertionError(
                "energy decompisition does not sum to total energy for"
                f" {name}: {energy_decomp}"
            )

        n_shape_mol = get_shape_molecule_nodes(
            constructed_molecule=conformer.molecule,
            name=name,
            element=node_element,
            topo_expected=node_expected_topologies(),
        )
        l_shape_mol = get_shape_molecule_ligands(
            constructed_molecule=conformer.molecule,
            name=name,
            element=ligand_element,
            topo_expected=ligand_expected_topologies(),
        )
        if n_shape_mol is None:
            node_shape_measures = None
        else:
            n_shape_mol.write(shape_molfile1)
            node_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_nshape"),
                shape_path=shape_path(),
                shape_string=None,
            ).calculate(n_shape_mol)

        if l_shape_mol is None:
            lig_shape_measures = None
        else:
            lig_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_lshape"),
                shape_path=shape_path(),
                shape_string=None,
            ).calculate(l_shape_mol)
            l_shape_mol.write(shape_molfile2)

        opt_pore_data = PoreMeasure().calculate_min_distance(
            conformer.molecule
        )

        # Always want to extract target torions if present.
        g_measure = GeomMeasure(
            target_torsions=(
                TargetTorsion(
                    search_string=("b", "a", "c", "a", "b"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0=openmm.unit.Quantity(
                        value=180, unit=openmm.unit.degrees
                    ),
                    torsion_k=openmm.unit.Quantity(
                        value=0,
                        unit=openmm.unit.kilojoules_per_mole,
                    ),
                    torsion_n=1,
                ),
            )
        )
        bond_data = g_measure.calculate_bonds(conformer.molecule)
        angle_data = g_measure.calculate_angles(conformer.molecule)
        dihedral_data = g_measure.calculate_torsions(
            molecule=conformer.molecule,
            absolute=True,
        )
        min_b2b_distance = g_measure.calculate_minb2b(conformer.molecule)
        radius_gyration = g_measure.calculate_radius_gyration(
            molecule=conformer.molecule,
        )
        max_diameter = g_measure.calculate_max_diameter(conformer.molecule)
        if radius_gyration > max_diameter:
            raise ValueError(
                f"{name} Rg ({radius_gyration}) > maxD ({max_diameter})"
            )

        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = force_field.get_targets()
        if "6P8" in name:
            torsions = "toff"
        else:
            torsions = (
                "ton"
                if ff_targets["torsions"][0].torsion_k.value_in_unit(
                    openmm.unit.kilojoules_per_mole
                )
                > 0
                else "toff"
            )

        c2r0 = None
        c3r0 = None
        clr0 = None
        for bt in ff_targets["bonds"]:
            cp = (bt.class1, bt.class2)
            if "6P8" in name:
                if ("b", "n") in (cp, tuple(reversed(cp))):
                    c3r0 = bt.bond_r.value_in_unit(openmm.unit.angstrom)
                if ("b", "m") in (cp, tuple(reversed(cp))):
                    clr0 = bt.bond_r.value_in_unit(openmm.unit.angstrom)
            else:
                if ("a", "c") in (cp, tuple(reversed(cp))):
                    c2r0 = bt.bond_r.value_in_unit(openmm.unit.angstrom)
                if ("b", "n") in (cp, tuple(reversed(cp))):
                    clr0 = bt.bond_r.value_in_unit(openmm.unit.angstrom)
                elif ("b", "m") in (cp, tuple(reversed(cp))):
                    clr0 = bt.bond_r.value_in_unit(openmm.unit.angstrom)

        c2angle = None
        c3angle = None
        clangle = None
        for at in ff_targets["angles"]:
            cp = (at.class1, at.class2, at.class3)
            if "6P8" in name:
                if ("b", "n", "b") in (cp, tuple(reversed(cp))):
                    c3angle = at.angle.value_in_unit(openmm.unit.degrees)
                if ("b", "m", "b") in (cp, tuple(reversed(cp))):
                    clangle = at.angle.value_in_unit(openmm.unit.degrees)
            else:
                if ("b", "a", "c") in (cp, tuple(reversed(cp))):
                    c2angle = at.angle.value_in_unit(openmm.unit.degrees)
                if ("b", "n", "b") in (cp, tuple(reversed(cp))):
                    clangle = at.angle.value_in_unit(openmm.unit.degrees)
                elif ("b", "m", "b") in (cp, tuple(reversed(cp))):
                    clangle = at.angle.value_in_unit(openmm.unit.degrees)

        force_field_dict = {
            "ff_id": force_field.get_identifier(),
            "torsions": torsions,
            "vdws": "von",
            "clbb_bead1": "",
            "clbb_bead2": "",
            "c2bb_bead1": "",
            "c2bb_bead2": "",
            "c2r0": c2r0,
            "c2angle": c2angle,
            "c3r0": c3r0,
            "c3angle": c3angle,
            "clr0": clr0,
            "clangle": clangle,
        }
        res_dict = {
            "optimised": True,
            "source": conformer.source,
            "fin_energy_kjmol": fin_energy,
            "fin_energy_decomp": energy_decomp,
            "opt_pore_data": opt_pore_data,
            "lig_shape_measures": lig_shape_measures,
            "node_shape_measures": node_shape_measures,
            "bond_data": bond_data,
            "angle_data": angle_data,
            "dihedral_data": dihedral_data,
            "min_b2b_distance": min_b2b_distance,
            "radius_gyration": radius_gyration,
            "max_diameter": max_diameter,
            "force_field_dict": force_field_dict,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)


def angle_str(num=None, unit=True):
    if unit is True:
        un = r" [$^\circ$]"
    else:
        un = ""

    if num is None:
        return f"$x$-topic angle{un}"
    else:
        if num == 3:
            return f"tritopic angle{un}"
        elif num == 4:
            return f"tetratopic angle{un}"
        elif num == 2:
            return f"ditopic angle{un}"


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
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return cosine_similarity


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
        (t_str, clbb_name, c2bb_name, ff_name) = name.split("_")

        row["cage_name"] = f"{t_str}_{clbb_name}_{c2bb_name}_{ff_name}"
        row["clbb_name"] = clbb_name
        row["c2bb_name"] = c2bb_name
        row["topology"] = t_str
        row["ff_name"] = ff_name
        row["torsions"] = res_dict["force_field_dict"]["torsions"]
        row["vdws"] = res_dict["force_field_dict"]["vdws"]
        row["run_number"] = 0

        row["cltopo"] = int(clbb_name[0])
        if t_str in cage_topology_options(
            "2p3"
        ) or t_str in cage_topology_options("2p4"):
            cltitle = "3C1" if row["cltopo"] == 3 else "4C1"
            row["c2r0"] = res_dict["force_field_dict"]["c2r0"]
            row["c2angle"] = res_dict["force_field_dict"]["c2angle"]
            row["target_bite_angle"] = (row["c2angle"] - 90) * 2

        elif t_str in cage_topology_options("3p4"):
            cltitle = "4C1"
            row["c3r0"] = res_dict["force_field_dict"]["c3r0"]
            row["c3angle"] = res_dict["force_field_dict"]["c3angle"]

        row["cltitle"] = cltitle
        row["clr0"] = res_dict["force_field_dict"]["clr0"]
        row["clangle"] = res_dict["force_field_dict"]["clangle"]

        row["bbpair"] = clbb_name + c2bb_name + ff_name
        row["optimised"] = res_dict["optimised"]
        row["source"] = res_dict["source"]

        if row["optimised"]:
            row["strain_energy"] = res_dict["fin_energy_kjmol"]
            row["energy_per_bb"] = res_dict["fin_energy_kjmol"] / stoich_map(
                t_str
            )
            for force_title in res_dict["fin_energy_decomp"]:
                if force_title in (
                    "CMMotionRemover_kJ/mol",
                    "tot_energy_kjmol",
                    "total energy_kJ/mol",
                ):
                    continue
                row[force_title] = res_dict["fin_energy_decomp"][force_title]

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
