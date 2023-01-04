#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to optimise CG models of fourplussix systems.

Author: Andrew Tarzia

"""

import sys
import stk
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import os
import json
from rdkit.Chem import AllChem as rdkit
from collections import defaultdict
import logging
import numpy as np


from env_set import unsymm
from utilities import check_directory, angle_between

from gulp_optimizer import (
    CGGulpOptimizer,
    lorentz_berthelot_sigma_mixing,
)

from cage_construction.topologies import (
    CGM12L24,
    unsymm_topology_options,
)

from beads import CgBead, bead_library_check

from precursor_db.topologies import UnsymmLigand, UnsymmBiteLigand

from ea_module import (
    RandomVA,
    VaGeneticRecombination,
    CgEvolutionaryAlgorithm,
    RecordFitnessFunction,
    VaKeyMaker,
    VaRoulette,
    VaBest,
)
from ea_plotters import (
    CgProgressPlotter,
    plot_existing_M2_data_distributions,
    plot_existing_M6_data_distributions,
    plot_existing_M12_data_distributions,
)


def lig_2c_beads():
    return (
        # CgBead("Ag", sigma=0.5, angle_centered=20),
        # CgBead("Al", sigma=1.0, angle_centered=20),
        # CgBead("Ac", sigma=2.0, angle_centered=20),
        # CgBead("Am", sigma=3.0, angle_centered=20),
        # CgBead("As", sigma=4.0, angle_centered=20),
        # CgBead("Au", sigma=5.0, angle_centered=20),
        CgBead("Ba", sigma=0.5, angle_centered=40),
        CgBead("Bi", sigma=1.0, angle_centered=40),
        CgBead("B", sigma=2.0, angle_centered=40),
        CgBead("Cd", sigma=3.0, angle_centered=40),
        CgBead("Ce", sigma=4.0, angle_centered=40),
        CgBead("Cf", sigma=5.0, angle_centered=40),
        CgBead("Cm", sigma=0.5, angle_centered=60),
        CgBead("Co", sigma=1.0, angle_centered=60),
        CgBead("Cu", sigma=2.0, angle_centered=60),
        CgBead("Cr", sigma=3.0, angle_centered=60),
        CgBead("Eu", sigma=4.0, angle_centered=60),
        CgBead("Er", sigma=5.0, angle_centered=60),
        CgBead("Fe", sigma=0.5, angle_centered=90),
        CgBead("Ga", sigma=1.0, angle_centered=90),
        CgBead("Gd", sigma=2.0, angle_centered=90),
        CgBead("Ge", sigma=3.0, angle_centered=90),
        CgBead("Hf", sigma=4.0, angle_centered=90),
        CgBead("He", sigma=5.0, angle_centered=90),
        CgBead("Hg", sigma=0.5, angle_centered=100),
        CgBead("Ho", sigma=1.0, angle_centered=100),
        CgBead("In", sigma=2.0, angle_centered=100),
        CgBead("I", sigma=3.0, angle_centered=100),
        CgBead("Ir", sigma=4.0, angle_centered=100),
        CgBead("La", sigma=5.0, angle_centered=100),
        CgBead("Lr", sigma=0.5, angle_centered=120),
        CgBead("Lu", sigma=1.0, angle_centered=120),
        CgBead("Mn", sigma=2.0, angle_centered=120),
        CgBead("Mo", sigma=3.0, angle_centered=120),
        CgBead("Nd", sigma=4.0, angle_centered=120),
        CgBead("Ne", sigma=5.0, angle_centered=120),
        CgBead("Md", sigma=0.5, angle_centered=130),
        CgBead("Nb", sigma=1.0, angle_centered=130),
        CgBead("Ni", sigma=2.0, angle_centered=130),
        CgBead("No", sigma=3.0, angle_centered=130),
        CgBead("Np", sigma=4.0, angle_centered=130),
        CgBead("Os", sigma=5.0, angle_centered=130),
        CgBead("Pa", sigma=0.5, angle_centered=150),
        CgBead("Pd", sigma=1.0, angle_centered=150),
        CgBead("Po", sigma=2.0, angle_centered=150),
        CgBead("Pr", sigma=3.0, angle_centered=150),
        CgBead("Pu", sigma=4.0, angle_centered=150),
        CgBead("P", sigma=5.0, angle_centered=150),
        CgBead("Re", sigma=0.5, angle_centered=180),
        CgBead("Rh", sigma=1.0, angle_centered=180),
        CgBead("Ru", sigma=2.0, angle_centered=180),
        CgBead("Se", sigma=3.0, angle_centered=180),
        CgBead("Si", sigma=4.0, angle_centered=180),
        CgBead("Sm", sigma=5.0, angle_centered=180),
    )


def binder_beads():
    return (CgBead("C", sigma=2.0, angle_centered=180),)


def beads_4c():
    return (CgBead("Pt", sigma=2.0, angle_centered=(90, 180, 130)),)


def get_initial_population(
    cage_topology_function,
    twoc_precursor,
    fourc_precursor,
    num_population,
    generator,
):

    va_dists = defaultdict(int)
    for i in range(num_population):
        selected_orderings = generator.randint(2, size=24)
        selected_va = {
            i: j for i, j in zip(range(12, 36), selected_orderings)
        }
        va_count = sum(selected_va.values())
        va_dists[va_count] += 1
        yield stk.MoleculeRecord(
            topology_graph=cage_topology_function(
                building_blocks=(twoc_precursor, fourc_precursor),
                vertex_alignments=selected_va,
            ),
        )
    logging.info(f"va dist count: {va_dists}")


def get_va_string(va_dict):
    return "".join(str(i) for i in va_dict.values())


def get_molecule_name_from_record(record):
    tg = record.get_topology_graph()
    bb4_str = tuple(
        tuple(
            i
            for i in tg.get_building_blocks()
            if i.get_num_functional_groups() == 4
        )[0].get_atoms()
    )[1].__class__.__name__
    bb2_atoms = tuple(
        tuple(
            i
            for i in tg.get_building_blocks()
            if i.get_num_functional_groups() == 2
        )[0].get_atoms()
    )
    bb2_str = "".join(
        i.__class__.__name__
        for i in (bb2_atoms[1], bb2_atoms[0], bb2_atoms[2])
    )
    va_str = get_va_string(tg.get_vertex_alignments())
    return f"{tg.__class__.__name__}_{bb4_str}_{bb2_str}_{va_str}"


def get_centre_dists(molecule, tg, run_prefix):

    num_nodes = {
        "CGM12L24": 12,
    }
    pos_mat = molecule.get_position_matrix()

    bb2_atoms = tuple(
        tuple(
            i
            for i in tg.get_building_blocks()
            if i.get_num_functional_groups() == 2
        )[0].get_atoms()
    )

    tg_str, cent_name, _, _ = run_prefix.split("_")

    query = (
        f"[{bb2_atoms[2].__class__.__name__}]~"
        f"[{cent_name}]"
        f"(~[{bb2_atoms[2].__class__.__name__}])"
        f"(~[{bb2_atoms[0].__class__.__name__}])"
        f"~[{bb2_atoms[0].__class__.__name__}]"
    )
    rdkit_mol = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_mol)
    num_matches = 0
    num_cis = 0
    num_trans = 0
    for match in rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(query),
    ):
        num_matches += 1

        # Get angle between same atom types, defined in smarts.
        vector1 = pos_mat[match[1]] - pos_mat[match[0]]
        vector2 = pos_mat[match[1]] - pos_mat[match[2]]
        curr_angle = np.degrees(angle_between(vector1, vector2))

        # Set cis or trans based on this angle.
        if curr_angle < 130:
            num_cis += 1
        elif curr_angle > 130:
            num_trans += 1

    return {
        "num_notcisortrans": num_nodes[tg_str] - num_matches,
        "num_trans": num_trans,
        "num_cis": num_cis,
    }


def get_results_dictionary(molecule_record):
    output_dir = unsymm() / "calculations"

    molecule = molecule_record.get_molecule()
    run_prefix = get_molecule_name_from_record(molecule_record)

    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")
    opt_xyz_file = os.path.join(output_dir, f"{run_prefix}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{run_prefix}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{run_prefix}_opted2.mol")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        # Does optimisation.
        if os.path.exists(opt1_mol_file):
            molecule = molecule.with_structure_from_file(opt1_mol_file)
        else:
            logging.info(f"optimising {run_prefix}")
            opt = CGGulpOptimizer(
                fileprefix=run_prefix,
                output_dir=output_dir,
                param_pool=beads_2c() + beads_4c(),
                max_cycles=1000,
                conjugate_gradient=True,
                bonds=True,
                angles=True,
                torsions=False,
                vdw=False,
            )
            run_data = opt.optimize(molecule)
            molecule = molecule.with_structure_from_file(opt_xyz_file)
            molecule.write(opt1_mol_file)

        opt = CGGulpOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=beads_2c() + beads_4c(),
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        if os.path.exists(opt2_mol_file):
            molecule = molecule.with_structure_from_file(opt2_mol_file)
            run_data = opt.extract_gulp()
        else:
            run_data = opt.optimize(molecule)
            molecule = molecule.with_structure_from_file(opt_xyz_file)
            molecule.write(opt2_mol_file)

        fin_energy = run_data["final_energy"]
        max_size = molecule.get_maximum_diameter()
        centre_dists = get_centre_dists(
            molecule=molecule,
            tg=molecule_record.get_topology_graph(),
            run_prefix=run_prefix,
        )
        res_dict = {
            "fin_energy": fin_energy,
            "max_size": max_size,
            "centre_dists": centre_dists,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    return res_dict


def get_final_energy(molecule_record):
    res_dict = get_results_dictionary(molecule_record)
    return res_dict["fin_energy"]


def get_num_trans(molecule_record):
    res_dict = get_results_dictionary(molecule_record)
    return res_dict["num_trans"]


def get_num_cis(molecule_record):
    res_dict = get_results_dictionary(molecule_record)
    return res_dict["num_cis"]


def get_num_notcisortrans(molecule_record):
    res_dict = get_results_dictionary(molecule_record)
    return res_dict["num_notcisortrans"]


def fitness_from_dictionary(res_dict):
    # size = res_dict["max_size"]
    # target_size = 43.2
    # size_score = abs(target_size - size)
    # score = 1 / (res_dict["fin_energy"] * 10 + size_score)

    target_notcisortrans = 0
    cdists = res_dict["centre_dists"]

    score = 1 / (
        res_dict["fin_energy"] * 1
        + abs(cdists["num_notcisortrans"] - target_notcisortrans) * 100
    )
    return score


def get_fitness_value(molecule_record):
    res_dict = get_results_dictionary(molecule_record)
    return fitness_from_dictionary(res_dict)


def mutator(generator, topology_options):
    return stk.RandomMutator(
        mutators=(
            # Substitutes a 2c CGBead with another.
            RandomVA(
                topology_options=topology_options,
                random_seed=generator.randint(0, 1000),
            ),
        ),
        random_seed=generator.randint(0, 1000),
    )


def crosser(generator, topology_options):
    return VaGeneticRecombination(
        get_gene=get_va_string,
        topology_options=topology_options,
    )


def get_fourc(element_string):
    four_c_bb = stk.BuildingBlock(
        smiles=f"[Br][{element_string}]([Br])([Br])[Br]",
        position_matrix=[
            [-2, 0, 0],
            [0, 0, 0],
            [0, -2, 0],
            [2, 0, 0],
            [0, 2, 0],
        ],
    )

    new_fgs = stk.SmartsFunctionalGroupFactory(
        smarts=(f"[{element_string}]" f"[Br]"),
        bonders=(0,),
        deleters=(1,),
        placers=(0, 1),
    )
    return stk.BuildingBlock.init_from_molecule(
        molecule=four_c_bb,
        functional_groups=(new_fgs,),
    )


def build_experimentals(
    cage_topology_function,
    twoc_precursor,
    fourc_precursor,
):

    orderings = {
        "E1": {
            # Right.
            12: 0,
            13: 0,
            14: 1,
            15: 1,
            # Left.
            16: 1,
            17: 1,
            18: 0,
            19: 0,
            # Top.
            20: 0,
            21: 0,
            22: 1,
            23: 1,
            # Bottom.
            24: 1,
            25: 1,
            26: 0,
            27: 0,
            # Front.
            28: 0,
            29: 1,
            30: 0,
            31: 1,
            # Back.
            32: 0,
            33: 1,
            34: 0,
            35: 1,
        },
        "E2": {
            # Right.
            12: 0,
            13: 1,
            14: 0,
            15: 1,
            # Left.
            16: 1,
            17: 0,
            18: 1,
            19: 0,
            # Top.
            20: 0,
            21: 0,
            22: 1,
            23: 1,
            # Bottom.
            24: 1,
            25: 1,
            26: 0,
            27: 0,
            # Front.
            28: 0,
            29: 1,
            30: 0,
            31: 0,
            # Back.
            32: 0,
            33: 1,
            34: 1,
            35: 1,
        },
        "E3": {
            # Right.
            12: 1,
            13: 0,
            14: 1,
            15: 0,
            # Left.
            16: 0,
            17: 1,
            18: 0,
            19: 1,
            # Top.
            20: 0,
            21: 1,
            22: 0,
            23: 1,
            # Bottom.
            24: 1,
            25: 0,
            26: 1,
            27: 0,
            # Front.
            28: 0,
            29: 0,
            30: 1,
            31: 1,
            # Back.
            32: 1,
            33: 1,
            34: 0,
            35: 0,
        },
        "G1": {
            # Right.
            12: 0,
            13: 0,
            14: 1,
            15: 1,
            # Left.
            16: 1,
            17: 1,
            18: 0,
            19: 0,
            # Top.
            20: 0,
            21: 0,
            22: 1,
            23: 1,
            # Bottom.
            24: 1,
            25: 1,
            26: 0,
            27: 0,
            # Front.
            28: 1,
            29: 0,
            30: 1,
            31: 0,
            # Back.
            32: 1,
            33: 0,
            34: 1,
            35: 0,
        },
    }

    logging.info("top fitness values:")
    target_orderings = {}
    for name in orderings:
        molrec = stk.MoleculeRecord(
            topology_graph=cage_topology_function(
                building_blocks=(twoc_precursor, fourc_precursor),
                vertex_alignments=orderings[name],
            ),
        )
        target_orderings[get_molecule_name_from_record(molrec)] = name
        fv = get_fitness_value(molrec)
        logging.info(f"for {name}, fitness: {fv}")


def optimise_ligand(molecule, name, output_dir, full_bead_library):

    opt_xyz_file = os.path.join(output_dir, f"{name}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")

    # Does optimisation.
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}")
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=2000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)
        os.system(f"rm {opt_xyz_file}")

    return molecule


def optimise_cage(molecule, name, output_dir, full_bead_library):

    opt_xyz_file = os.path.join(output_dir, f"{name}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")

    # Does optimisation.
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}")
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=2000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)
        os.system(f"rm {opt_xyz_file}")

    return molecule


def shortest_distance_to_plane(plane, point):
    """
    Calculate the perpendicular distance beween a point and a plane.

    """

    top = abs(
        plane[0] * point[0]
        + plane[1] * point[1]
        + plane[2] * point[2]
        - plane[3]
    )
    bottom = np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    distance = top / bottom
    return distance


def analyse_cage(molecule, name, output_dir, full_bead_library):

    output_file = os.path.join(output_dir, f"{name}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )

        ini_ids = []
        for i in molecule.get_atoms():
            if i.__class__.__name__ == "C":
                ini_ids.append(i.get_id())

            if i.__class__.__name__ == "Pt":
                ini_ids.append(i.get_id())

        centroid = molecule.get_centroid(atom_ids=ini_ids)
        normal = molecule.get_plane_normal(atom_ids=ini_ids)
        # Plane of equation ax + by + cz = d.
        binder_atom_plane = np.append(normal, np.sum(normal * centroid))
        # Define the plane deviation as the sum of the distance of all
        # atoms from the plane defined by all atoms.
        planarity_CPd = sum(
            [
                shortest_distance_to_plane(
                    binder_atom_plane,
                    tuple(
                        molecule.get_atomic_positions(atom_ids=i),
                    )[0],
                )
                for i in ini_ids
            ]
        )

        run_data = opt.extract_gulp()
        fin_energy = run_data["final_energy"]
        res_dict = {
            "fin_energy": fin_energy,
            "planarity_CPd": planarity_CPd,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    return res_dict


def analyse_big_cage(molecule, name, output_dir, full_bead_library):

    output_file = os.path.join(output_dir, f"{name}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )

        run_data = opt.extract_gulp()
        fin_energy = run_data["final_energy"]
        res_dict = {
            "fin_energy": fin_energy,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    return res_dict


def get_CGBead_from_string(string, bead_library):
    return tuple(i for i in bead_library if i.element_string == string)[
        0
    ]


def get_present_beads(c2_bbname):
    wtopo = c2_bbname[2:]
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

    if len(present_beads_names) != 4:
        raise ValueError(f"{present_beads_names} length != {c2_bbname}")
    return present_beads_names


def phase_space_1(
    cage_set_data,
    figure_output,
    bead_library,
    ligand_data,
):
    fig, axs = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(16, 5),
    )

    vamap = {
        "0000": "#1f77b4",
        "1000": "#ff7f0e",
        "1100": "#2ca02c",
        "1010": "#d62728",
    }
    convert_vstrname = {
        "0000": "A",
        "1000": "B",
        "1100": "C",
        "1010": "D",
    }

    isomer_energy = 5
    planarity_limit = 2
    max_energy = 20
    all_data = []
    for cs in cage_set_data:
        csd = cage_set_data[cs]
        present_beads_names = get_present_beads(cs)

        theta1 = get_CGBead_from_string(
            present_beads_names[1], bead_library
        ).angle_centered

        theta2 = get_CGBead_from_string(
            present_beads_names[2], bead_library
        ).angle_centered

        min_energy = 1e24
        preferred_vastr = None
        for i in csd:
            energy = csd[i]["fin_energy"]
            if energy < min_energy:
                min_energy = energy
                preferred_vastr = i

        if min_energy > isomer_energy:
            colour = "white"
        else:
            planarity = csd[preferred_vastr]["planarity_CPd"]
            if planarity < planarity_limit:
                colour = "k"
            else:
                colour = vamap[preferred_vastr]

        cis_energy = csd["1100"]["fin_energy"]
        other_energies = [
            csd[i]["fin_energy"] - cis_energy
            for i in csd
            if i != "1100"
        ]
        cis_preference = min(other_energies)

        if cis_preference >= isomer_energy:
            planarity = csd["1100"]["planarity_CPd"]
            if planarity >= planarity_limit:
                all_data.append((theta1, theta2, cis_preference))

        axs[0].scatter(
            theta1,
            theta2,
            c=colour,
            edgecolor="k",
            s=50,
            marker="s",
        )

    axs[1].scatter(
        [i[0] for i in all_data],
        [i[1] for i in all_data],
        c=[i[2] for i in all_data],
        vmin=isomer_energy,
        vmax=max_energy,
        alpha=1.0,
        edgecolor="none",
        marker="s",
        s=50,
        cmap="Blues",
    )

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=isomer_energy, vmax=max_energy)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("cis preference (eV)", fontsize=16)

    for i in vamap:
        axs[0].scatter(
            None,
            None,
            c=vamap[i],
            edgecolor="k",
            s=50,
            marker="s",
            alpha=1.0,
            label=convert_vstrname[i],
        )
    axs[0].scatter(
        None,
        None,
        c="white",
        edgecolor="k",
        s=50,
        marker="s",
        label="unstable",
    )
    axs[0].scatter(
        None,
        None,
        c="k",
        edgecolor="k",
        s=50,
        marker="s",
        label="flat",
    )

    lmarker = {1: "X", 2: "P"}
    lcolour = {0: "gold", 1: "r"}
    for i in ligand_data:
        ldata = ligand_data[i]
        x = min((ldata["angle1"], ldata["angle2"]))
        y = max((ldata["angle1"], ldata["angle2"]))
        c = lcolour[ldata["result"][1]]
        m = lmarker[ldata["result"][0]]
        axs[0].scatter(
            x,
            y,
            c=c,
            marker=m,
            s=50,
        )
        axs[1].scatter(
            x,
            y,
            c=c,
            marker=m,
            s=50,
        )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("theta1", fontsize=16)
    axs[0].set_ylabel("theta2", fontsize=16)
    axs[0].set_xlim(0, 180)
    axs[0].set_ylim(0, 180)
    axs[0].set_title(f"{isomer_energy}eV", fontsize=16)
    axs[0].legend(fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("theta1", fontsize=16)
    axs[1].set_ylabel("theta2", fontsize=16)
    axs[1].set_xlim(0, 180)
    axs[1].set_ylim(0, 180)
    axs[1].set_title(f"{isomer_energy} to {max_energy}eV", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_2(
    cage_set_data,
    figure_output,
    bead_library,
    ligand_data,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    vamap = {
        "0000": "#1f77b4",
        "1000": "#ff7f0e",
        "1100": "#2ca02c",
        "1010": "#d62728",
    }
    convert_vstrname = {
        "0000": "A",
        "1000": "B",
        "1100": "C",
        "1010": "D",
    }

    isomer_energy = 5
    planarity_limit = 2
    for cs in cage_set_data:
        csd = cage_set_data[cs]
        present_beads_names = get_present_beads(cs)
        all_sigma = [
            get_CGBead_from_string(i, bead_library).sigma
            for i in present_beads_names
        ]

        bl1 = lorentz_berthelot_sigma_mixing(all_sigma[1], all_sigma[3])
        bl2 = lorentz_berthelot_sigma_mixing(all_sigma[2], all_sigma[3])

        angle = get_CGBead_from_string(
            present_beads_names[2], bead_library
        ).angle_centered

        if angle == 90:
            marker = "o"
            geom_mismatch = bl1 - bl2
        else:
            marker = "s"
            hyp = lorentz_berthelot_sigma_mixing(
                all_sigma[2], all_sigma[0]
            ) + lorentz_berthelot_sigma_mixing(
                all_sigma[1], all_sigma[0]
            )
            extension = np.cos(np.radians(angle)) * hyp
            mod_length = extension + bl1
            geom_mismatch = mod_length - (bl2)

        min_energy = 1e24
        preferred_vastr = None
        for i in csd:
            energy = csd[i]["fin_energy"]
            if energy < min_energy:
                min_energy = energy
                preferred_vastr = i

        if min_energy > isomer_energy:
            colour = "white"
        else:
            planarity = csd[preferred_vastr]["planarity_CPd"]
            if planarity < planarity_limit:
                colour = "k"
            else:
                colour = vamap[preferred_vastr]

        cis_energy = csd["1100"]["fin_energy"]
        other_energies = [
            csd[i]["fin_energy"] - cis_energy
            for i in csd
            if i != "1100"
        ]
        cis_preference = min(other_energies)

        ax.scatter(
            geom_mismatch,
            cis_preference,
            c=colour,
            edgecolor="k",
            s=50,
            marker=marker,
        )

    for i in vamap:
        ax.scatter(
            None,
            None,
            c=vamap[i],
            edgecolor="k",
            s=50,
            marker="s",
            alpha=1.0,
            label=convert_vstrname[i],
        )
    ax.scatter(
        None,
        None,
        c="white",
        edgecolor="k",
        s=50,
        marker="s",
        label="unstable",
    )
    ax.scatter(
        None,
        None,
        c="k",
        edgecolor="k",
        s=50,
        marker="s",
        label="flat",
    )

    lcolour = {0: "gold", 1: "r"}
    for i in ligand_data:
        ldata = ligand_data[i]
        x = ldata["extension"]
        c = lcolour[ldata["result"][1]]
        ax.axvline(
            x,
            c=c,
            lw=2,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("geom. mismatch", fontsize=16)
    ax.set_ylabel("cis preference", fontsize=16)
    ax.set_title(f"{isomer_energy}eV", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_3(
    cage_set_data,
    figure_output,
    bead_library,
    ligand_data,
):
    fig, axs = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(16, 5),
    )

    vamap = {
        "M6A1": "#1f77b4",
        "M6A2": "#9BC53D",
        "M6def": "#C3423F",
        "M12def": "#B279A7",
        "M12E1": "#ff7f0e",
        "M12E2": "#2ca02c",
        "M12E3": "#d62728",
        "M12G1": "#6969B3",
    }

    isomer_energy = 2
    s = 120
    all_data = []
    for cs in cage_set_data:
        csd = cage_set_data[cs]
        present_beads_names = get_present_beads(cs)

        min_energy = 1e24
        preferred_tstr = None
        for i in csd:
            energy = csd[i]["fin_energy"]
            if energy < min_energy:
                min_energy = energy
                preferred_tstr = i

        all_sigma = [
            get_CGBead_from_string(i, bead_library).sigma
            for i in present_beads_names
        ]
        distance_1 = lorentz_berthelot_sigma_mixing(
            all_sigma[2], all_sigma[3]
        )

        distance_2 = (
            lorentz_berthelot_sigma_mixing(all_sigma[1], all_sigma[3])
            + lorentz_berthelot_sigma_mixing(all_sigma[1], all_sigma[0])
            + lorentz_berthelot_sigma_mixing(all_sigma[2], all_sigma[0])
        )

        distance_ratio = distance_2 / distance_1

        theta = get_CGBead_from_string(
            present_beads_names[2], bead_library
        ).angle_centered

        if min_energy > isomer_energy:
            colour = "white"
        else:
            colour = vamap[preferred_tstr]
        all_data.append((distance_ratio, theta, min_energy))

        axs[0].scatter(
            distance_ratio,
            theta,
            c=colour,
            edgecolor="k",
            s=s,
            marker="s",
        )

    axs[1].scatter(
        [i[0] for i in all_data],
        [i[1] for i in all_data],
        c=[i[2] for i in all_data],
        vmin=0,
        vmax=isomer_energy,
        alpha=1.0,
        edgecolor="k",
        marker="s",
        s=s,
        cmap="Blues_r",
    )

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=isomer_energy)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("energy (eV)", fontsize=16)

    for i in vamap:
        axs[0].scatter(
            None,
            None,
            c=vamap[i],
            edgecolor="k",
            s=s,
            marker="s",
            alpha=1.0,
            label=i,
        )
    axs[0].scatter(
        None,
        None,
        c="white",
        edgecolor="k",
        s=s,
        marker="s",
        label=f"mixed ({isomer_energy}eV)",
    )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("distance ratio", fontsize=16)
    axs[0].set_ylabel("theta", fontsize=16)
    axs[0].set_title(f"{isomer_energy}eV", fontsize=16)
    axs[0].legend(fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("distance ratio", fontsize=16)
    axs[1].set_ylabel("theta", fontsize=16)
    axs[1].set_title(f"{isomer_energy}eV", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def unsymm_1(ligand_data):
    struct_output = unsymm() / "structures"
    check_directory(struct_output)
    figure_output = unsymm() / "figures"
    check_directory(figure_output)
    calculation_output = unsymm() / "calculations"
    check_directory(calculation_output)
    ligand_output = unsymm() / "ligands"
    check_directory(ligand_output)

    # M2L4 problem, modelling theta, theta phase space.
    m2l4_2_beads = (
        CgBead("Ba", sigma=2, angle_centered=20),
        CgBead("Bi", sigma=2, angle_centered=25),
        CgBead("B", sigma=2, angle_centered=30),
        CgBead("Cd", sigma=2, angle_centered=35),
        CgBead("Ce", sigma=2, angle_centered=40),
        CgBead("Cf", sigma=2, angle_centered=45),
        CgBead("Cm", sigma=2, angle_centered=50),
        CgBead("Co", sigma=2, angle_centered=55),
        CgBead("Cu", sigma=2, angle_centered=60),
        CgBead("Cr", sigma=2, angle_centered=65),
        CgBead("Eu", sigma=2, angle_centered=70),
        CgBead("Er", sigma=2, angle_centered=75),
        CgBead("Fe", sigma=2, angle_centered=80),
        CgBead("Ga", sigma=2, angle_centered=85),
        CgBead("Gd", sigma=2, angle_centered=90),
        CgBead("Ge", sigma=2, angle_centered=95),
        CgBead("Hf", sigma=2, angle_centered=100),
        CgBead("He", sigma=2, angle_centered=105),
        CgBead("Hg", sigma=2, angle_centered=110),
        CgBead("Ho", sigma=2, angle_centered=115),
        CgBead("In", sigma=2, angle_centered=120),
        CgBead("I", sigma=2, angle_centered=125),
        CgBead("Ir", sigma=2, angle_centered=130),
        CgBead("La", sigma=2, angle_centered=135),
        CgBead("Lr", sigma=2, angle_centered=140),
        CgBead("Lu", sigma=2, angle_centered=145),
        CgBead("Md", sigma=2, angle_centered=150),
        CgBead("Mn", sigma=2, angle_centered=155),
        CgBead("Mo", sigma=2, angle_centered=160),
        CgBead("Nb", sigma=2, angle_centered=165),
        CgBead("Nd", sigma=2, angle_centered=170),
    )
    m2l4_4_beads = (
        CgBead("Pt", sigma=2.0, angle_centered=(90, 180, 130)),
    )
    m2l4_c_beads = (CgBead("C", sigma=6.0, angle_centered=180),)
    m2l4_b_beads = (CgBead("N", sigma=2.0, angle_centered=180),)
    full_bead_library = (
        m2l4_2_beads + m2l4_4_beads + m2l4_c_beads + m2l4_b_beads
    )
    bead_library_check(full_bead_library)

    c2_blocks = {}
    for c2_options in itertools.combinations(m2l4_2_beads, r=2):
        if c2_options[0].element_string == c2_options[1].element_string:
            continue

        temp = UnsymmLigand(
            centre_bead=m2l4_c_beads[0],
            lhs_bead=c2_options[0],
            rhs_bead=c2_options[1],
            binder_bead=m2l4_b_beads[0],
        )

        opt_bb = optimise_ligand(
            molecule=temp.get_building_block(),
            name=temp.get_name(),
            output_dir=calculation_output,
            full_bead_library=full_bead_library,
        )
        opt_bb.write(str(ligand_output / f"{temp.get_name()}_optl.mol"))
        c2_blocks[temp.get_name()] = opt_bb

    logging.info(f"there are {len(c2_blocks)} building blocks.")
    fourc_precursor = get_fourc(m2l4_4_beads[0].element_string)

    cage_set_data = {}
    for c2blk in c2_blocks:
        cage_set_data[c2blk] = {}
        for vastr in ("0000", "1000", "1100", "1010"):
            va_dict = {i: int(j) for i, j in zip(range(2, 6), vastr)}
            cage_topo_str = "m2"
            name = f"{cage_topo_str}_{c2blk}_{vastr}"
            cage = stk.ConstructedMolecule(
                topology_graph=stk.cage.M2L4Lantern(
                    building_blocks=(
                        c2_blocks[c2blk],
                        fourc_precursor,
                    ),
                    vertex_alignments=va_dict,
                ),
            )
            cage = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )

            cage.write(str(struct_output / f"{name}_optc.mol"))
            symm_data = analyse_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )

            cage_set_data[c2blk][vastr] = symm_data

    phase_space_1(
        cage_set_data,
        figure_output,
        full_bead_library,
        ligand_data,
    )


def unsymm_2(ligand_data):
    struct_output = unsymm() / "structures"
    check_directory(struct_output)
    figure_output = unsymm() / "figures"
    check_directory(figure_output)
    calculation_output = unsymm() / "calculations"
    check_directory(calculation_output)
    ligand_output = unsymm() / "ligands"
    check_directory(ligand_output)

    # M2L4 problem, modelling theta, r phase space.
    m2l4_lhs_beads = (
        CgBead("No", sigma=2, angle_centered=120),
        CgBead("Ni", sigma=2, angle_centered=90),
    )
    m2l4_rhs_beads = (
        CgBead("Np", sigma=0.5, angle_centered=60),
        CgBead("Os", sigma=1.0, angle_centered=60),
        CgBead("Pa", sigma=1.5, angle_centered=60),
        CgBead("Pd", sigma=1.8, angle_centered=60),
        CgBead("Po", sigma=2.0, angle_centered=60),
        CgBead("Pr", sigma=2.2, angle_centered=60),
        CgBead("Pu", sigma=2.5, angle_centered=60),
        CgBead("P", sigma=3.0, angle_centered=60),
        CgBead("Re", sigma=3.5, angle_centered=60),
        CgBead("Rh", sigma=4.0, angle_centered=60),
        CgBead("Ru", sigma=0.5, angle_centered=90),
        CgBead("Se", sigma=1.0, angle_centered=90),
        CgBead("Si", sigma=1.5, angle_centered=90),
        CgBead("Sm", sigma=1.8, angle_centered=90),
        CgBead("S", sigma=2.0, angle_centered=90),
        CgBead("Ti", sigma=2.2, angle_centered=90),
        CgBead("Tm", sigma=2.5, angle_centered=90),
        CgBead("V", sigma=3.0, angle_centered=90),
        CgBead("Y", sigma=3.5, angle_centered=90),
        CgBead("Zr", sigma=4.0, angle_centered=90),
    )
    m2l4_4_beads = (
        CgBead("Pt", sigma=2.0, angle_centered=(90, 180, 130)),
    )
    m2l4_c_beads = (CgBead("C", sigma=6.0, angle_centered=180),)
    m2l4_b_beads = (CgBead("N", sigma=2.0, angle_centered=180),)
    full_bead_library = (
        m2l4_rhs_beads
        + m2l4_lhs_beads
        + m2l4_4_beads
        + m2l4_c_beads
        + m2l4_b_beads
    )
    bead_library_check(full_bead_library)

    c2_blocks = {}
    for rhs_bead in m2l4_rhs_beads:
        if rhs_bead.angle_centered == 60:
            lhs_bead = m2l4_lhs_beads[0]
        elif rhs_bead.angle_centered == 90:
            lhs_bead = m2l4_lhs_beads[1]

        temp = UnsymmLigand(
            centre_bead=m2l4_c_beads[0],
            lhs_bead=lhs_bead,
            rhs_bead=rhs_bead,
            binder_bead=m2l4_b_beads[0],
        )

        opt_bb = optimise_ligand(
            molecule=temp.get_building_block(),
            name=temp.get_name(),
            output_dir=calculation_output,
            full_bead_library=full_bead_library,
        )
        opt_bb.write(str(ligand_output / f"{temp.get_name()}_optl.mol"))
        c2_blocks[temp.get_name()] = opt_bb

    logging.info(f"there are {len(c2_blocks)} building blocks.")
    fourc_precursor = get_fourc(m2l4_4_beads[0].element_string)

    cage_set_data = {}
    for c2blk in c2_blocks:
        cage_set_data[c2blk] = {}
        for vastr in ("0000", "1000", "1100", "1010"):
            va_dict = {i: int(j) for i, j in zip(range(2, 6), vastr)}
            cage_topo_str = "m2"
            name = f"{cage_topo_str}_{c2blk}_{vastr}"
            cage = stk.ConstructedMolecule(
                topology_graph=stk.cage.M2L4Lantern(
                    building_blocks=(
                        c2_blocks[c2blk],
                        fourc_precursor,
                    ),
                    vertex_alignments=va_dict,
                ),
            )
            cage = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )

            cage.write(str(struct_output / f"{name}_optc.mol"))
            symm_data = analyse_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )

            cage_set_data[c2blk][vastr] = symm_data

    phase_space_2(
        cage_set_data,
        figure_output,
        full_bead_library,
        ligand_data,
    )


def unsymm_3(ligand_data):
    struct_output = unsymm() / "structures"
    check_directory(struct_output)
    figure_output = unsymm() / "figures"
    check_directory(figure_output)
    calculation_output = unsymm() / "calculations"
    check_directory(calculation_output)
    ligand_output = unsymm() / "ligands"
    check_directory(ligand_output)

    # Multi topo problem, higher bite angle: theta vs r1-r2 phase space.
    m2l4_lhs_beads = (CgBead("No", sigma=2, angle_centered=180),)
    m2l4_rhs_beads = (
        CgBead("Ag", sigma=0.5, angle_centered=80),
        CgBead("Al", sigma=1.0, angle_centered=80),
        CgBead("Ac", sigma=1.5, angle_centered=80),
        CgBead("Am", sigma=2.0, angle_centered=80),
        CgBead("As", sigma=2.5, angle_centered=80),
        CgBead("Au", sigma=3.0, angle_centered=80),
        CgBead("Ba", sigma=3.5, angle_centered=80),
        CgBead("Bi", sigma=4.0, angle_centered=80),
        CgBead("B", sigma=0.5, angle_centered=90),
        CgBead("Cd", sigma=1.0, angle_centered=90),
        CgBead("Ce", sigma=1.5, angle_centered=90),
        CgBead("Cf", sigma=2.0, angle_centered=90),
        CgBead("Cm", sigma=2.5, angle_centered=90),
        CgBead("Co", sigma=3.0, angle_centered=90),
        CgBead("Cu", sigma=3.5, angle_centered=90),
        CgBead("Cr", sigma=4.0, angle_centered=90),
        CgBead("Eu", sigma=0.5, angle_centered=100),
        CgBead("Er", sigma=1.0, angle_centered=100),
        CgBead("Fe", sigma=1.5, angle_centered=100),
        CgBead("Ga", sigma=2.0, angle_centered=100),
        CgBead("Gd", sigma=2.5, angle_centered=100),
        CgBead("Ge", sigma=3.0, angle_centered=100),
        CgBead("Hf", sigma=3.5, angle_centered=100),
        CgBead("He", sigma=4.0, angle_centered=100),
        CgBead("Hg", sigma=0.5, angle_centered=110),
        CgBead("Ho", sigma=1.0, angle_centered=110),
        CgBead("In", sigma=1.5, angle_centered=110),
        CgBead("I", sigma=2.0, angle_centered=110),
        CgBead("Ir", sigma=2.5, angle_centered=110),
        CgBead("La", sigma=3.0, angle_centered=110),
        CgBead("Lr", sigma=3.5, angle_centered=110),
        CgBead("Lu", sigma=4.0, angle_centered=110),
        CgBead("Md", sigma=0.5, angle_centered=120),
        CgBead("Mn", sigma=1.0, angle_centered=120),
        CgBead("Mo", sigma=1.5, angle_centered=120),
        CgBead("Nb", sigma=2.0, angle_centered=120),
        CgBead("Nd", sigma=2.5, angle_centered=120),
        CgBead("Ne", sigma=3.0, angle_centered=120),
        CgBead("Ni", sigma=3.5, angle_centered=120),
        CgBead("Np", sigma=4.0, angle_centered=120),
        CgBead("Os", sigma=0.5, angle_centered=130),
        CgBead("Pa", sigma=1.0, angle_centered=130),
        CgBead("Pd", sigma=1.5, angle_centered=130),
        CgBead("Po", sigma=2.0, angle_centered=130),
        CgBead("Pr", sigma=2.5, angle_centered=130),
        CgBead("Pu", sigma=3.0, angle_centered=130),
        CgBead("P", sigma=3.5, angle_centered=130),
        CgBead("Re", sigma=4.0, angle_centered=130),
    )
    m2l4_4_beads = (
        CgBead("Pt", sigma=2.0, angle_centered=(90, 180, 130)),
    )
    m2l4_c_beads = (CgBead("C", sigma=4.0, angle_centered=180),)
    m2l4_b_beads = (CgBead("N", sigma=2.0, angle_centered=180),)
    full_bead_library = (
        m2l4_rhs_beads
        + m2l4_lhs_beads
        + m2l4_4_beads
        + m2l4_c_beads
        + m2l4_b_beads
    )
    bead_library_check(full_bead_library)

    c2_blocks = {}
    for rhs_bead in m2l4_rhs_beads:
        temp = UnsymmBiteLigand(
            centre_bead=m2l4_c_beads[0],
            lhs_bead=m2l4_lhs_beads[0],
            rhs_bead=rhs_bead,
            binder_bead=m2l4_b_beads[0],
        )

        opt_bb = optimise_ligand(
            molecule=temp.get_building_block(),
            name=temp.get_name(),
            output_dir=calculation_output,
            full_bead_library=full_bead_library,
        )
        opt_bb.write(str(ligand_output / f"{temp.get_name()}_optl.mol"))
        c2_blocks[temp.get_name()] = opt_bb

    logging.info(f"there are {len(c2_blocks)} building blocks.")
    fourc_precursor = get_fourc(m2l4_4_beads[0].element_string)

    experimental_structures = {
        "M6A1": {
            "tfunc": stk.cage.M6L12Cube,
            "vastr": "010100101001",
            "nmetals": 6,
            "nvertices": 12,
        },
        "M6A2": {
            "tfunc": stk.cage.M6L12Cube,
            "vastr": "011000010110",
            "nmetals": 6,
            "nvertices": 12,
        },
        "M6def": {
            "tfunc": stk.cage.M6L12Cube,
            "vastr": "000000000000",
            "nmetals": 6,
            "nvertices": 12,
        },
        "M12def": {
            "tfunc": CGM12L24,
            "vastr": "000000000000000000000000",
            "nmetals": 12,
            "nvertices": 24,
        },
        "M12E1": {
            "tfunc": CGM12L24,
            "vastr": "001111000011110001010101",
            "nmetals": 12,
            "nvertices": 24,
        },
        "M12E2": {
            "tfunc": CGM12L24,
            "vastr": "010110100011110001000111",
            "nmetals": 12,
            "nvertices": 24,
        },
        "M12E3": {
            "tfunc": CGM12L24,
            "vastr": "101001010101101000111100",
            "nmetals": 12,
            "nvertices": 24,
        },
        "M12G1": {
            "tfunc": CGM12L24,
            "vastr": "001111000011110010101010",
            "nmetals": 12,
            "nvertices": 24,
        },
    }

    cage_set_data = {}
    for c2blk in c2_blocks:
        cage_set_data[c2blk] = {}
        for expt_struct in experimental_structures:
            estruct_data = experimental_structures[expt_struct]
            vastr = estruct_data["vastr"]
            nmetals = estruct_data["nmetals"]
            nvertices = estruct_data["nvertices"]
            tfunc = estruct_data["tfunc"]

            va_dict = {
                i: int(j)
                for i, j in zip(
                    range(nmetals, nvertices + nmetals), vastr
                )
            }

            name = f"{expt_struct}_{c2blk}_{vastr}"
            cage = stk.ConstructedMolecule(
                topology_graph=tfunc(
                    building_blocks=(
                        c2_blocks[c2blk],
                        fourc_precursor,
                    ),
                    vertex_alignments=va_dict,
                ),
            )

            cage = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )

            cage.write(str(struct_output / f"{name}_optc.mol"))
            symm_data = analyse_big_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )

            cage_set_data[c2blk][expt_struct] = symm_data

    phase_space_3(
        cage_set_data,
        figure_output,
        full_bead_library,
        ligand_data,
    )


def get_a_ligand(cage, metal_atom_nos):

    # Produce a graph from the cage that does not include metals.
    cage_g = nx.Graph()
    atom_ids_in_G = set()
    for atom in cage.get_atoms():
        if atom.get_atomic_number() in metal_atom_nos:
            continue
        cage_g.add_node(atom)
        atom_ids_in_G.add(atom.get_id())

    # Add edges.
    for bond in cage.get_bonds():
        a1id = bond.get_atom1().get_id()
        a2id = bond.get_atom2().get_id()
        if a1id in atom_ids_in_G and a2id in atom_ids_in_G:
            cage_g.add_edge(bond.get_atom1(), bond.get_atom2())

    # Get disconnected subgraphs as molecules.
    # Sort and sort atom ids to ensure molecules are read by RDKIT
    # correctly.
    connected_graphs = [
        sorted(subgraph, key=lambda a: a.get_id())
        for subgraph in sorted(nx.connected_components(cage_g))
    ]
    for i, cg in enumerate(connected_graphs):
        # Get atoms from nodes.
        atoms = list(cg)
        atom_ids = [i.get_id() for i in atoms]
        cage.write("temporary_linker.mol", atom_ids=atom_ids)
        temporary_linker = stk.BuildingBlock.init_from_file(
            "temporary_linker.mol"
        )
        os.system("rm temporary_linker.mol")

        return temporary_linker


def extract_experimental_values():
    struct_input = unsymm() / "experimentals" / "ligands"
    fgs = (
        stk.SmartsFunctionalGroupFactory(
            smarts="[#6]~[#7X2]~[#6]",
            bonders=(1,),
            deleters=(),
        ),
    )
    results_dict = {
        # study (1,2), self-sort (0=yes)
        "3D1_C": (1, 1),
        "4B1_C": (2, 0),
        "4B3_C": (2, 1),
        "4D2_C": (1, 0),
        "5A1_C": (2, 0),
        "5A3_C": (2, 0),
        "5B4_C": (2, 0),
        "5D1_C": (1, 0),
        "5D3_C": (1, 0),
    }

    all_cages = struct_input.glob("*_optc.mol")
    ligand_data = {}
    for cage in all_cages:
        ligand_name = str(cage.name).replace("_optc.mol", "")
        print(ligand_name)
        ligand_file = str(struct_input / f"{ligand_name}_l.mol")
        anchor_file = str(struct_input / f"{ligand_name}_a.xyz")

        struct = stk.BuildingBlock.init_from_file(str(cage))
        ligand = get_a_ligand(cage=struct, metal_atom_nos=(46,))
        ligand.write(ligand_file)

        ligand = stk.BuildingBlock.init_from_molecule(
            molecule=ligand,
            functional_groups=fgs,
        )
        lposmat = ligand.get_position_matrix()
        print(ligand.get_num_functional_groups())
        position_set = {}
        for i, fg in enumerate(ligand.get_functional_groups()):
            (nitrogen,) = fg.get_bonders()
            carbon1, carbon2 = (
                i
                for i in fg.get_atoms()
                if i.get_id() != nitrogen.get_id()
            )
            print(fg)
            print(nitrogen, carbon1, carbon2)
            n_pos = lposmat[nitrogen.get_id()]
            cc_centroid = ligand.get_centroid(
                atom_ids=(carbon1.get_id(), carbon2.get_id())
            )
            print(n_pos, cc_centroid)
            position_set[f"n{i}"] = n_pos
            position_set[f"c{i}"] = cc_centroid

        print(position_set)
        with open(anchor_file, "w") as f:
            f.write("4\n\n")
            for i in position_set:
                name = i[0].upper()
                x, y, z = position_set[i]
                f.write(f"{name} {x} {y} {z}\n")

        angle1 = np.degrees(
            angle_between(
                v1=position_set["n0"] - position_set["c0"],
                v2=position_set["c1"] - position_set["c0"],
            )
        )
        angle2 = np.degrees(
            angle_between(
                v1=position_set["n1"] - position_set["c1"],
                v2=position_set["c0"] - position_set["c1"],
            )
        )

        print(angle1, angle2)
        smaller_angle = min((angle1, angle2))

        hyp = np.linalg.norm(position_set["n1"] - position_set["n0"])
        extension = np.cos(np.radians(smaller_angle)) * hyp
        print(extension)

        ligand_data[ligand_name] = {
            "angle1": angle1,
            "angle2": angle2,
            "extension": extension,
            "result": results_dict[ligand_name],
        }

    return ligand_data


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    ligand_data = extract_experimental_values()
    unsymm_1(ligand_data)
    unsymm_2(ligand_data)
    unsymm_3(ligand_data)

    raise SystemExit()

    # Define bead libraries.
    beads_4c_lib = beads_4c()
    beads_2c_lib = lig_2c_beads()
    beads_binder_lib = binder_beads()
    full_bead_library = beads_4c_lib + beads_2c_lib + beads_binder_lib
    bead_library_check(full_bead_library)

    # For now, just build N options and calculate properties.
    logging.info("building building blocks")

    c2_blocks = {}
    for c2_options in itertools.product(beads_2c_lib, repeat=3):
        if c2_options[2].element_string == c2_options[1].element_string:
            continue

        temp = UnsymmLigand(
            centre_bead=c2_options[0],
            lhs_bead=c2_options[1],
            rhs_bead=c2_options[2],
            binder_bead=beads_binder_lib[0],
        )
        opt_bb = optimise_ligand(
            molecule=temp.get_building_block(),
            name=temp.get_name(),
            output_dir=calculation_output,
            full_bead_library=full_bead_library,
        )
        opt_bb.write(str(ligand_output / f"{temp.get_name()}_optl.mol"))
        c2_blocks[temp.get_name()] = opt_bb

    logging.info(f"there are {len(c2_blocks)} building blocks.")
    fourc_precursor = get_fourc()

    ga_run_infos = {
        0: {
            "tfunc": stk.cage.M2L4Lantern,
            "plot_fun": plot_existing_M2_data_distributions,
            "num_vertices": 4,
            "seed_set": [256, 9, 123, 986],
        },
        1: {
            "tfunc": stk.cage.M6L12Cube,
            "plot_fun": plot_existing_M6_data_distributions,
            "num_vertices": 12,
            "seed_set": [12, 673, 98, 22],
        },
        2: {
            "tfunc": CGM12L24,
            "plot_fun": plot_existing_M12_data_distributions,
            "num_vertices": 24,
            "seed_set": [726, 23, 44, 1],
        },
        # 3: {
        #     "tfunc": CGM4L8,
        #     "seed_set": [14, 33, 2, 4122],
        # },
    }

    # Settings for runs.
    population_size_per_step = 40
    num_generations = 40

    for ga_run in ga_run_infos:
        ga_data = ga_run_infos[ga_run]
        seeds = ga_data["seed_set"]
        plot_fn = ga_data["plot_fun"]
        plot_fn(
            calculation_dir=calculation_output,
            figures_dir=figure_output,
        )
        cage_topology_function = ga_data["tfunc"]

        # Set seeds for reproducible results.
        for seed in seeds:
            logging.info(
                f"setting up the EA for seed {seed} and {ga_run}"
            )
            run_name = f"s{seed}_i{ga_run}"
            generator = np.random.RandomState(seed)

            # For now, just build N options and calculate properties.
            logging.info(
                f"building population of {population_size_per_step}"
            )

            raise SystemExit(
                "Won't work from here - you need to decide on approach!"
                " either, one ligand, find symm or many ligands, find "
                "best. The second approach needs to redefine GA! "
            )
            initial_population = tuple(
                get_initial_population(
                    cage_topology_function=cage_topology_function,
                    twoc_precursor=twoc_precursor,
                    fourc_precursor=fourc_precursor,
                    num_population=population_size_per_step,
                    generator=generator,
                )
            )

            mutation_selector = VaRoulette(
                num_batches=30,
                # Small batch sizes are MUCH more efficient.
                batch_size=1,
                duplicate_batches=False,
                duplicate_molecules=False,
                random_seed=generator.randint(0, 1000),
                key_maker=VaKeyMaker(),
            )

            crossover_selector = VaRoulette(
                num_batches=30,
                # Small batch sizes are MUCH more efficient.
                batch_size=1,
                duplicate_batches=False,
                duplicate_molecules=False,
                random_seed=generator.randint(0, 1000),
                key_maker=VaKeyMaker(),
            )

            ea = CgEvolutionaryAlgorithm(
                initial_population=initial_population,
                fitness_calculator=RecordFitnessFunction(
                    get_fitness_value
                ),
                mutator=mutator(generator, unsymm_topology_options()),
                crosser=crosser(generator, unsymm_topology_options()),
                generation_selector=VaBest(
                    num_batches=population_size_per_step,
                    batch_size=1,
                ),
                mutation_selector=mutation_selector,
                crossover_selector=crossover_selector,
                key_maker=VaKeyMaker(),
                num_processes=1,
            )

            generations = []
            logging.info(
                f"running EA for {num_generations} generations"
            )
            for i, generation in enumerate(
                ea.get_generations(num_generations)
            ):
                generations.append(generation)

                fitness_progress = CgProgressPlotter(
                    generations=generations,
                    get_property=(
                        lambda record: record.get_fitness_value()
                    ),
                    y_label="fitness value",
                )
                fitness_progress.write(
                    str(
                        figure_output / f"unsymm_fitness_{run_name}.pdf"
                    )
                )

            logging.info("EA done!")

            fitness_progress.write(
                str(figure_output / f"unsymm_fitness_{run_name}.pdf")
            )

        plot_existing_M12_data_distributions(
            calculation_dir=calculation_output,
            figures_dir=figure_output,
        )

        raise SystemExit(
            "You want a set of ligands relevant to each cage type"
        )

        for iname, twoc_precursor in enumerate(get_twocs()):
            build_experimentals(
                cage_topology_function=cage_topology_function,
                twoc_precursor=twoc_precursor,
                fourc_precursor=fourc_precursor,
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
