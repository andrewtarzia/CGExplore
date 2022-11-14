#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to optimise CG models of fourplussix systems.

Author: Andrew Tarzia

"""

import sys
import stk
import os
import json
from rdkit.Chem import AllChem as rdkit
from collections import defaultdict
import logging
import numpy as np


from env_set import unsymm
from utilities import check_directory, angle_between

from gulp_optimizer import CGGulpOptimizer

from cage_construction.topologies import (
    CGM12L24,
    unsymm_topology_options,
)

from beads import CgBead

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
    plot_existing_M12_data_distributions,
)


def beads_2c():
    return (
        CgBead("Ge", sigma=2.0, angle_centered=120),
        CgBead("Eu", sigma=5.0, angle_centered=180),
        CgBead("Lu", sigma=6.0, angle_centered=180),
        CgBead("Ce", sigma=4.0, angle_centered=180),
        CgBead("He", sigma=2.0, angle_centered=180),
        CgBead("Be", sigma=5.0, angle_centered=120),
    )


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
            logging.info(f"optimising {run_prefix}...")
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


def build_from_beads(abead1, central, abead2):
    new_fgs = (
        stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{abead1.element_string}]"
                f"[{central.element_string}]"
            ),
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        ),
        stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{abead2.element_string}]"
                f"[{central.element_string}]"
            ),
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        ),
    )

    return stk.BuildingBlock(
        smiles=(
            f"[{abead1.element_string}][{central.element_string}]"
            f"[{abead2.element_string}]"
        ),
        functional_groups=new_fgs,
        position_matrix=[[-5, 0, 0], [0, 0, 0], [5, 0, 0]],
    )


def get_twocs():
    bead_lib = beads_2c()
    abead2 = tuple(i for i in bead_lib if i.element_string == "Ce")[0]
    central = tuple(i for i in bead_lib if i.element_string == "Eu")[0]
    abead1 = tuple(i for i in bead_lib if i.element_string == "Ge")[0]
    yield build_from_beads(abead1, central, abead2)

    central = tuple(i for i in bead_lib if i.element_string == "Be")[0]
    abead2 = tuple(i for i in bead_lib if i.element_string == "He")[0]

    for abead1_estring in ("Eu", "Ce", "Lu"):
        abead1 = tuple(
            i for i in bead_lib if i.element_string == abead1_estring
        )[0]
        yield build_from_beads(abead1, central, abead2)


def get_fourc():
    bead = beads_4c()[0]
    four_c_bb = stk.BuildingBlock(
        smiles=f"[Br][{bead.element_string}]([Br])([Br])[Br]",
        position_matrix=[
            [-2, 0, 0],
            [0, 0, 0],
            [0, -2, 0],
            [2, 0, 0],
            [0, 2, 0],
        ],
    )

    new_fgs = stk.SmartsFunctionalGroupFactory(
        smarts=(f"[{bead.element_string}]" f"[Br]"),
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


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = unsymm() / "optimisation"
    check_directory(struct_output)
    figure_output = unsymm() / "figures"
    check_directory(figure_output)
    calculation_output = unsymm() / "calculations"
    check_directory(calculation_output)

    # Define list of topology functions.
    cage_topology_function = CGM12L24

    seed_sets = {
        0: [256, 9, 123, 986],
        1: [12, 673, 98, 22],
        2: [726, 23, 44, 1],
        3: [14, 33, 2, 4122],
    }

    # Define precursor topologies.
    fourc_precursor = get_fourc()
    for iname, twoc_precursor in enumerate(get_twocs()):
        seeds = seed_sets[iname]
        build_experimentals(
            cage_topology_function=cage_topology_function,
            twoc_precursor=twoc_precursor,
            fourc_precursor=fourc_precursor,
        )

        plot_existing_M12_data_distributions(
            calculation_dir=calculation_output,
            figures_dir=figure_output,
        )

        # Settings for runs.
        population_size_per_step = 40
        num_generations = 40
        # Set seeds for reproducible results.
        for seed in seeds:
            logging.info(
                f"setting up the EA for seed {seed} and {iname}..."
            )
            run_name = f"s{seed}_i{iname}"
            generator = np.random.RandomState(seed)

            # For now, just build N options and calculate properties.
            logging.info(
                f"building population of {population_size_per_step}..."
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
                f"running EA for {num_generations} generations..."
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
