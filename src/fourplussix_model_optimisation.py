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
import logging
from rdkit.Chem import rdMolHash
import pore_mapper as pm
import numpy as np

from shape import ShapeMeasure

from env_set import (
    fourplussix_figures,
    fourplussix_optimisation,
    fourplussix_calculations,
)

from gulp_optimizer import CGGulpOptimizer

from fourplusix_construction.topologies import cage_topology_options

from precursor_db.precursors import (
    three_precursor_topology_options,
    two_precursor_topology_options,
)

from beads import (
    core_beads,
    beads_2c,
    beads_3c,
)

from ea_module import (
    RandomCgBead,
    CgGeneticRecombination,
    CgEvolutionaryAlgorithm,
    RecordFitnessFunction,
)
from ea_plotters import (
    CgProgressPlotter,
    plot_existing_data_distributions,
)


def calculate_pore(molecule, output_file):

    xyz_file = output_file.replace(".json", ".xyz")
    molecule.write(xyz_file)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            pore_data = json.load(f)
        return pore_data

    host = pm.Host.init_from_xyz_file(xyz_file)
    host = host.with_centroid([0.0, 0.0, 0.0])

    # Define calculator object.
    calculator = pm.Inflater(bead_sigma=1.2)
    # Run calculator on host object, analysing output.
    final_result = calculator.get_inflated_blob(host=host)
    pore = final_result.pore
    blob = final_result.pore.get_blob()
    windows = pore.get_windows()
    pore_data = {
        "step": final_result.step,
        "num_movable_beads": final_result.num_movable_beads,
        "windows": windows,
        "blob_max_diam": blob.get_maximum_diameter(),
        "pore_max_rad": pore.get_maximum_distance_to_com(),
        "pore_mean_rad": pore.get_mean_distance_to_com(),
        "pore_volume": pore.get_volume(),
        "asphericity": pore.get_asphericity(),
        "shape": pore.get_relative_shape_anisotropy(),
    }
    with open(output_file, "w") as f:
        json.dump(pore_data, f)
    return pore_data


def get_initial_population(
    cage_topologies,
    three_precursor_topologies,
    two_precursor_topologies,
    num_population,
    generator,
):

    for i in range(num_population):
        selected_cage_topology = generator.choice(
            list(cage_topologies.keys())
        )

        s_2c_topology = generator.choice(
            list(two_precursor_topologies.keys())
        )
        bb_2c_template = two_precursor_topologies[s_2c_topology]
        bb_2c = bb_2c_template.get_building_block(
            core_bead_lib=core_beads(),
            bead_2c_lib=beads_2c(),
            generator=generator,
        )

        s_3c_topology = generator.choice(
            list(three_precursor_topologies.keys())
        )
        bb_3c_template = three_precursor_topologies[s_3c_topology]
        bb_3c = bb_3c_template.get_building_block(
            bead_2c_lib=beads_2c(),
            bead_3c_lib=beads_3c(),
            generator=generator,
        )
        yield stk.MoleculeRecord(
            topology_graph=cage_topologies[selected_cage_topology](
                building_blocks=(bb_2c, bb_3c),
            ),
        )


def get_molecule_formula(molecule):
    rdk_mol = molecule.to_rdkit_mol()
    hash_function = rdMolHash.HashFunction.MolFormula
    return rdMolHash.MolHash(rdk_mol, hash_function)


def get_molecule_name_from_record(record):
    tg = record.get_topology_graph()
    molecule = record.get_molecule()
    chemmform = get_molecule_formula(molecule)
    return f"{tg.__class__.__name__}_{chemmform}"


def get_shape_atom_numbers(molecule_record):
    # Get the atom number for atoms used in shape measure.
    bbs = list(
        molecule_record.get_topology_graph().get_building_blocks()
    )
    two_c_bb = bbs[1]
    max_atom_id = max([i.get_id() for i in two_c_bb.get_atoms()])
    central_2c_atom = tuple(two_c_bb.get_atoms(atom_ids=max_atom_id))[0]
    central_2c_atom_number = central_2c_atom.get_atomic_number()
    return central_2c_atom_number


def get_results_dictionary(molecule_record):
    output_dir = fourplussix_calculations()

    central_2c_atom_number = get_shape_atom_numbers(molecule_record)

    molecule = molecule_record.get_molecule()
    run_prefix = get_molecule_name_from_record(molecule_record)

    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")
    pm_output_file = os.path.join(
        output_dir, f"{run_prefix}_opted.json"
    )
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
            logging.info(f"running optimisation of {run_prefix}...")
            opt = CGGulpOptimizer(
                fileprefix=run_prefix,
                output_dir=output_dir,
                param_pool=core_beads() + beads_2c() + beads_3c(),
                max_cycles=1000,
                conjugate_gradient=True,
            )
            run_data = opt.optimize(molecule)
            molecule = molecule.with_structure_from_file(opt_xyz_file)
            molecule.write(opt1_mol_file)

        opt = CGGulpOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=core_beads() + beads_2c() + beads_3c(),
            max_cycles=1000,
            conjugate_gradient=False,
        )
        if os.path.exists(opt2_mol_file):
            molecule = molecule.with_structure_from_file(opt2_mol_file)
            run_data = opt.extract_gulp()
        else:
            run_data = opt.optimize(molecule)
            molecule = molecule.with_structure_from_file(opt_xyz_file)
            molecule.write(opt2_mol_file)

        # Minimises energy, OH shape and targets pore size of 5 A.
        oh6_measure = ShapeMeasure(
            output_dir=(output_dir / f"{run_prefix}_shape"),
            target_atmnum=central_2c_atom_number,
            shape_string="oc6",
        ).calculate(molecule)

        fin_energy = run_data["final_energy"]
        opt_pore_data = calculate_pore(molecule, pm_output_file)
        res_dict = {
            "fin_energy": fin_energy,
            "opt_pore_data": opt_pore_data,
            "oh6_measure": oh6_measure,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    return res_dict


def get_pore_radius(molecule_record):
    res_dict = get_results_dictionary(molecule_record)

    return res_dict["opt_pore_data"]["pore_max_rad"]


def get_OH6_measure(molecule_record):
    res_dict = get_results_dictionary(molecule_record)

    return res_dict["oh6_measure"]


def get_final_energy(molecule_record):
    res_dict = get_results_dictionary(molecule_record)

    return res_dict["fin_energy"]


def get_fitness_value(molecule_record):

    res_dict = get_results_dictionary(molecule_record)

    pore_radius = res_dict["opt_pore_data"]["pore_max_rad"]
    pore_size_diff = abs(5 - pore_radius * 2) / 5
    score = (
        1
        / (
            res_dict["fin_energy"]
            + res_dict["oh6_measure"] * 100
            + pore_size_diff * 100
        )
        * 10
    )
    return score


def mutator(generator, cage_topologies):
    bead_2c_lib = beads_2c()
    bead_3c_lib = beads_3c()
    bead_core_lib = core_beads()

    return stk.RandomMutator(
        mutators=(
            # Substitutes a 2c CGBead with another.
            RandomCgBead(
                bead_library=bead_2c_lib,
                random_seed=generator.randint(0, 1000),
            ),
            # Substitutes a 3c CGBead with another.
            RandomCgBead(
                bead_library=bead_3c_lib,
                random_seed=generator.randint(0, 1000),
            ),
            # Substitutes a core CGBead with another.
            RandomCgBead(
                bead_library=bead_core_lib,
                random_seed=generator.randint(0, 1000),
            ),
            stk.RandomTopologyGraph(
                replacement_funcs=tuple(
                    lambda graph: cage_topologies[i](
                        graph.get_building_blocks()
                    )
                    for i in cage_topologies
                )
            ),
        ),
        random_seed=generator.randint(0, 1000),
    )


def crosser(generator):
    def get_num_functional_groups(building_block):
        return building_block.get_num_functional_groups()

    return CgGeneticRecombination(
        get_gene=get_num_functional_groups,
    )


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = fourplussix_optimisation()
    figure_output = fourplussix_figures()
    calculation_output = fourplussix_calculations()

    # Define list of topology functions.
    cage_topologies = cage_topology_options()

    # Define precursor topologies.
    three_precursor_topologies = three_precursor_topology_options()
    two_precursor_topologies = two_precursor_topology_options()

    population_size_per_step = 20
    num_generations = 100
    # Set seed for reproducible results.
    logging.info(
        "using a constant, reproducible random state from seed 4"
    )
    generator = np.random.RandomState(4)

    # For now, just build N options and calculate properties.
    initial_population = tuple(
        get_initial_population(
            cage_topologies=cage_topologies,
            three_precursor_topologies=three_precursor_topologies,
            two_precursor_topologies=two_precursor_topologies,
            num_population=population_size_per_step,
            generator=generator,
        )
    )

    initial_fitness_values = []
    for i, mol in enumerate(initial_population):
        initial_fitness_values.append(get_fitness_value(mol))

    plot_existing_data_distributions(
        calculation_dir=calculation_output,
        figures_dir=figure_output,
    )

    # mut = RandomCgBead(
    #     bead_library=core_beads(),
    #     random_seed=np.random.RandomState(4).randint(0, 1000),
    # )
    # print(initial_population[0])
    # record = mut.mutate(record=initial_population[0])
    # print(record)
    # raise SystemExit()

    logging.info("setting up the EA...")
    ea = CgEvolutionaryAlgorithm(
        initial_population=initial_population,
        fitness_calculator=RecordFitnessFunction(get_fitness_value),
        mutator=mutator(generator, cage_topologies),
        crosser=crosser(generator),
        generation_selector=stk.Best(
            num_batches=population_size_per_step,
            batch_size=1,
        ),
        mutation_selector=stk.Roulette(
            num_batches=10,
            # Small batch sizes are MUCH more efficient.
            batch_size=2,
            duplicate_batches=False,
            duplicate_molecules=False,
            random_seed=generator.randint(0, 1000),
        ),
        crossover_selector=stk.Roulette(
            num_batches=10,
            # Small batch sizes are MUCH more efficient.
            batch_size=2,
            duplicate_batches=False,
            duplicate_molecules=False,
            random_seed=generator.randint(0, 1000),
        ),
        num_processes=1,
    )

    writer = stk.MolWriter()
    generations = []
    logging.info(f"running the EA for {num_generations}...")
    for i, generation in enumerate(ea.get_generations(num_generations)):
        generations.append(generation)

        for molecule_id, molecule_record in enumerate(
            generation.get_molecule_records()
        ):
            molecule_name = get_molecule_name_from_record(
                molecule_record
            )
            opt2_mol_file = os.path.join(
                fourplussix_calculations(),
                f"{molecule_name}_opted2.mol",
            )
            opt_mol = (
                molecule_record.get_molecule().with_structure_from_file(
                    opt2_mol_file
                )
            )
            writer.write(
                molecule=opt_mol,
                path=os.path.join(
                    struct_output,
                    f"g_{i}_m_{molecule_id}_{molecule_name}.mol",
                ),
            )

    fitness_progress = CgProgressPlotter(
        generations=generations,
        get_property=lambda record: record.get_fitness_value(),
        y_label="Fitness Value",
    )
    fitness_progress.write(
        str(fourplussix_figures() / "fitness_progress.pdf")
    )

    fitness_progress = CgProgressPlotter(
        generations=generations,
        get_property=lambda record: get_pore_radius(record),
        y_label="pore radius",
    )
    fitness_progress.write(
        str(fourplussix_figures() / "pore_progress.pdf")
    )

    fitness_progress = CgProgressPlotter(
        generations=generations,
        get_property=lambda record: get_OH6_measure(record),
        y_label="OH6 measure",
    )
    fitness_progress.write(
        str(fourplussix_figures() / "shape_progress.pdf")
    )

    fitness_progress = CgProgressPlotter(
        generations=generations,
        get_property=lambda record: get_final_energy(record),
        y_label="final energy",
    )
    fitness_progress.write(
        str(fourplussix_figures() / "energy_progress.pdf")
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
