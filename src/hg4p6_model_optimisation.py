#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to optimise CG models of fourplussix host-guest systems.

Author: Andrew Tarzia

"""

import sys
import stk
import os
import json
import logging
import numpy as np
from scipy.spatial.distance import cdist
import spindry as spd
from itertools import combinations

from shape import ShapeMeasure

from env_set import (
    fourplussix_figures,
    fourplussix_hg_optimisation,
    fourplussix_calculations,
    guest_structures,
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
    guest_beads,
)

from ea_module import (
    CgEvolutionaryAlgorithm,
    RecordFitnessFunction,
)
from ea_plotters import (
    CgProgressPlotter,
    plot_existing_guest_data_distributions,
)

from guest import Guest
from fourplussix_model_optimisation import (
    mutator,
    crosser,
    get_molecule_name_from_record,
    calculate_pore,
    get_shape_atom_numbers,
    get_initial_population,
    get_final_energy,
    get_OH6_measure,
)


def get_supramolecule(hgcomplex):

    atoms = tuple(
        spd.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        )
        for atom in hgcomplex.get_atoms()
    )
    bonds = list(
        (
            spd.Bond(
                id=i,
                atom_ids=(
                    bond.get_atom1().get_id(),
                    bond.get_atom2().get_id(),
                ),
            )
            for i, bond in enumerate(hgcomplex.get_bonds())
        )
    )

    pairs = combinations(hgcomplex.get_atoms(), 2)
    last_id = bonds[-1].get_id()
    for pair in pairs:
        atom1, atom2 = pair
        estring1 = atom1.__class__.__name__
        estring2 = atom2.__class__.__name__
        guest_estrings = tuple(
            i
            for i in (estring1, estring2)
            if i in tuple(i.element_string for i in guest_beads())
        )
        if len(guest_estrings) != 2:
            continue

        bonds.append(
            spd.Bond(
                id=last_id + 1,
                atom_ids=(atom1.get_id(), atom2.get_id()),
            )
        )
        last_id += 1

    return spd.SupraMolecule(
        atoms=atoms,
        bonds=bonds,
        position_matrix=hgcomplex.get_position_matrix(),
    )


def get_results_dictionary(molecule_record):
    output_dir = fourplussix_calculations()
    # Define target guest.

    central_2c_atom_number = get_shape_atom_numbers(molecule_record)

    molecule = molecule_record.get_molecule()
    run_prefix = get_molecule_name_from_record(molecule_record)

    pm_output_file = os.path.join(
        output_dir, f"{run_prefix}_opted.json"
    )
    output_file = os.path.join(output_dir, f"{run_prefix}_hgres.json")
    opt_xyz_file = os.path.join(output_dir, f"{run_prefix}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{run_prefix}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{run_prefix}_opted2.mol")
    hg_mol_file = os.path.join(output_dir, f"{run_prefix}_hg.mol")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        target_guest = Guest(
            stk.BuildingBlock.init_from_file(
                str(guest_structures() / "3.mol")
            )
        )
        # Does optimisation.
        if os.path.exists(opt1_mol_file):
            molecule = molecule.with_structure_from_file(opt1_mol_file)
        else:
            logging.info(f"optimising {run_prefix}...")
            opt = CGGulpOptimizer(
                fileprefix=run_prefix,
                output_dir=output_dir,
                param_pool=(
                    core_beads()
                    + beads_2c()
                    + beads_3c()
                    + guest_beads()
                ),
                max_cycles=1000,
                conjugate_gradient=True,
            )
            run_data = opt.optimize(molecule)
            molecule = molecule.with_structure_from_file(opt_xyz_file)
            molecule.write(opt1_mol_file)

        opt = CGGulpOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=(
                core_beads() + beads_2c() + beads_3c() + guest_beads()
            ),
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

        # Build HG complex.
        hgcomplex = stk.ConstructedMolecule(
            topology_graph=stk.host_guest.Complex(
                host=stk.BuildingBlock.init_from_molecule(molecule),
                guests=stk.host_guest.Guest(
                    building_block=(
                        target_guest.get_convexhul_stk_molecule()
                    ),
                ),
            ),
        )
        # Want to optimize host position.
        # Can either, 1) use pywindow alg to get opt centred, and place
        # it there. 2) MC on guest position to minimise some E func.
        # 3) Run spindry with roations.
        spindry_complex = get_supramolecule(hgcomplex)

        logging.info("SPD: no optimmisation of complex yet.")
        hgcomplex.write(hg_mol_file)
        comps = list(spindry_complex.get_components())
        if len(comps) > 2:
            raise ValueError("more than one guest there buddy!")
        if len(comps) < 2:
            raise ValueError("less than one guest there buddy!")

        hg_distances = cdist(
            comps[0].get_position_matrix(),
            comps[1].get_position_matrix(),
        )

        res_dict = {
            "fin_energy": fin_energy,
            "opt_pore_data": opt_pore_data,
            "oh6_measure": oh6_measure,
            "guest_volume": target_guest.get_ch_volume(),
            "hg_distances": hg_distances.tolist(),
        }

        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    return res_dict


def flatten(li):
    return [item for sublist in li for item in sublist]


def get_volume_ratio(molecule_record):
    res_dict = get_results_dictionary(molecule_record)

    return (
        res_dict["guest_volume"]
        / res_dict["opt_pore_data"]["pore_volume"]
    )


def get_min_distance(molecule_record):
    res_dict = get_results_dictionary(molecule_record)

    return np.min(flatten(res_dict["hg_distances"]))


def fitness_from_dictionary(res_dict):
    pore_data = res_dict["opt_pore_data"]
    volume_ratio = res_dict["guest_volume"] / pore_data["pore_volume"]
    volume_ratio_score = abs(volume_ratio - 0.55)

    min_distance = np.min(flatten(res_dict["hg_distances"]))
    min_distance_score = abs((min_distance - 2)) / 2.0
    score = 10 / (
        res_dict["fin_energy"]
        + res_dict["oh6_measure"] * 100
        + volume_ratio_score * 100
        + min_distance_score * 100
    )
    return score


def get_fitness_value(molecule_record):
    res_dict = get_results_dictionary(molecule_record)
    return fitness_from_dictionary(res_dict)


def main():
    first_line = f"Usage: {__file__}.py mode"
    if not len(sys.argv) == 2:
        print(f"{first_line}")
        print(
            "\nmode must be either `run` or `plot`, where `plot` just "
            "outputs known distributions."
        )
        sys.exit()
    else:
        mode = sys.argv[1]

    if mode not in ("run", "plot"):
        raise ValueError(f"mode must be run or plot, not {mode}")

    struct_output = fourplussix_hg_optimisation()
    figure_output = fourplussix_figures()
    calculation_output = fourplussix_calculations()

    # Define list of topology functions.
    cage_topologies = cage_topology_options()

    # Define precursor topologies.
    three_precursor_topologies = three_precursor_topology_options()
    two_precursor_topologies = two_precursor_topology_options()

    target_guest = Guest(
        stk.BuildingBlock.init_from_file(
            str(guest_structures() / "3.mol")
        )
    )

    plot_existing_guest_data_distributions(
        calculation_dir=calculation_output,
        figures_dir=figure_output,
        suffix="hgres",
        # fitness_function=fitness_from_dictionary,
    )
    if mode == "plot":
        raise SystemExit("done plotting, bye!")

    # Settings for runs.
    population_size_per_step = 100
    num_generations = 100
    # Set seeds for reproducible results.
    seeds = [32, 2, 1223, 442, 5, 895]

    for seed in seeds:
        logging.info(f"setting up the EA for seed {seed}...")
        run_name = f"hg_s{seed}"
        generator = np.random.RandomState(seed)

        # For now, just build N options and calculate properties.
        logging.info(
            f"building population of {population_size_per_step}..."
        )
        initial_population = tuple(
            get_initial_population(
                cage_topologies=cage_topologies,
                three_precursor_topologies=three_precursor_topologies,
                two_precursor_topologies=two_precursor_topologies,
                num_population=population_size_per_step,
                generator=generator,
            )
        )

        mutation_selector = stk.Roulette(
            num_batches=40,
            # Small batch sizes are MUCH more efficient.
            batch_size=1,
            duplicate_batches=False,
            duplicate_molecules=False,
            random_seed=generator.randint(0, 1000),
        )

        crossover_selector = stk.Roulette(
            num_batches=40,
            # Small batch sizes are MUCH more efficient.
            batch_size=1,
            duplicate_batches=False,
            duplicate_molecules=False,
            random_seed=generator.randint(0, 1000),
        )

        ea = CgEvolutionaryAlgorithm(
            initial_population=initial_population,
            fitness_calculator=RecordFitnessFunction(get_fitness_value),
            mutator=mutator(generator, cage_topologies),
            crosser=crosser(generator),
            generation_selector=stk.Best(
                num_batches=population_size_per_step,
                batch_size=1,
            ),
            mutation_selector=mutation_selector,
            crossover_selector=crossover_selector,
            num_processes=1,
        )

        writer = stk.MolWriter()
        generations = []
        logging.info(f"running EA for {num_generations} generations...")
        for i, generation in enumerate(
            ea.get_generations(num_generations)
        ):
            generations.append(generation)

            fitness_progress = CgProgressPlotter(
                generations=generations,
                get_property=lambda record: record.get_fitness_value(),
                y_label="fitness value",
            )
            fitness_progress.write(
                str(figure_output / f"fitness_progress_{run_name}.pdf")
            )

            for molecule_id, molecule_record in enumerate(
                generation.get_molecule_records()
            ):
                molecule_name = get_molecule_name_from_record(
                    molecule_record
                )
                opt2_mol_file = os.path.join(
                    fourplussix_calculations(),
                    f"{molecule_name}_hg.mol",
                )
                mol = molecule_record.get_molecule()
                opt_mol = mol.with_structure_from_file(opt2_mol_file)
                hgcomplex = stk.ConstructedMolecule(
                    topology_graph=stk.host_guest.Complex(
                        host=stk.BuildingBlock.init_from_molecule(
                            opt_mol
                        ),
                        guests=stk.host_guest.Guest(
                            building_block=(
                                target_guest.get_convexhul_stk_molecule()
                            ),
                        ),
                    ),
                )
                writer.write(
                    molecule=hgcomplex,
                    path=os.path.join(
                        struct_output,
                        f"HG_g_{i}_m_{molecule_id}_{molecule_name}.mol",
                    ),
                )

        logging.info("EA done!")

        fitness_progress.write(
            str(figure_output / f"fitness_progress_{run_name}.pdf")
        )

        CgProgressPlotter(
            generations=generations,
            get_property=lambda record: get_min_distance(record),
            y_label="min distance",
        ).write(
            str(figure_output / f"distance_progress_{run_name}.pdf")
        )

        CgProgressPlotter(
            generations=generations,
            get_property=lambda record: get_OH6_measure(record),
            y_label="OH6 measure",
        ).write(str(figure_output / f"shape_progress_{run_name}.pdf"))

        CgProgressPlotter(
            generations=generations,
            get_property=lambda record: get_final_energy(record),
            y_label="final energy",
        ).write(str(figure_output / f"energy_progress_{run_name}.pdf"))

        CgProgressPlotter(
            generations=generations,
            get_property=lambda record: get_volume_ratio(record),
            y_label="volume ratio",
        ).write(str(figure_output / f"volume_progress_{run_name}.pdf"))

        plot_existing_guest_data_distributions(
            calculation_dir=calculation_output,
            figures_dir=figure_output,
            suffix="hgres",
            # fitness_function=fitness_from_dictionary,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
