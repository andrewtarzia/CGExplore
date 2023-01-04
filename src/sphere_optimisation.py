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


from env_set import sphere
import utilities
from gulp_optimizer import CGGulpOptimizer

from beads import CgBead

from ea_module import (
    CgEvolutionaryAlgorithm,
    RecordFitnessFunction,
)
from ea_plotters import (
    CgProgressPlotter,
    plot_existing_guest_data_distributions,
)

from fourplussix_model_optimisation import (
    crosser,
    get_final_energy,
)


class Sphere(stk.host_guest.Complex):
    def __init__(
        self,
        host,
        num_processes=1,
        optimizer=stk.NullOptimizer(),
    ) -> None:

        building_block_vertices = {
            host: (stk.host_guest.HostVertex(0, (0.0, 0.0, 0.0)),)
        }

        super(stk.host_guest.Complex, self).__init__(
            building_block_vertices=building_block_vertices,
            edges=(),
            reaction_factory=stk.GenericReactionFactory(),
            construction_stages=(),
            num_processes=num_processes,
            optimizer=optimizer,
            edge_groups=(),
        )

    def __repr__(self) -> str:
        return "Sphere()"


def bead_options():
    return (
        CgBead("O", 3.0, 1, (60, 160, 90)),
        CgBead("Cr", 3.0, 1, (60, 160, 90)),
        CgBead("Co", 3.0, 1, (60, 160, 90)),
        CgBead("Cu", 3.0, 1, (60, 160, 90)),
        CgBead("Pb", 3.0, 1, (60, 160, 90)),
        CgBead("Er", 3.0, 1, (60, 160, 90)),
        CgBead("Eu", 3.0, 1, (60, 160, 90)),
        CgBead("Mn", 4.0, 1, (60, 160, 90)),
        CgBead("Gd", 4.0, 1, (60, 160, 90)),
        CgBead("Ga", 4.0, 1, (60, 160, 90)),
        CgBead("Ge", 4.0, 1, (60, 160, 90)),
        CgBead("Au", 4.0, 1, (60, 160, 90)),
        CgBead("Ni", 4.0, 1, (60, 160, 90)),
        CgBead("He", 4.0, 1, (60, 160, 90)),
        CgBead("Al", 3.0, 2, (60, 160, 90)),
        CgBead("Sb", 3.0, 2, (60, 160, 90)),
        CgBead("Ar", 3.0, 2, (60, 160, 90)),
        CgBead("As", 3.0, 2, (60, 160, 90)),
        CgBead("Ba", 3.0, 2, (60, 160, 90)),
        CgBead("Be", 3.0, 2, (60, 160, 90)),
        CgBead("Bi", 3.0, 2, (60, 160, 90)),
        CgBead("B", 4.0, 2, (60, 160, 90)),
        CgBead("Mg", 4.0, 2, (60, 160, 90)),
        CgBead("Cd", 4.0, 2, (60, 160, 90)),
        CgBead("Hf", 4.0, 2, (60, 160, 90)),
        CgBead("Ca", 4.0, 2, (60, 160, 90)),
        CgBead("C", 4.0, 2, (60, 160, 90)),
        CgBead("Ce", 4.0, 2, (60, 160, 90)),
        CgBead("Ho", 2.0, 3, (60, 160, 90)),
        CgBead("Fe", 2.5, 3, (60, 160, 90)),
        CgBead("In", 3.0, 3, (60, 160, 90)),
        CgBead("I", 3.5, 3, (60, 160, 90)),
        CgBead("Ir", 4.0, 3, (60, 160, 90)),
    )


def build_sphere(atom_string):

    # Define a sphere molecule.
    sphere_radius = 10
    num_beads = 100
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(num_beads)
    z = np.linspace(
        1 - 1.0 / num_beads,
        1.0 / num_beads - 1.0,
        num_beads,
    )
    radius = np.sqrt(1 - z * z)
    points = np.zeros((3, num_beads))
    points[0, :] = radius * np.cos(theta) * sphere_radius
    points[1, :] = radius * np.sin(theta) * sphere_radius
    points[2, :] = z * sphere_radius

    position_matrix = np.array(
        points,
        dtype=np.float64,
    ).T
    dists = cdist(position_matrix, position_matrix)

    atoms = tuple(
        vars(stk)[atom_string](id=i) for i in range(num_beads)
    )
    bonds = []
    paired_ids = set()
    for atom in atoms:
        adist = dists[atom.get_id()]
        min_ids = np.argpartition(adist, 7)[:7]
        for id2 in min_ids:
            if id2 == atom.get_id():
                continue

            pair_idx = tuple(sorted((atom.get_id(), id2)))
            if pair_idx in paired_ids:
                continue
            bonds.append(
                stk.Bond(
                    atom1=atom,
                    atom2=vars(stk)[atom_string](id2),
                    order=1,
                )
            )
            paired_ids.add(pair_idx)

    return stk.BuildingBlock.init(
        atoms=atoms,
        bonds=tuple(bonds),
        position_matrix=position_matrix,
    )


def get_initial_population(num_population, generator):
    struct_output = sphere() / "structures"

    for i in range(num_population):
        selected_atom = generator.choice(bead_options())

        bb = build_sphere(atom_string=selected_atom.element_string)
        bb.write(str(struct_output / "input_sphere.mol"))

        yield stk.MoleculeRecord(
            topology_graph=Sphere(bb),
        )


def get_molecule_name_from_record(record):
    atomtype = tuple(record.get_molecule().get_atoms())[
        0
    ].__class__.__name__
    return f"{atomtype}"


def get_results_dictionary(molecule_record):
    output_dir = sphere() / "calculations"

    molecule = molecule_record.get_molecule()
    run_prefix = get_molecule_name_from_record(molecule_record)

    output_file = os.path.join(output_dir, f"{run_prefix}_sres.json")
    opt_xyz_file = os.path.join(output_dir, f"{run_prefix}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{run_prefix}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{run_prefix}_opted2.mol")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        molecule.write("input.mol")

        # Does optimisation.
        if os.path.exists(opt1_mol_file):
            molecule = molecule.with_structure_from_file(opt1_mol_file)
        else:
            logging.info(f"optimising {run_prefix}")
            opt = CGGulpOptimizer(
                fileprefix=run_prefix,
                output_dir=output_dir,
                param_pool=bead_options(),
                max_cycles=100,
                conjugate_gradient=False,
                bonds=True,
                angles=False,
                torsions=False,
                vdw=False,
            )
            run_data = opt.optimize(molecule)
            molecule = molecule.with_structure_from_file(opt_xyz_file)
            molecule.write(opt1_mol_file)

        opt = CGGulpOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=bead_options(),
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=False,
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

        raise SystemExit()

        fin_energy = run_data["final_energy"]

        res_dict = {
            "fin_energy": fin_energy,
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


class SphereBead(stk.ea.MoleculeMutator):
    def __init__(
        self,
        bead_library,
        name="SphereBead",
        random_seed=None,
    ):

        self._bead_library = bead_library
        self._name = name
        self._generator = np.random.RandomState(random_seed)

    def mutate(self, record):
        original_type = tuple(record.get_molecule().get_atoms())[
            0
        ].__class__.__name__
        new_bead_library = tuple(
            i
            for i in self._bead_library
            if i.element_string != original_type
        )
        selected_atom = self._generator.choice(new_bead_library)
        new_bb = build_sphere(atom_string=selected_atom.element_string)

        # Build the new ConstructedMolecule.
        graph = Sphere(new_bb)
        return stk.ea.MutationRecord(
            molecule_record=stk.ea.MoleculeRecord(graph),
            mutator_name=self._name,
        )


def mutator(generator):
    bead_lib = bead_options()

    return stk.RandomMutator(
        mutators=(
            SphereBead(
                bead_library=bead_lib,
                random_seed=generator.randint(0, 1000),
            ),
        ),
        random_seed=generator.randint(0, 1000),
    )


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = sphere() / "structures"
    utilities.check_directory(struct_output)
    figure_output = sphere() / "figures"
    utilities.check_directory(figure_output)
    calculation_output = sphere() / "calculations"
    utilities.check_directory(calculation_output)

    # Settings for runs.
    population_size_per_step = 10
    num_generations = 100
    # Set seeds for reproducible results.
    seeds = [32, 2, 1223, 442, 5, 895]

    for seed in seeds:
        logging.info(f"setting up the EA for seed {seed}")
        run_name = f"sphere_s{seed}"
        generator = np.random.RandomState(seed)

        # For now, just build N options and calculate properties.
        logging.info(
            f"building population of {population_size_per_step}"
        )
        initial_population = tuple(
            get_initial_population(
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
            mutator=mutator(generator),
            crosser=crosser(generator),
            generation_selector=stk.Best(
                num_batches=population_size_per_step,
                batch_size=1,
            ),
            mutation_selector=mutation_selector,
            crossover_selector=crossover_selector,
            num_processes=1,
        )

        generations = []
        logging.info(f"running EA for {num_generations} generations")
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
