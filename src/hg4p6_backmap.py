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
from collections import Counter
import logging
import numpy as np
from scipy.spatial.distance import cdist
import spindry as spd
from itertools import combinations

from shape import ShapeMeasure

from env_set import (
    fourplussix_figures,
    fourplussix_optimisation,
    fourplussix_calculations,
    fourplussix_backmap,
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


def get_a_cage():

    # Define list of topology functions.
    cage_topologies = cage_topology_options()

    bb_2c = stk.BuildingBlock.init_from_file(
        path="temp_bbc2.mol",
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts=("[Ce][Eu]"),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        ),
    )

    bb_3c = stk.BuildingBlock.init_from_file(
        path="temp_bbc3.mol",
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts=("[Ir][Sb]"),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        ),
    )

    mol_rec = stk.MoleculeRecord(
        topology_graph=cage_topologies["FourPlusSix"](
            building_blocks=(bb_2c, bb_3c),
        ),
    )
    return mol_rec

    # generator = np.random.RandomState(4)
    # # Define precursor topologies.
    # three_precursor_topologies = three_precursor_topology_options()
    # two_precursor_topologies = two_precursor_topology_options()

    # for i in range(10000):

    #     selected_cage_topology = "FourPlusSix"

    #     s_2c_topology = "2c-1"

    #     bb_2c_template = two_precursor_topologies[s_2c_topology]
    #     bb_2c = bb_2c_template.get_building_block(
    #         core_bead_lib=core_beads(),
    #         bead_2c_lib=beads_2c(),
    #         generator=generator,
    #     )

    #     s_3c_topology = "3c-1"

    #     bb_3c_template = three_precursor_topologies[s_3c_topology]
    #     bb_3c = bb_3c_template.get_building_block(
    #         bead_2c_lib=beads_2c(),
    #         bead_3c_lib=beads_3c(),
    #         generator=generator,
    #     )
    #     mol_rec = stk.MoleculeRecord(
    #         topology_graph=cage_topologies[selected_cage_topology](
    #             building_blocks=(bb_2c, bb_3c),
    #         ),
    #     )
    #     mol_name = get_molecule_name_from_record(mol_rec)
    #     print(mol_name)
    #     if mol_name == "FourPlusSix_Ce12Eu6Ir4Sb12":
    #         print(bb_2c)
    #         print(bb_3c)
    #         bb_3c.write("temp_bbc3.mol")
    #         bb_2c.write("temp_bbc2.mol")
    #         return mol_rec

    # print("failed")
    # raise SystemExit()


def main():
    first_line = f"Usage: {__file__}.py "
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = fourplussix_backmap()
    cage_struct_output = fourplussix_optimisation()
    figure_output = fourplussix_figures()
    calculation_output = fourplussix_calculations()

    test_bb2 = stk.BuildingBlock(
        smiles="C1C(N([H])[H])=CC=C(C2C=CC(N([H])[H])=CC=2)C=1",
        functional_groups=(stk.PrimaryAminoFactory(),),
    )
    test_bb3 = stk.BuildingBlock(
        smiles="C1C(C=O)=CC(C=O)=CC=1C=O",
        functional_groups=(stk.AldehydeFactory(),),
    )
    test_bb2.write("test_bb2.mol")
    test_bb3.write("test_bb3.mol")
    topology_graph = stk.cage.FourPlusSix((test_bb2, test_bb3))
    construction_result = topology_graph.construct()
    test_cage = stk.ConstructedMolecule.init_from_construction_result(
        construction_result=construction_result,
    ).with_centroid((0, 0, 0))
    test_cage.write("test_cage.mol")

    temp_mol_rec = get_a_cage()

    temp_host_molecule = stk.BuildingBlock.init_from_file(
        str(
            cage_struct_output
            / "g_99_m_0_FourPlusSix_Ce12Eu6Ir4Sb12.mol"
        )
    ).with_centroid((0, 0, 0))
    temp_host_molecule.write("temp_host.mol")

    print(
        "you have AA as ConstructedMolecule..."
        "you have CG as MolRecord...\n\n"
        "This gives you all BB ids, plus AtomInfos- \n\n"
        "you could technically do MolRecord as ConstructedMolecule\n\n"
        "What needs doing is - match BB_id=0 in AA with BB_id=0 in CG\n\n"
        "Set positions of conneciton atoms of AA BB_ID to CG postions\n\n"
        "Set Centroid of AA BB to match CG BB\n\n"
        "Optimise -- this requires a UFF Gulp optimiser with "
        "opt.fix_atoms(atom_ids=()) as a new method, that fixes the "
        "placed atoms.\n\n"
        "Calculate strain relative to input?\n\n"
    )
    raise SystemExit()

    connection_beads = beads_2c()
    connection_bead_strings = tuple(
        i.element_string for i in connection_beads
    )
    present_connection_beads = {}
    for atom in temp_host_molecule.get_atoms():
        acname = atom.__class__.__name__
        if acname in connection_bead_strings:
            if acname not in present_connection_beads:
                present_connection_beads[acname] = []
            present_connection_beads[acname].append(atom)

    print(present_connection_beads)

    cg_beads_counts = Counter(
        tuple(
            i.__class__.__name__ for i in temp_host_molecule.get_atoms()
        )
    )
    print(cg_beads_counts)
    bb3_centre_atom_string = list(cg_beads_counts.keys())[
        list(cg_beads_counts.values()).index(4)
    ]
    bb2_centre_atom_string = list(cg_beads_counts.keys())[
        list(cg_beads_counts.values()).index(6)
    ]
    print(bb3_centre_atom_string)
    print(bb2_centre_atom_string)

    selected_bead_string = list(present_connection_beads.keys())[0]
    print(selected_bead_string)

    connection_atoms = present_connection_beads[selected_bead_string]
    print(connection_atoms)
    connection_pos_mat = np.array(
        tuple(
            temp_host_molecule.get_atomic_positions(
                atom_ids=(i.get_id() for i in connection_atoms)
            )
        )
    )
    print(connection_pos_mat)
    temp_host_molecule.write(
        "temp_connections.mol",
        atom_ids=tuple(i.get_id() for i in connection_atoms),
    )

    tpg_bbs = list(topology_graph.get_building_blocks())
    print(tpg_bbs)
    for bb_id in tpg_bbs:
        print(bb_id, tpg_bbs[bb_id])

    raise SystemExit()

    placed_atom_ids = set()
    for bond_info in test_cage.get_bond_infos():
        if bond_info.get_building_block() is not None:
            continue

        target_bond_info = bond_info
        print(target_bond_info)
        associated_atom_1 = target_bond_info.get_bond().get_atom1()
        associated_atom_2 = target_bond_info.get_bond().get_atom2()
        assoc_ids = tuple(
            i.get_id()
            for i in (
                associated_atom_1,
                associated_atom_2,
            )
        )
        if (
            assoc_ids[0] in placed_atom_ids
            or assoc_ids[1] in placed_atom_ids
        ):
            continue

        average_position = np.array(
            (
                test_cage.get_centroid(
                    atom_ids=assoc_ids,
                )
            )
        ).reshape(1, 3)
        print(average_position)
        atom_positions = tuple(
            test_cage.get_atomic_positions(atom_ids=assoc_ids)
        )
        atom_atom_vector = atom_positions[0] - atom_positions[1]
        print(atom_atom_vector)
        atom_atom_vector = atom_atom_vector / np.linalg.norm(
            atom_atom_vector
        )
        print(atom_atom_vector)

        conn_distances = cdist(
            average_position,
            connection_pos_mat,
        )
        print(conn_distances)
        conn_pos_mat_id = conn_distances.argmin()
        print(conn_pos_mat_id)
        connection_atom = connection_atoms[conn_pos_mat_id]
        print(connection_atom)
        connection_atom_pos = connection_pos_mat[conn_pos_mat_id]
        print(connection_atom_pos)

        new_posmat = []
        for i, pos in enumerate(test_cage.get_position_matrix()):
            if i in assoc_ids:
                new_posmat.append(connection_atom_pos)
            else:
                new_posmat.append(pos)

        new_posmat = np.array(new_posmat)
        test_cage = test_cage.with_position_matrix(new_posmat)

        placed_atom_ids.add(associated_atom_1.get_id())
        placed_atom_ids.add(associated_atom_2.get_id())

    test_cage.write("updated.mol")

    raise SystemExit()

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
