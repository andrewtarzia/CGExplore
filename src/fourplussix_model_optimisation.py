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
import random
import pore_mapper as pm
import numpy as np

from shape import ShapeMeasure

from env_set import (
    fourplussix_figures,
    fourplussix_optimisation,
    fourplussix_calculations,
)
from utilities import (
    get_distances,
    get_angles,
)
from gulp_optimizer import (
    CGGulpOptimizer,
    HarmBond,
    ThreeAngle,
    IntSet,
)

from fourplusix_construction.topologies import cage_topology_options

from fourplusix_construction.plotting import (
    convergence,
    scatter,
    geom_distributions,
    heatmap,
    ey_vs_shape,
)

from precursor_db.precursors import (
    three_precursor_topology_options,
    two_precursor_topology_options,
)

from beads import beads_1c, beads_2c, beads_3c


def calculate_pore(xyz_file):

    output_file = xyz_file.replace(".xyz", ".json")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            pore_data = json.load(f)
        return pore_data

    host = pm.Host.init_from_xyz_file(xyz_file)
    host = host.with_centroid([0.0, 0.0, 0.0])

    # Define calculator object.
    logging.warning(
        "currently using very small pore, would want to use normal "
        "size in future."
    )
    calculator = pm.Inflater(bead_sigma=0.5)
    # Run calculator on host object, analysing output.
    logging.info(f"calculating pore of {xyz_file}...")
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


class FourPlusSixOptimizer(CGGulpOptimizer):
    def define_bond_potentials(self):

        bonds = (
            # Intra-BB.
            HarmBond("B", "N", bond_r=1, bond_k=10),
        )

        return IntSet(self._update_bonds(bonds))

    def define_angle_potentials(self):
        angles = (
            # Intra-BB.
            ThreeAngle(
                atom1_type="B",
                atom2_type="N",
                atom3_type="N",
                theta=100,
                angle_k=20,
            ),
        )
        return IntSet(self._update_angles(angles))

    def define_torsion_potentials(self):
        torsions = ()
        return IntSet(self._update_torsions(torsions))

    def define_vdw_potentials(self):
        pairs = ()
        return IntSet(self._update_pairs(pairs))


def run_optimisation(
    cage,
    ff_modifications,
    ffname,
    topo_str,
    output_dir,
):

    run_prefix = f"{topo_str}_{ffname}"
    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f": running optimisation of {run_prefix}")
        opt = FourPlusSixOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=ff_modifications["param_pool"],
        )
        run_data = opt.optimize(cage)

        opted = cage.with_structure_from_file(
            path=os.path.join(output_dir, f"{run_prefix}_final.xyz"),
        )
        opted.write(os.path.join(output_dir, f"{run_prefix}_final.mol"))

        oh6_measure = ShapeMeasure(
            output_dir=output_dir / f"{run_prefix}_shape",
            target_atmnum=5,
            shape_string="oc6",
        ).calculate(opted)
        distances = get_distances(optimizer=opt, cage=opted)
        angles = get_angles(optimizer=opt, cage=opted)

        num_steps = len(run_data["traj"])
        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        traj_data = run_data["traj"]
        logging.info(
            f"{run_prefix}: {num_steps} {fin_energy} {fin_gnorm} "
            f"{oh6_measure}"
        )
        res_dict = {
            "fin_energy": fin_energy,
            "oh6": oh6_measure,
            "traj": traj_data,
            "distances": distances,
            "angles": angles,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f)

    return res_dict


def get_initial_population(
    cage_topologies,
    three_precursor_topologies,
    two_precursor_topologies,
    num_population,
):

    for i in range(num_population):
        selected_cage_topology = random.choice(
            list(cage_topologies.keys())
        )

        s_2c_topology = random.choice(
            list(two_precursor_topologies.keys())
        )
        bb_2c_template = two_precursor_topologies[s_2c_topology]

        bb_2c = bb_2c_template.get_building_block(
            bead_1c_lib=beads_1c(),
            bead_2c_lib=beads_2c(),
        )

        s_3c_topology = random.choice(
            list(three_precursor_topologies.keys())
        )
        bb_3c_template = three_precursor_topologies[s_3c_topology]
        bb_3c = bb_3c_template.get_building_block(
            bead_2c_lib=beads_2c(),
            bead_3c_lib=beads_3c(),
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


def get_fitness_value(molecule_record, output_dir):

    tg = molecule_record.get_topology_graph().__class__.__name__
    molecule = molecule_record.get_molecule()
    chemmform = get_molecule_formula(molecule)
    run_prefix = f"{tg}_{chemmform}"

    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")
    opt_xyz_file = os.path.join(output_dir, f"{run_prefix}_opted.xyz")
    opt_mol_file = os.path.join(output_dir, f"{run_prefix}_opted.mol")

    if os.path.exists(output_file):
        logging.info(f"loading {output_file}...")
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:

        # Does optimisation.
        logging.info(f": running optimisation of {run_prefix}")
        opt = FourPlusSixOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
        )
        run_data = opt.optimize(molecule)
        print(run_data)
        opted = molecule.with_structure_from_file(opt_xyz_file)
        opted.write(opt_mol_file)

        raise SystemExit()
        # Minimises energy, OH shape and targets pore size of 5 A.
        oh6_measure = ShapeMeasure(
            output_dir=(output_dir / f"{run_prefix}_shape"),
            target_atmnum=5,
            shape_string="oc6",
        ).calculate(opted)

        raise SystemExit()

        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        logging.info(f"{run_prefix}: {fin_energy} {fin_gnorm} ")
        opt_pore_data = calculate_pore(opt_xyz_file)
        print(opt_pore_data)
        res_dict = {
            "fin_energy": fin_energy,
            "opt_pore_data": opt_pore_data,
            "oh6_measure": oh6_measure,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    pore_radius = res_dict["opt_pore_data"]
    pore_size_diff = (5 - pore_radius * 2) / 5
    score = (
        res_dict["final_energy"]
        + res_dict["oh6_measure"] * 100
        + pore_size_diff * 100
    )
    print(score)
    return score


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = fourplussix_optimisation()
    calc_output = fourplussix_calculations()
    figure_output = fourplussix_figures()

    # Define list of topology functions.
    cage_topologies = cage_topology_options()

    # Define precursor topologies.
    three_precursor_topologies = three_precursor_topology_options()
    two_precursor_topologies = two_precursor_topology_options()

    logging.info("you need to figure out a hashing to avoid reruns.")

    # For now, just build N options and calculate properties.
    initial_population = tuple(
        get_initial_population(
            cage_topologies=cage_topologies,
            three_precursor_topologies=three_precursor_topologies,
            two_precursor_topologies=two_precursor_topologies,
            num_population=10,
        )
    )

    for i, mol in enumerate(initial_population):
        print(mol.get_molecule())
        print(get_fitness_value(mol, calc_output))

    raise SystemExit()

    def get_num_functional_groups(building_block):
        return building_block.get_num_functional_groups()

    def is_mono(building_block):
        return get_num_functional_groups(building_block) == 1

    def is_tri(building_block):
        return get_num_functional_groups(building_block) == 3

    generator = np.random.RandomState(4)
    mutator = stk.RandomMutator(
        mutators=(
            # Substitutes a monotopic building block with a
            # random monotopic building block.
            stk.RandomBuildingBlock(
                building_blocks=monotopics,
                is_replaceable=is_mono,
                random_seed=generator.randint(0, 1000),
            ),
            # Substitutes a tritopic building block with a
            # random tritopic building block.
            stk.RandomBuildingBlock(
                building_blocks=tritopics,
                is_replaceable=is_tri,
                random_seed=generator.randint(0, 1000),
            ),
        ),
        random_seed=generator.randint(0, 1000),
    )
    ea = stk.EvolutionaryAlgorithm(
        initial_population=initial_population,
        fitness_calculator=stk.FitnessFunction(get_fitness_value),
        mutator=mutator,
        crosser=stk.GeneticRecombination(
            get_gene=get_num_functional_groups,
        ),
        generation_selector=stk.Best(
            num_batches=5,
            duplicate_molecules=False,
        ),
        mutation_selector=stk.Roulette(
            num_batches=5,
            random_seed=generator.randint(0, 1000),
        ),
        crossover_selector=stk.Roulette(
            num_batches=3,
            batch_size=2,
            random_seed=generator.randint(0, 1000),
        ),
    )

    writer = stk.MolWriter()
    generations = []
    for i, generation in enumerate(ea.get_generations(10)):
        generations.append(generation)

        for molecule_id, molecule_record in enumerate(
            generation.get_molecule_records()
        ):
            writer.write(
                molecule=molecule_record.get_molecule(),
                path=f"g_{i}_m_{molecule_id}.mol",
            )

    fitness_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record: record.get_fitness_value(),
        y_label="Fitness Value",
    )
    fitness_progress.write(
        str(fourplussix_figures() / "fitness_progress.pdf")
    )

    raise SystemExit()

    # Make cage of each symmetry.
    topologies = topology_options()

    ff_options = {}

    bite_angles = np.arange(10, 181, 10)
    for ba in bite_angles:
        ff_options[f"ba{ba}"] = {
            "param_pool": {
                "bonds": {},
                "angles": {
                    ("B", "N", "N"): (20, ba),
                },
                "torsions": {},
                "pairs": {},
            },
            "notes": f"bite-angle change: {ba}, rigid",
            "name": f"ba{ba}",
        }

    results = {i: {} for i in ff_options}
    for topo_str in topologies:
        topology_graph = topologies[topo_str]
        cage = stk.ConstructedMolecule(topology_graph)
        cage.write(os.path.join(struct_output, f"{topo_str}_unopt.mol"))

        for ff_str in ff_options:
            res_dict = run_optimisation(
                cage=cage,
                ffname=ff_str,
                ff_modifications=ff_options[ff_str],
                topo_str=topo_str,
                output_dir=struct_output,
            )
            continue
            results[ff_str][topo_str] = res_dict

    topo_to_c = {
        "FourPlusSix": ("o", "k", 0),
        "FourPlusSix2": ("D", "r", 1),
    }

    convergence(
        results=results,
        output_dir=figure_output,
        filename="convergence.pdf",
    )

    ey_vs_shape(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="e_vs_shape.pdf",
    )

    geom_distributions(
        results=results,
        output_dir=figure_output,
        filename="dist.pdf",
    )

    heatmap(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="energy_map.pdf",
        vmin=0,
        vmax=20,
        clabel="energy (eV)",
    )

    heatmap(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="shape_map.pdf",
        vmin=0,
        vmax=2.2,
        clabel="OH-6",
    )

    scatter(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="energy.pdf",
        ylabel="energy (eV)",
    )
    scatter(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="shape.pdf",
        ylabel="OH-6",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
