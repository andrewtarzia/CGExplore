#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for cage generation utilities.

Author: Andrew Tarzia

"""

import itertools as it
import logging
import os

import openmm
import stk
from analysis import analyse_cage
from cgexplore.assigned_system import AssignedSystem
from cgexplore.ensembles import Conformer, Ensemble
from cgexplore.generation_utilities import (
    optimise_ligand,
    run_constrained_optimisation,
    run_optimisation,
    run_soft_md_cycle,
    yield_near_models,
    yield_shifted_models,
)
from define_forcefields import get_neighbour_library

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def optimise_cage(
    molecule,
    name,
    output_dir,
    forcefield,
    platform,
    database,
):
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if os.path.exists(fina_mol_file):
        ensemble = Ensemble(
            base_molecule=molecule,
            base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
            conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
            data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
            overwrite=False,
        )
        conformer = ensemble.get_lowest_e_conformer()
        database.add_molecule(molecule=conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": conformer.energy_decomposition,
                "source": conformer.source,
                "optimised": True,
            },
        )
        return ensemble.get_lowest_e_conformer()

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)

    ensemble = Ensemble(
        base_molecule=molecule,
        base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
        conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
        data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
        overwrite=True,
    )
    temp_molecule = run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    try:
        logging.info(f"optimisation of {name}")
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
                molecule=temp_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="opt1",
            output_dir=output_dir,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="opt1")
    except openmm.OpenMMException as error:
        if "Particle coordinate is NaN. " not in str(error):
            raise

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
    ):
        try:
            conformer = run_optimisation(
                assigned_system=AssignedSystem(
                    molecule=test_molecule,
                    forcefield_terms=assigned_system.forcefield_terms,
                    system_xml=assigned_system.system_xml,
                    topology_xml=assigned_system.topology_xml,
                    bead_set=assigned_system.bead_set,
                    vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
                ),
                name=name,
                file_suffix="sopt",
                output_dir=output_dir,
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="shifted")
        except openmm.OpenMMException as error:
            if "Particle coordinate is NaN. " not in str(error):
                raise

    # Collect and optimise structures nearby in phase space.
    logging.info(f"optimisation of nearby structures of {name}")
    neighbour_library = get_neighbour_library(
        ffnum=int(forcefield.get_identifier()),
        fftype=forcefield.get_prefix(),
    )
    for test_molecule in yield_near_models(
        molecule=molecule,
        name=name,
        output_dir=output_dir,
        neighbour_library=neighbour_library,
    ):
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
                molecule=test_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="nopt",
            output_dir=output_dir,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="nearby_opt")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = run_soft_md_cycle(
        name=name,
        assigned_system=AssignedSystem(
            molecule=ensemble.get_lowest_e_conformer().molecule,
            forcefield_terms=assigned_system.forcefield_terms,
            system_xml=assigned_system.system_xml,
            topology_xml=assigned_system.topology_xml,
            bead_set=assigned_system.bead_set,
            vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
        ),
        output_dir=output_dir,
        suffix="smd",
        bond_ff_scale=10,
        angle_ff_scale=10,
        temperature=300 * openmm.unit.kelvin,
        num_steps=num_steps,
        time_step=0.5 * openmm.unit.femtoseconds,
        friction=1.0 / openmm.unit.picosecond,
        reporting_freq=traj_freq,
        traj_freq=traj_freq,
        platform=platform,
    )
    if soft_md_trajectory is None:
        logging.info(f"!!!!! {name} MD exploded !!!!!")
        msg = "OpenMM Exception"
        raise ValueError(msg)

    soft_md_data = soft_md_trajectory.get_data()
    logging.info(f"collected trajectory {len(soft_md_data)} confs long")
    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        raise ValueError

    # Go through each conformer from soft MD.
    # Optimise them all.
    for md_conformer in soft_md_trajectory.yield_conformers():
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
                molecule=md_conformer.molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="smd")
    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        f"Min. energy conformer: {min_energy_conformerid} from "
        f"{min_energy_conformer.source}"
        f" with energy: {min_energy} kJ.mol-1"
    )

    # Add to atomlite database.
    database.add_molecule(molecule=min_energy_conformer.molecule, key=name)
    database.add_properties(
        key=name,
        property_dict={
            "energy_decomposition": min_energy_conformer.energy_decomposition,
            "source": min_energy_conformer.source,
            "optimised": True,
        },
    )
    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


def build_populations(
    populations,
    struct_output,
    calculation_output,
    ligand_output,
    node_element,
    ligand_element,
    platform,
    database,
):
    count = 0
    for population in populations:
        logging.info(f"running population {population}")
        popn_dict = populations[population]
        topologies = popn_dict["topologies"]
        forcefields = tuple(popn_dict["fflibrary"].yield_forcefields())
        logging.info(f"there are {len(topologies)} topologies")
        logging.info(f"there are {len(forcefields)} ffs")
        logging.info(f"building {len(forcefields) * len(topologies)} cages")
        popn_iterator = it.product(topologies, forcefields)
        for cage_topo_str, forcefield in popn_iterator:
            c2_precursor = popn_dict["c2"]
            cl_precursor = popn_dict["cl"]
            name = (
                f"{cage_topo_str}_{cl_precursor.get_name()}_"
                f"{c2_precursor.get_name()}_"
                f"f{forcefield.get_identifier()}"
            )

            # Write out force field.
            forcefield.write_human_readable(calculation_output)

            # Optimise building blocks.
            c2_name = (
                f"{c2_precursor.get_name()}_f{forcefield.get_identifier()}"
            )
            c2_building_block = optimise_ligand(
                molecule=c2_precursor.get_building_block(),
                name=c2_name,
                output_dir=calculation_output,
                forcefield=forcefield,
                platform=None,
            )
            c2_building_block.write(str(ligand_output / f"{c2_name}_optl.mol"))

            cl_name = (
                f"{cl_precursor.get_name()}_f{forcefield.get_identifier()}"
            )
            cl_building_block = optimise_ligand(
                molecule=cl_precursor.get_building_block(),
                name=cl_name,
                output_dir=calculation_output,
                forcefield=forcefield,
                platform=None,
            )
            cl_building_block.write(str(ligand_output / f"{cl_name}_optl.mol"))

            logging.info(f"building {name}")
            cage = stk.ConstructedMolecule(
                topology_graph=popn_dict["topologies"][cage_topo_str](
                    building_blocks=(c2_building_block, cl_building_block),
                ),
            )

            conformer = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                forcefield=forcefield,
                platform=platform,
                database=database,
            )
            if conformer is not None:
                conformer.molecule.write(
                    str(struct_output / f"{name}_optc.mol")
                )
            count += 1

            analyse_cage(
                conformer=conformer,
                name=name,
                output_dir=calculation_output,
                forcefield=forcefield,
                node_element=node_element,
                ligand_element=ligand_element,
                database=database,
            )

        logging.info(f"{count} {population} cages built.")
