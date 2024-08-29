# Distributed under the terms of the MIT License.

"""Module for cage generation utilities.

Author: Andrew Tarzia

"""

import itertools as it
import logging
import pathlib

import cgexplore
import openmm
import stk
from analysis import analyse_cage
from define_forcefields import get_neighbour_library

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def optimise_cage(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: cgexplore.forcefields.ForceField,
    platform: str,
    database: cgexplore.utilities.AtomliteDatabase,
) -> cgexplore.molecular.Conformer:
    """Optimise a toy model cage."""
    fina_mol_file = output_dir / f"{name}_final.mol"
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return cgexplore.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if fina_mol_file.exists():
        ensemble = cgexplore.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=output_dir / f"{name}_base.mol",
            conformer_xyz=output_dir / f"{name}_ensemble.xyz",
            data_json=output_dir / f"{name}_ensemble.json",
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

    ensemble = cgexplore.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemble.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
        overwrite=True,
    )
    temp_molecule = cgexplore.utilities.run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    try:
        logging.info("optimisation of %s", name)
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
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
    logging.info("optimisation of shifted structures of %s", name)
    for test_molecule in cgexplore.utilities.yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
    ):
        try:
            conformer = cgexplore.utilities.run_optimisation(
                assigned_system=cgexplore.forcefields.AssignedSystem(
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
    logging.info("optimisation of nearby structures of %s", name)
    neighbour_library = get_neighbour_library(
        ffnum=int(forcefield.get_identifier()),
        fftype=forcefield.get_prefix(),
    )
    for test_molecule in cgexplore.utilities.yield_near_models(
        molecule=molecule,
        name=name,
        output_dir=output_dir,
        neighbour_library=neighbour_library,
    ):
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
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

    logging.info("soft MD run of %s", name)
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = cgexplore.utilities.run_soft_md_cycle(
        name=name,
        assigned_system=cgexplore.forcefields.AssignedSystem(
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
        msg = f"!!!!! {name} MD exploded !!!!!"

        raise ValueError(msg)

    soft_md_data = soft_md_trajectory.get_data()

    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        msg = f"!!!!! {name} MD failed !!!!!"
        raise ValueError(msg)

    # Go through each conformer from soft MD.
    # Optimise them all.
    for md_conformer in soft_md_trajectory.yield_conformers():
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
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
        "Min. energy conformer: %s from %s with energy: %s kJ.mol-1",
        min_energy_conformerid,
        min_energy_conformer.source,
        min_energy,
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
    populations: dict[str, dict],
    struct_output: pathlib.Path,
    calculation_output: pathlib.Path,
    ligand_output: pathlib.Path,
    node_element: str,
    ligand_element: str,
    platform: str,
    database: cgexplore.utilities.AtomliteDatabase,
) -> None:
    """Build a population."""
    count = 0
    for population in populations:
        logging.info("running population %s", population)
        popn_dict = populations[population]
        topologies = popn_dict["topologies"]
        forcefields = tuple(popn_dict["fflibrary"].yield_forcefields())
        logging.info("there are %s topologies", len(topologies))
        logging.info("there are %s ffs", len(forcefields))
        logging.info("building %s cages", len(forcefields) * len(topologies))
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
            c2_building_block = cgexplore.utilities.optimise_ligand(
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
            cl_building_block = cgexplore.utilities.optimise_ligand(
                molecule=cl_precursor.get_building_block(),
                name=cl_name,
                output_dir=calculation_output,
                forcefield=forcefield,
                platform=None,
            )
            cl_building_block.write(str(ligand_output / f"{cl_name}_optl.mol"))

            logging.info("building %s", name)
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

        logging.info("%s %s cages built.", count, population)
