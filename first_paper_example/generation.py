#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for cage generation utilities.

Author: Andrew Tarzia

"""

import itertools
import json
import logging
import os

import openmm
import stk
from analysis import ligand_expected_topologies, node_expected_topologies
from cgexplore.assigned_system import AssignedSystem
from cgexplore.ensembles import Ensemble
from cgexplore.generation_utilities import (
    optimise_ligand,
    run_constrained_optimisation,
    run_optimisation,
    run_soft_md_cycle,
    yield_shifted_models,
)
from cgexplore.geom import GeomMeasure
from cgexplore.pore import PoreMeasure
from cgexplore.shape import (
    ShapeMeasure,
    get_shape_molecule_ligands,
    get_shape_molecule_nodes,
)
from env_set import shape_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def optimise_cage(
    molecule,
    name,
    output_dir,
    force_field,
    platform,
):
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")
    if os.path.exists(fina_mol_file):
        ensemble = Ensemble(
            base_molecule=molecule,
            base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
            conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
            data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
            overwrite=False,
        )
        return ensemble.get_lowest_e_conformer()

    assigned_system = force_field.assign_terms(molecule, name, output_dir)

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
                force_field_terms=assigned_system.force_field_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="opt1",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="opt1")
    except openmm.OpenMMException as error:
        if "Particle coordinate is NaN. " not in str(error):
            raise error

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in yield_shifted_models(
        temp_molecule, force_field, kicks=(1, 2, 3, 4)
    ):
        try:
            conformer = run_optimisation(
                assigned_system=AssignedSystem(
                    molecule=test_molecule,
                    force_field_terms=assigned_system.force_field_terms,
                    system_xml=assigned_system.system_xml,
                    topology_xml=assigned_system.topology_xml,
                    bead_set=assigned_system.bead_set,
                    vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
                ),
                name=name,
                file_suffix="sopt",
                output_dir=output_dir,
                # max_iterations=50,
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="shifted")
        except openmm.OpenMMException as error:
            if "Particle coordinate is NaN. " not in str(error):
                raise error

    # Collect and optimise structures nearby in phase space.
    # logging.info(f"optimisation of nearby structures of {name}")
    logging.info("optimisation of nearby structures turned off for now")
    # for test_molecule in yield_near_models(
    #     molecule=molecule,
    #     name=name,
    #     output_dir=output_dir,
    # ):
    #     conformer = run_optimisation(
    #         assigned_system=AssignedSystem(
    #             molecule=test_molecule,
    #             force_field_terms=assigned_system.force_field_terms,
    #         ),
    #         name=name,
    #         file_suffix="nopt",
    #         output_dir=output_dir,
    #         force_field=force_field,
    #         # max_iterations=50,
    #         platform=platform,
    #     )
    #     ensemble.add_conformer(conformer=conformer, source="nearby_opt")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = run_soft_md_cycle(
        name=name,
        assigned_system=AssignedSystem(
            molecule=ensemble.get_lowest_e_conformer().molecule,
            force_field_terms=assigned_system.force_field_terms,
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
        # md_exploded = True
        raise ValueError("OpenMM Exception")

    soft_md_data = soft_md_trajectory.get_data()
    logging.info(f"collected trajectory {len(soft_md_data)} confs long")
    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        # md_failed = True
        raise ValueError()

    # Go through each conformer from soft MD.
    # Optimise them all.
    for md_conformer in soft_md_trajectory.yield_conformers():
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
                molecule=md_conformer.molecule,
                force_field_terms=assigned_system.force_field_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            # max_iterations=50,
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

    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


def analyse_cage(
    conformer,
    name,
    output_dir,
    custom_torsion_options,
    node_element,
    ligand_element,
):
    output_file = os.path.join(output_dir, f"{name}_res.json")
    shape_molfile1 = os.path.join(output_dir, f"{name}_shape1.mol")
    shape_molfile2 = os.path.join(output_dir, f"{name}_shape2.mol")

    if not os.path.exists(output_file):
        logging.info(f"analysing {name}")

        energy_decomp = {
            f"{i}_{conformer.energy_decomposition[i][1]}": float(
                conformer.energy_decomposition[i][0]
            )
            for i in conformer.energy_decomposition
        }
        fin_energy = energy_decomp["total energy_kJ/mol"]

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
        try:
            target_torsions = custom_torsion_options["ton"]
        except KeyError:
            target_torsions = None
        assert len(target_torsions) == 1
        g_measure = GeomMeasure(target_torsions)
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

        res_dict = {
            "optimised": True,
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
            # "trajectory": trajectory_data,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)


def build_populations(
    populations,
    struct_output,
    calculation_output,
    ligand_output,
    node_element,
    ligand_element,
    platform,
):
    count = 0
    for population in populations:
        logging.info(f"running population {population}")
        popn_dict = populations[population]
        popn_iterator = itertools.product(
            popn_dict["topologies"],
            tuple(popn_dict["fflibrary"].yield_forcefields()),
        )
        for cage_topo_str, force_field in popn_iterator:
            c2_precursor = popn_dict["c2"]
            cl_precursor = popn_dict["cl"]
            name = (
                f"{cage_topo_str}_{cl_precursor.get_name()}_"
                f"{c2_precursor.get_name()}_"
                f"f{force_field.get_identifier()}"
            )

            # Write out force field.
            force_field.write_human_readable(calculation_output)

            # Optimise building blocks.
            c2_name = (
                f"{c2_precursor.get_name()}_f{force_field.get_identifier()}"
            )
            c2_building_block = optimise_ligand(
                molecule=c2_precursor.get_building_block(),
                name=c2_name,
                output_dir=calculation_output,
                force_field=force_field,
                platform=None,
            )
            c2_building_block.write(str(ligand_output / f"{c2_name}_optl.mol"))

            cl_name = (
                f"{cl_precursor.get_name()}_f{force_field.get_identifier()}"
            )
            cl_building_block = optimise_ligand(
                molecule=cl_precursor.get_building_block(),
                name=cl_name,
                output_dir=calculation_output,
                force_field=force_field,
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
                force_field=force_field,
                platform=platform,
            )
            if conformer is not None:
                conformer.molecule.write(
                    str(struct_output / f"{name}_optc.mol")
                )
            logging.info("skipping analysis right now.")
            continue

            analyse_cage(
                conformer=conformer,
                name=name,
                output_dir=calculation_output,
                node_element=node_element,
                ligand_element=ligand_element,
            )
            count += 1

        logging.info(f"{count} {population} cages built.")
