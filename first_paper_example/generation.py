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

import stk
from analysis import (
    ligand_expected_topologies,
    node_expected_topologies,
)
from cgexplore.ensembles import Ensemble
from cgexplore.generation_utilities import (
    run_constrained_optimisation,
    run_optimisation,
    run_soft_md_cycle,
    yield_near_models,
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
from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def custom_torsion_definitions(population):
    return {
        "2p3": {
            "ton": (180, 50),
            "toff": None,
        },
        "2p4": {
            "ton": (180, 50),
            "toff": None,
        },
        "3p4": {"toff": None},
    }[population]


def custom_vdw_definitions(population):
    return {
        "2p3": {
            "von": True,
            # "voff": False,
        },
        "2p4": {
            "von": True,
            # "voff": False,
        },
        "3p4": {
            "von": True,
            # "voff": False,
        },
    }[population]


def optimise_cage(
    molecule,
    name,
    output_dir,
    bead_set,
    custom_torsion_set,
    custom_vdw_set,
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

    ensemble = Ensemble(
        base_molecule=molecule,
        base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
        conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
        data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
        overwrite=True,
    )

    molecule = run_constrained_optimisation(
        molecule=molecule,
        bead_set=bead_set,
        name=name,
        output_dir=output_dir,
        custom_vdw_set=custom_vdw_set,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    logging.info(f"optimisation of {name}")
    conformer = run_optimisation(
        molecule=molecule,
        bead_set=bead_set,
        name=name,
        file_suffix="opt1",
        output_dir=output_dir,
        custom_vdw_set=custom_vdw_set,
        custom_torsion_set=custom_torsion_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw_bond_cutoff=2,
        # max_iterations=50,
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in yield_shifted_models(molecule, bead_set):
        conformer = run_optimisation(
            molecule=test_molecule,
            bead_set=bead_set,
            name=name,
            file_suffix="sopt",
            output_dir=output_dir,
            custom_vdw_set=custom_vdw_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw_bond_cutoff=2,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

    # Collect and optimise structures nearby in phase space.
    logging.info(f"optimisation of nearby structures of {name}")
    for test_molecule in yield_near_models(
        molecule=molecule,
        name=name,
        bead_set=bead_set,
        output_dir=output_dir,
    ):
        conformer = run_optimisation(
            molecule=test_molecule,
            bead_set=bead_set,
            name=name,
            file_suffix="nopt",
            output_dir=output_dir,
            custom_vdw_set=custom_vdw_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw_bond_cutoff=2,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="nearby_opt")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = run_soft_md_cycle(
        name=name,
        molecule=ensemble.get_lowest_e_conformer().molecule,
        bead_set=bead_set,
        output_dir=output_dir,
        custom_vdw_set=custom_vdw_set,
        custom_torsion_set=None,
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
            molecule=md_conformer.molecule,
            bead_set=bead_set,
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            custom_vdw_set=custom_vdw_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw_bond_cutoff=2,
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
    bead_set,
    custom_torsion_set,
    custom_vdw_set,
    node_element,
    ligand_element,
):
    output_file = os.path.join(output_dir, f"{name}_res.json")
    shape_molfile1 = os.path.join(output_dir, f"{name}_shape1.mol")
    shape_molfile2 = os.path.join(output_dir, f"{name}_shape2.mol")

    if not os.path.exists(output_file):
        logging.info(f"analysing {name}")

        # Always want to extract target torions.
        temp_custom_torsion_set = target_torsions(
            bead_set=bead_set,
            custom_torsion_option=None,
        )
        if temp_custom_torsion_set is None:
            custom_torsion_atoms = None
        else:
            custom_torsion_atoms = [
                bead_set[j].element_string
                for i in temp_custom_torsion_set
                for j in i
            ]

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
                target_atmnums=None,
                shape_string=None,
            ).calculate(n_shape_mol)

        if l_shape_mol is None:
            lig_shape_measures = None
        else:
            lig_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_lshape"),
                shape_path=shape_path(),
                target_atmnums=None,
                shape_string=None,
            ).calculate(l_shape_mol)
            l_shape_mol.write(shape_molfile2)

        opt_pore_data = PoreMeasure().calculate_min_distance(
            conformer.molecule
        )

        g_measure = GeomMeasure(custom_torsion_atoms)
        bond_data = g_measure.calculate_bonds(conformer.molecule)
        angle_data = g_measure.calculate_angles(conformer.molecule)
        dihedral_data = g_measure.calculate_torsions(
            molecule=conformer.molecule,
            absolute=True,
            path_length=5,
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


def target_torsions(bead_set, custom_torsion_option):
    try:
        (t_key_1,) = (i for i in bead_set if i[0] == "a")
    except ValueError:
        # For when 3+4 cages are being built - there are no target
        # torsions.
        return None

    (c_key,) = (i for i in bead_set if i[0] == "c")
    (t_key_2,) = (i for i in bead_set if i[0] == "b")
    custom_torsion_set = {
        (
            t_key_2,
            t_key_1,
            c_key,
            t_key_1,
            t_key_2,
        ): custom_torsion_option,
    }
    return custom_torsion_set


def collect_custom_torsion(
    bb2_bead_set,
    custom_torsion_options,
    custom_torsion,
    bead_set,
):
    if custom_torsion_options[custom_torsion] is None:
        custom_torsion_set = None
    else:
        tors_option = custom_torsion_options[custom_torsion]
        custom_torsion_set = target_torsions(
            bead_set=bead_set,
            custom_torsion_option=tors_option,
        )

    return custom_torsion_set


def build_populations(
    populations,
    custom_torsion_options,
    custom_vdw_options,
    struct_output,
    calculation_output,
    node_element,
    ligand_element,
):
    num_runs = 1
    for popn in populations:
        logging.info(f"building {popn} population")
        popn_iterator = itertools.product(
            populations[popn]["t"],
            populations[popn]["c2"],
            populations[popn]["cl"],
            custom_torsion_options,
            custom_vdw_options,
        )
        count = 0
        for iteration in popn_iterator:
            (
                cage_topo_str,
                bb2_str,
                bbl_str,
                custom_torsion,
                custom_vdw,
            ) = iteration

            bb2, bb2_bead_set = populations[popn]["c2"][bb2_str]
            bbl, bbl_bead_set = populations[popn]["cl"][bbl_str]

            bead_set = bb2_bead_set.copy()
            bead_set.update(bbl_bead_set)

            custom_torsion_set = collect_custom_torsion(
                bb2_bead_set=bb2_bead_set,
                custom_torsion_options=(custom_torsion_options),
                custom_torsion=custom_torsion,
                bead_set=bead_set,
            )

            custom_vdw_set = custom_vdw_options[custom_vdw]

            for run in range(num_runs):
                name = (
                    f"{cage_topo_str}_{bbl_str}_{bb2_str}_"
                    f"{custom_torsion}_{custom_vdw}_{run}"
                )

                logging.info(f"building {name}")
                cage = stk.ConstructedMolecule(
                    topology_graph=populations[popn]["t"][cage_topo_str](
                        building_blocks=(bb2, bbl),
                    ),
                )

                conformer = optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_output,
                    bead_set=bead_set,
                    custom_torsion_set=custom_torsion_set,
                    custom_vdw_set=custom_vdw_set,
                    platform=None,
                )

                if conformer is not None:
                    conformer.molecule.write(
                        str(struct_output / f"{name}_optc.mol")
                    )

                analyse_cage(
                    conformer=conformer,
                    name=name,
                    output_dir=calculation_output,
                    bead_set=bead_set,
                    custom_torsion_set=custom_torsion_set,
                    custom_vdw_set=custom_vdw_set,
                    node_element=node_element,
                    ligand_element=ligand_element,
                )
            count += 1

        logging.info(f"{count} {popn} cages built.")
