#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for cage generation utilities.

Author: Andrew Tarzia

"""

import stk
import os
import json
import logging
import numpy as np
import itertools

from dataclasses import replace
from openmm import openmm, OpenMMException

from shape import (
    ShapeMeasure,
    get_shape_molecule_ligands,
    get_shape_molecule_nodes,
)
from geom import GeomMeasure
from pore import PoreMeasure
from utilities import check_long_distances
from openmm_optimizer import (
    CGOMMOptimizer,
    CGOMMDynamics,
)
from beads import produce_bead_library


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


def bond_k():
    return 1e5


def angle_k():
    return 1e2


def core_2c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        angles=(180,),
        bond_rs=(2,),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def arm_2c_beads():
    return produce_bead_library(
        type_prefix="a",
        element_string="Ba",
        bond_rs=(1,),
        angles=range(90, 181, 5),
        # angles=(90, 100, 120, 140, 160, 180),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def binder_beads():
    return produce_bead_library(
        type_prefix="b",
        element_string="Pb",
        bond_rs=(1,),
        angles=(180,),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def beads_3c():
    return produce_bead_library(
        type_prefix="n",
        element_string="C",
        bond_rs=(2,),
        angles=(50, 60, 70, 80, 90, 100, 110, 120),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=3,
    )


def beads_4c():
    return produce_bead_library(
        type_prefix="m",
        element_string="Pd",
        bond_rs=(2,),
        angles=(50, 60, 70, 80, 90),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=4,
    )


def optimise_ligand(molecule, name, output_dir, bead_set):

    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")

    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}, no max_iterations")
        opt = CGOMMOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
            # max_iterations=1000,
        )
        molecule = opt.optimize(molecule)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)

    return molecule


def deform_molecule(molecule, generator, kick):
    old_pos_mat = molecule.get_position_matrix()
    centroid = molecule.get_centroid()

    new_pos_mat = []
    for atom, pos in zip(molecule.get_atoms(), old_pos_mat):
        if atom.get_atomic_number() in (6, 46):
            c_v = centroid - pos
            c_v = c_v / np.linalg.norm(c_v)
            # move = generator.choice([-1, 1]) * c_v
            # move = generator.random((3,)) * percent
            move = c_v * kick * generator.random()
            new_pos = pos - move
        else:
            # move = generator.random((3,)) * 2
            new_pos = pos
        new_pos_mat.append(new_pos)

    return molecule.with_position_matrix(np.array((new_pos_mat)))


def deform_and_optimisations(
    name,
    molecule,
    opt,
    kick,
    seed,
    num_deformations=None,
):
    """
    Run metropolis MC scheme to find a new lower energy conformer.

    """

    # Ensure same stream of random numbers.
    generator = np.random.default_rng(seed=seed)
    beta = 10

    if num_deformations is None:
        num_iterations = 200
    else:
        num_iterations = num_deformations

    num_passed = 0
    num_run = 0
    for drun in range(num_iterations):
        # logging.info(f"running MC step {drun}")

        current_energy = opt.calculate_energy(molecule).value_in_unit(
            openmm.unit.kilojoules_per_mole
        )

        # Perform deformation.
        test_molecule = deform_molecule(molecule, generator, kick)
        test_molecule = opt.optimize(test_molecule)

        # Calculate energy.
        test_energy = opt.calculate_energy(test_molecule).value_in_unit(
            openmm.unit.kilojoules_per_mole
        )

        passed = False
        if test_energy < current_energy:
            passed = True
        elif (
            np.exp(-beta * (test_energy - current_energy))
            > generator.random()
        ):
            passed = True

        # Pass or fail.
        if passed:
            # logging.info(
            #     f"new conformer (MC): "
            #     f"{test_energy}, cf. {current_energy}"
            # )
            num_passed += 1
            molecule = test_molecule.clone().with_centroid((0, 0, 0))
            if num_deformations is None:
                return molecule

        num_run += 1

    logging.info(
        f"{num_passed} passed out of {num_run} ({num_passed/num_run})"
    )

    if num_deformations is None:
        raise RuntimeError(
            f"no lower energy conformers found in {num_iterations}"
        )

    return molecule


def run_md_cycle(
    name,
    molecule,
    md_class,
    expected_num_steps,
    opt_class=None,
    min_energy=None,
):
    failed = False
    exploded = False
    try:
        trajectory = md_class.run_dynamics(molecule)
        traj_log = trajectory.get_data()

        # Check that the trajectory is as long as it should be.
        num_conformers = len(traj_log)
        if num_conformers != expected_num_steps:
            raise ValueError()

        if min_energy is None:
            min_energy = md_class.calculate_energy(
                molecule
            ).value_in_unit(openmm.unit.kilojoules_per_mole)

        for conformer in trajectory.yield_conformers():
            if opt_class is None:
                conformer_energy = float(
                    traj_log[traj_log['#"Step"'] == conformer.timestep][
                        "Potential Energy (kJ/mole)"
                    ]
                )
            else:
                opt_conformer = opt_class.optimize(conformer.molecule)
                conformer_energy = opt_class.calculate_energy(
                    opt_conformer
                ).value_in_unit(openmm.unit.kilojoules_per_mole)

            if conformer_energy < min_energy:
                logging.info(
                    f"new low. E conformer (MD): "
                    f"{conformer_energy}, cf. {min_energy}"
                )
                min_energy = conformer_energy
                if opt_class is None:
                    molecule = molecule.with_position_matrix(
                        conformer.molecule.get_position_matrix(),
                    )
                else:
                    molecule = molecule.with_position_matrix(
                        opt_conformer.get_position_matrix(),
                    )

        molecule = molecule.with_centroid((0, 0, 0))

    except ValueError:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        failed = True
    except OpenMMException:
        logging.info(f"!!!!! {name} MD exploded !!!!!")
        exploded = True

    return molecule, failed, exploded


def optimise_cage(
    molecule,
    name,
    output_dir,
    bead_set,
    custom_torsion_set,
    custom_vdw_set,
    run_seed,
):

    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{name}_opted2.mol")
    opt2_fai_file = os.path.join(output_dir, f"{name}_mdfailed.txt")
    opt2_exp_file = os.path.join(output_dir, f"{name}_mdexploded.txt")
    opt3_mol_file = os.path.join(output_dir, f"{name}_opted3.mol")
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")

    if os.path.exists(fina_mol_file):
        return molecule.with_structure_from_file(fina_mol_file)

    soft_bead_set = {}
    for i in bead_set:
        new_bead = replace(bead_set[i])
        new_bead.bond_k = bead_set[i].bond_k / 10
        new_bead.angle_k = bead_set[i].angle_k / 10
        soft_bead_set[i] = new_bead

    # if custom_torsion_set is None:
    #     new_custom_torsion_set = None
    # else:
    #     new_custom_torsion_set = {
    #         i: (
    #             custom_torsion_set[i][0],
    #             custom_torsion_set[i][1] / 10,
    #         )
    #         for i in custom_torsion_set
    #     }

    intra_bb_bonds = []
    for bond_info in molecule.get_bond_infos():
        if bond_info.get_building_block_id() is not None:
            bond = bond_info.get_bond()
            intra_bb_bonds.append(
                (bond.get_atom1().get_id(), bond.get_atom2().get_id())
            )

    constrained_opt = CGOMMOptimizer(
        fileprefix=f"{name}_o1",
        output_dir=output_dir,
        param_pool=soft_bead_set,
        custom_torsion_set=None,
        bonds=True,
        angles=False,
        torsions=False,
        vdw=custom_vdw_set,
        max_iterations=10,
        vdw_bond_cutoff=2,
        atom_constraints=intra_bb_bonds,
    )
    opt = CGOMMOptimizer(
        fileprefix=f"{name}_o1d",
        output_dir=output_dir,
        param_pool=bead_set,
        custom_torsion_set=custom_torsion_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw=custom_vdw_set,
        max_iterations=50,
        vdw_bond_cutoff=2,
    )
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(
            f"optimising {name} with {len(intra_bb_bonds)} constraints"
        )
        molecule = constrained_opt.optimize(molecule)
        molecule = opt.optimize(molecule)
        # Run an optimisation on deformed structures and use that if
        # lower in energy.
        molecule = deform_and_optimisations(
            name=name,
            molecule=molecule,
            opt=opt,
            kick=3,
            num_deformations=50,
            seed=run_seed,
        )
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)

    opt_energy = opt.calculate_energy(molecule)

    if os.path.exists(opt2_mol_file):
        molecule = molecule.with_structure_from_file(opt2_mol_file)
    else:
        mdopt = CGOMMOptimizer(
            fileprefix=f"{name}_o2opt",
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=custom_vdw_set,
            # max_iterations=50,
            vdw_bond_cutoff=2,
        )

        soft_num_steps = 100
        soft_traj_freq = 10
        softmd = CGOMMDynamics(
            fileprefix=f"{name}_o2soft",
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=custom_vdw_set,
            # max_iterations=1000,
            vdw_bond_cutoff=2,
            temperature=10 * openmm.unit.kelvin,
            random_seed=run_seed,
            num_steps=soft_num_steps,
            time_step=0.1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=10,
            traj_freq=soft_traj_freq,
        )
        logging.info(f"running soft MD {name}")
        molecule, failed, exploded = run_md_cycle(
            name=name,
            molecule=molecule,
            md_class=softmd,
            expected_num_steps=soft_num_steps / soft_traj_freq,
            opt_class=None,
        )

        num_steps = 1000
        traj_freq = 100
        md = CGOMMDynamics(
            fileprefix=f"{name}_o2",
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=custom_vdw_set,
            # max_iterations=1000,
            vdw_bond_cutoff=2,
            temperature=300 * openmm.unit.kelvin,
            random_seed=run_seed,
            num_steps=num_steps,
            time_step=2 * openmm.unit.femtoseconds,
            friction=10.0 / openmm.unit.picosecond,
            reporting_freq=traj_freq,
            traj_freq=traj_freq,
        )
        logging.info(f"running MD {name}")
        molecule, failed, exploded = run_md_cycle(
            name=name,
            molecule=molecule,
            md_class=md,
            expected_num_steps=num_steps / traj_freq,
            opt_class=mdopt,
        )

        if failed or exploded:
            # Do a run of deformations and try again.
            molecule = deform_and_optimisations(
                name=name,
                molecule=molecule,
                opt=mdopt,
                kick=4,
                num_deformations=50,
                seed=run_seed,
            )
            molecule, failed, exploded = run_md_cycle(
                name=name,
                molecule=molecule,
                md_class=md,
                expected_num_steps=num_steps / traj_freq,
                opt_class=mdopt,
            )

        # Only try once, then accept defeat.
        if failed:
            with open(opt2_exp_file, "w") as f:
                f.write("exploded.")
        if exploded:
            with open(opt2_fai_file, "w") as f:
                f.write("failed.")

        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt2_mol_file)

    opt = CGOMMOptimizer(
        fileprefix=f"{name}_o3",
        output_dir=output_dir,
        param_pool=bead_set,
        custom_torsion_set=custom_torsion_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw=custom_vdw_set,
        # max_iterations=1000,
        vdw_bond_cutoff=2,
    )
    if os.path.exists(opt3_mol_file):
        molecule = molecule.with_structure_from_file(opt3_mol_file)
    else:
        molecule = opt.optimize(molecule)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt3_mol_file)
    md_opt_energy = opt.calculate_energy(molecule)

    try:
        check_long_distances(
            molecule,
            name=name,
            max_distance=15,
            step=1,
        )
    except ValueError:
        logging.info(f"{name} opt failed in step 1. Should be ignored.")
        raise SystemExit()
        return None

    if opt_energy < md_opt_energy:
        logging.info(
            "energy after first optimisation < energy after MD "
            " and optimisation."
        )
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    molecule.write(fina_mol_file)
    return molecule


def analyse_cage(
    molecule,
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

    if molecule is None:
        res_dict = {"optimised": False}
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f"analysing {name}")

        mdexploded = False
        mdfailed = False

        opt = CGOMMOptimizer(
            fileprefix=f"{name}_ey",
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=custom_vdw_set,
            # max_iterations=1000,
            vdw_bond_cutoff=2,
        )
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

        energy_decomp = opt.calculate_energy_decomposed(molecule)
        energy_decomp = {
            f"{i}_kjmol": energy_decomp[i].value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            for i in energy_decomp
        }
        fin_energy = energy_decomp["tot_energy_kjmol"]

        n_shape_mol = get_shape_molecule_nodes(
            molecule,
            name,
            node_element,
        )
        l_shape_mol = get_shape_molecule_ligands(
            molecule,
            name,
            ligand_element,
        )
        if n_shape_mol is None:
            node_shape_measures = None
        else:
            n_shape_mol.write(shape_molfile1)
            node_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_nshape"),
                target_atmnums=None,
                shape_string=None,
            ).calculate(n_shape_mol)

        if l_shape_mol is None:
            lig_shape_measures = None
        else:
            lig_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_lshape"),
                target_atmnums=None,
                shape_string=None,
            ).calculate(l_shape_mol)
            l_shape_mol.write(shape_molfile2)

        opt_pore_data = PoreMeasure().calculate_min_distance(molecule)

        g_measure = GeomMeasure(custom_torsion_atoms)
        bond_data = g_measure.calculate_bonds(molecule)
        angle_data = g_measure.calculate_angles(molecule)
        dihedral_data = g_measure.calculate_torsions(molecule)
        min_b2b_distance = g_measure.calculate_minb2b(molecule)
        radius_gyration = g_measure.calculate_radius_gyration(molecule)
        max_diameter = g_measure.calculate_max_diameter(molecule)
        if radius_gyration > max_diameter:
            raise ValueError(
                f"{name} Rg ({radius_gyration}) > maxD ({max_diameter})"
            )

        failed_file = output_dir / f"{name}_mdfailed.txt"
        exploded_file = output_dir / f"{name}_mdexploded.txt"
        if os.path.exists(failed_file):
            mdfailed = True
        if os.path.exists(exploded_file):
            mdexploded = True
        # if mdfailed or mdexploded:
        #     trajectory_data = None
        # else:
        #     fileprefix = f"{name}_o2"
        #     trajectory = OMMTrajectory(
        #         base_molecule=molecule,
        #         data_path=output_dir / f"{fileprefix}_traj.dat",
        #         traj_path=output_dir / f"{fileprefix}_traj.pdb",
        #         forcefield_path=output_dir / f"{fileprefix}_ff.xml",
        #         output_path=output_dir / f"{fileprefix}_omm.out",
        #         temperature=300 * openmm.unit.kelvin,
        #         random_seed=1000,
        #         num_steps=10000,
        #         time_step=1 * openmm.unit.femtoseconds,
        #         friction=1.0 / openmm.unit.picosecond,
        #         reporting_freq=100,
        #         traj_freq=100,
        #     )
        #     traj_log = trajectory.get_data()
        #     trajectory_data = {}
        #     for conformer in trajectory.yield_conformers():
        #         timestep = conformer.timestep
        #         time_ps = timestep / 1e3
        #         row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
        #         pot_energy = float(row["Potential Energy (kJ/mole)"])
        #         kin_energy = float(row["Kinetic Energy (kJ/mole)"])
        #         conf_temp = float(row["Temperature (K)"])

        #         conf_energy_decomp = opt.calculate_energy_decomposed(
        #             conformer.molecule
        #         )
        #         conf_energy_decomp = {
        #             f"{i}_kjmol": conf_energy_decomp[i].value_in_unit(
        #                 openmm.unit.kilojoules_per_mole
        #             )
        #             for i in conf_energy_decomp
        #         }

        #         c_n_shape_mol = get_shape_molecule_nodes(
        #             conformer.molecule,
        #             name,
        #         )
        #         c_l_shape_mol = get_shape_molecule_ligands(
        #             conformer.molecule,
        #             name,
        #         )
        #         if c_n_shape_mol is None:
        #             conf_node_shape_measures = None
        #         else:
        #             conf_node_shape_measures = ShapeMeasure(
        #                 output_dir=(
        #                     output_dir / f"{name}_{timestep}_nshape"
        #                 ),
        #                 target_atmnums=None,
        #                 shape_string=None,
        #             ).calculate(c_n_shape_mol)

        #         if l_shape_mol is None:
        #             conf_lig_shape_measures = None
        #         else:
        #             conf_lig_shape_measures = ShapeMeasure(
        #                 output_dir=(
        #                     output_dir / f"{name}_{timestep}_lshape"
        #                 ),
        #                 target_atmnums=None,
        #                 shape_string=None,
        #             ).calculate(c_l_shape_mol)

        #         conf_pore_data = PoreMeasure().calculate_min_distance(
        #             conformer.molecule,
        #         )

        #         g_measure = GeomMeasure(custom_torsion_atoms)
        #         conf_bond_data = g_measure.calculate_bonds(
        #             conformer.molecule
        #         )
        #         conf_angle_data = g_measure.calculate_angles(
        #             conformer.molecule
        #         )
        #         conf_dihedral_data = g_measure.calculate_torsions(
        #             conformer.molecule
        #         )
        #         conf_min_b2b_distance = g_measure.calculate_minb2b(
        #             conformer.molecule
        #         )
        #         conf_radius_gyration = (
        #             g_measure.calculate_radius_gyration(
        #                 conformer.molecule
        #             )
        #         )
        #         conf_max_diameter = g_measure.calculate_max_diameter(
        #             conformer.molecule
        #         )
        #         if radius_gyration > max_diameter:
        #             raise ValueError(
        #                 f"{name} Rg ({radius_gyration}) > "
        #                 f"max D ({max_diameter})"
        #             )
        #         trajectory_data[timestep] = {
        #             "time_ps": time_ps,
        #             "pot_energy_kjmol": pot_energy,
        #             "kin_energy_kjmol": kin_energy,
        #             "temperature_K": conf_temp,
        #             "energy_decomp": conf_energy_decomp,
        #             "pore_data": conf_pore_data,
        #             "lig_shape_measures": conf_lig_shape_measures,
        #             "node_shape_measures": conf_node_shape_measures,
        #             "bond_data": conf_bond_data,
        #             "angle_data": conf_angle_data,
        #             "dihedral_data": conf_dihedral_data,
        #             "min_b2b_distance": conf_min_b2b_distance,
        #             "radius_gyration": conf_radius_gyration,
        #             "max_diameter": conf_max_diameter,
        #         }

        res_dict = {
            "optimised": True,
            "mdexploded": mdexploded,
            "mdfailed": mdfailed,
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

    return res_dict


def build_building_block(
    topology,
    option1_lib,
    option2_lib,
    full_bead_library,
    calculation_output,
    ligand_output,
):
    blocks = {}
    for options in itertools.product(option1_lib, option2_lib):
        option1 = option1_lib[options[0]]
        option2 = option2_lib[options[1]]
        temp = topology(bead=option1, abead1=option2)

        opt_bb = optimise_ligand(
            molecule=temp.get_building_block(),
            name=temp.get_name(),
            output_dir=calculation_output,
            bead_set=temp.get_bead_set(),
        )
        opt_bb.write(str(ligand_output / f"{temp.get_name()}_optl.mol"))
        blocks[temp.get_name()] = (opt_bb, temp.get_bead_set())
    return blocks


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

    seeds = {
        0: 1000,
        1: 2973,
        2: 87,
        3: 5552,
        4: 9741244,
        5: 6425,
        6: 123,
        7: 26533,
        8: 782,
        9: 10000,
    }
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

            for run in seeds:
                name = (
                    f"{cage_topo_str}_{bbl_str}_{bb2_str}_"
                    f"{custom_torsion}_{custom_vdw}_{run}"
                )
                run_seed = seeds[run]

                logging.info(f"building {name}")
                cage = stk.ConstructedMolecule(
                    topology_graph=populations[popn]["t"][
                        cage_topo_str
                    ](
                        building_blocks=(bb2, bbl),
                    ),
                )

                cage = optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_output,
                    bead_set=bead_set,
                    custom_torsion_set=custom_torsion_set,
                    custom_vdw_set=custom_vdw_set,
                    run_seed=run_seed,
                )

                if cage is not None:
                    cage.write(str(struct_output / f"{name}_optc.mol"))

                analyse_cage(
                    molecule=cage,
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
