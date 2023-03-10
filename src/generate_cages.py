#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate and optimise all CG models.

Author: Andrew Tarzia

"""

import sys
import stk
import os
import json
import logging
import numpy as np
import itertools
from rdkit import RDLogger
from openmm import openmm  # , OpenMMException

from shape import (
    ShapeMeasure,
    get_shape_molecule_ligands,
    get_shape_molecule_nodes,
)
from geom import GeomMeasure
from pore import PoreMeasure
from env_set import cages
from utilities import check_directory, check_long_distances
from openmm_optimizer import (
    CGOMMOptimizer,
    CGOMMDynamics,
    # OMMTrajectory,
)
from cage_construction.topologies import cage_topology_options
from precursor_db.topologies import TwoC1Arm, ThreeC1Arm, FourC1Arm
from beads import bead_library_check, produce_bead_library


def core_2c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        angles=(180,),
        bond_rs=(2, 5),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
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
        bond_ks=(5e5,),
        angle_ks=(5e2,),
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
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def beads_3c():
    return produce_bead_library(
        type_prefix="n",
        element_string="C",
        bond_rs=(2, 5),
        angles=(50, 60, 70, 80, 90, 100, 110, 120),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        sigma=1,
        epsilon=10.0,
        coordination=3,
    )


def beads_4c():
    return produce_bead_library(
        type_prefix="m",
        element_string="Pd",
        bond_rs=(2, 5),
        angles=(50, 60, 70, 80, 90),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
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


def deform_molecule(molecule, generator, percent):
    old_pos_mat = molecule.get_position_matrix()
    centroid = molecule.get_centroid()

    new_pos_mat = []
    for atom, pos in zip(molecule.get_atoms(), old_pos_mat):
        if atom.get_atomic_number() in (6, 46):
            c_v = centroid - pos
            c_v = c_v / np.linalg.norm(c_v)
            # move = generator.choice([-1, 1]) * c_v
            # move = generator.random((3,)) * percent
            move = (c_v + generator.random((3,))) * percent
            new_pos = pos - move
        else:
            move = generator.random((3,)) * percent
            new_pos = pos - move
        new_pos_mat.append(new_pos)
    return molecule.with_position_matrix(np.array((new_pos_mat)))


def deform_and_optimisations(name, molecule, opt):
    min_energy = opt.calculate_energy(molecule)
    # molecule.write("1_o.mol")
    deformations = 20
    generator = np.random.default_rng(seed=1000)
    for drun in range(deformations):
        # logging.info(f"optimising deformation {drun}")
        dmolecule = deform_molecule(molecule, generator, percent=1)
        # dmolecule.write(f"1_{drun}.mol")
        dmolecule = opt.optimize(dmolecule)
        dmolecule = dmolecule.with_centroid((0, 0, 0))
        # dmolecule.write(f"1_{drun}o.mol")
        post_deform_energy = opt.calculate_energy(dmolecule)
        if post_deform_energy < min_energy:
            logging.info(
                f"new low. E conformer (deform.): "
                f"{post_deform_energy}, cf. {min_energy}"
            )
            min_energy = post_deform_energy
            molecule = dmolecule.clone()

    # molecule.write("1_f.mol")
    return molecule


def optimise_cage(
    molecule,
    name,
    output_dir,
    bead_set,
    custom_torsion_set,
    custom_vdw_set,
):

    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{name}_opted2.mol")
    opt2_fai_file = os.path.join(output_dir, f"{name}_mdfailed.txt")
    opt2_exp_file = os.path.join(output_dir, f"{name}_mdexploded.txt")
    opt3_mol_file = os.path.join(output_dir, f"{name}_opted3.mol")
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")

    if os.path.exists(fina_mol_file):
        return molecule.with_structure_from_file(fina_mol_file)

    opt = CGOMMOptimizer(
        fileprefix=f"{name}_o1",
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
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}")
        soft_opt = CGOMMOptimizer(
            fileprefix=f"{name}_o1soft",
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=custom_vdw_set,
            max_iterations=5,
            vdw_bond_cutoff=2,
        )
        molecule = soft_opt.optimize(molecule)
        # molecule.write("1_s.mol")
        molecule = opt.optimize(molecule)
        # molecule.write("1_oo.mol")

        # Run an optimisation on a deformed step and use that if lower
        # in energy.
        molecule = deform_and_optimisations(name, molecule, opt)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)

    opt_energy = opt.calculate_energy(molecule)

    if os.path.exists(opt2_mol_file):
        molecule = molecule.with_structure_from_file(opt2_mol_file)
    else:
        logging.info(f"running MD {name}")
        opt = CGOMMDynamics(
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
            random_seed=1000,
            num_steps=5000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
        )
        try:
            trajectory = opt.run_dynamics(molecule)
            traj_log = trajectory.get_data()
            min_energy = 1e24
            for conformer in trajectory.yield_conformers():
                timestep = conformer.timestep
                row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
                pot_energy = float(row["Potential Energy (kJ/mole)"])
                if pot_energy < min_energy:
                    logging.info(
                        f"new low. E conformer {timestep} (MD): "
                        f"{pot_energy} kJ/mol-1"
                    )
                    min_energy = pot_energy
                    molecule = molecule.with_position_matrix(
                        conformer.molecule.get_position_matrix(),
                    )
            molecule = molecule.with_centroid((0, 0, 0))
            molecule.write(opt2_mol_file)
        except ValueError:
            logging.info(f"{name} MD failed")
            molecule.write(opt2_mol_file)
            with open(opt2_fai_file, "w") as f:
                f.write("failed.")
        except OpenMMException:
            logging.info(f"{name} MD exploded")
            molecule.write(opt2_mol_file)
            with open(opt2_exp_file, "w") as f:
                f.write("exploded.")

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
        logging.info(f"optimising {name}")
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
):

    output_file = os.path.join(output_dir, f"{name}_res.json")
    shape_molfile1 = os.path.join(output_dir, f"{name}_shape1.mol")
    shape_molfile2 = os.path.join(output_dir, f"{name}_shape2.mol")
    # xyz_template = os.path.join(output_dir, f"{name}_temp.xyz")
    # pm_output_file = os.path.join(output_dir, f"{name}_pm.json")

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
            fileprefix=f"{name}_o1",
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

        n_shape_mol = get_shape_molecule_nodes(molecule, name)
        l_shape_mol = get_shape_molecule_ligands(molecule, name)
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
):
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

            name = (
                f"{cage_topo_str}_{bbl_str}_{bb2_str}_"
                f"{custom_torsion}_{custom_vdw}"
            )
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

            cage = stk.ConstructedMolecule(
                topology_graph=populations[popn]["t"][cage_topo_str](
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
            )
            count += 1

        logging.info(f"{count} {popn} cages built.")


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = cages() / "ommstructures"
    check_directory(struct_output)
    figure_output = cages() / "ommfigures"
    check_directory(figure_output)
    calculation_output = cages() / "ommcalculations"
    check_directory(calculation_output)
    ligand_output = cages() / "ommligands"
    check_directory(ligand_output)

    # Define bead libraries.
    beads_core_2c_lib = core_2c_beads()
    beads_4c_lib = beads_4c()
    beads_3c_lib = beads_3c()
    beads_arm_2c_lib = arm_2c_beads()
    beads_binder_lib = binder_beads()
    full_bead_library = (
        list(beads_3c_lib.values())
        + list(beads_4c_lib.values())
        + list(beads_arm_2c_lib.values())
        + list(beads_core_2c_lib.values())
        + list(beads_binder_lib.values())
    )
    bead_library_check(full_bead_library)

    logging.info("building building blocks")
    c2_blocks = build_building_block(
        topology=TwoC1Arm,
        option1_lib=beads_core_2c_lib,
        option2_lib=beads_arm_2c_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )

    c3_blocks = build_building_block(
        topology=ThreeC1Arm,
        option1_lib=beads_3c_lib,
        option2_lib=beads_binder_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )
    c4_blocks = build_building_block(
        topology=FourC1Arm,
        option1_lib=beads_4c_lib,
        option2_lib=beads_binder_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )

    logging.info(
        f"there are {len(c2_blocks)} 2-C and "
        f"{len(c3_blocks)} 3-C and "
        f"{len(c4_blocks)} 4-C building blocks."
    )

    # Define list of topology functions.
    cage_3p2_topologies = cage_topology_options("2p3")
    cage_4p2_topologies = cage_topology_options("2p4")
    cage_3p4_topologies = cage_topology_options("3p4")

    populations = {
        "3 + 2": {
            "t": cage_3p2_topologies,
            "c2": c2_blocks,
            "cl": c3_blocks,
        },
        "4 + 2": {
            "t": cage_4p2_topologies,
            "c2": c2_blocks,
            "cl": c4_blocks,
        },
    }
    custom_torsion_options = {
        "ton": (180, 50),
        "toff": None,
    }
    custom_vdw_options = {
        "von": True,
        "voff": False,
    }
    build_populations(
        populations=populations,
        custom_torsion_options=custom_torsion_options,
        custom_vdw_options=custom_vdw_options,
        struct_output=struct_output,
        calculation_output=calculation_output,
    )

    # Non-ditopic populations.
    populations = {
        "3 + 4": {
            "t": cage_3p4_topologies,
            "c2": c3_blocks,
            "cl": c4_blocks,
        },
    }
    custom_torsion_options = {"toff": None}
    custom_vdw_options = {
        "von": True,
        "voff": False,
    }
    build_populations(
        populations=populations,
        custom_torsion_options=custom_torsion_options,
        custom_vdw_options=custom_vdw_options,
        struct_output=struct_output,
        calculation_output=calculation_output,
    )


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
