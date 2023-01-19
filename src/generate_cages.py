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
import itertools
from rdkit import RDLogger

from shape import (
    ShapeMeasure,
    get_shape_molecule_ligands,
    get_shape_molecule_nodes,
)
from geom import GeomMeasure
from pore import PoreMeasure
from env_set import cages
from utilities import check_directory, check_long_distances
from openmm_optimizer import CGOMMOptimizer, CGOMMDynamics
from cage_construction.topologies import cage_topology_options
from precursor_db.topologies import TwoC1Arm, ThreeC1Arm, FourC1Arm
from beads import bead_library_check, produce_bead_library


def core_2c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        sigmas=(2, 5),
        angles=(180,),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        epsilon=10.0,
        coordination=2,
    )


def arm_2c_beads():
    return produce_bead_library(
        type_prefix="a",
        element_string="Ba",
        sigmas=(1,),
        angles=range(90, 181, 5),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        epsilon=10.0,
        coordination=2,
    )


def binder_beads():
    return produce_bead_library(
        type_prefix="b",
        element_string="Pb",
        sigmas=(1,),
        angles=(180,),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        epsilon=10.0,
        coordination=2,
    )


def beads_3c():
    return produce_bead_library(
        type_prefix="n",
        element_string="C",
        sigmas=(2, 5),
        angles=(60, 90, 109.5, 120),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
        epsilon=10.0,
        coordination=3,
    )


def beads_4c():
    return produce_bead_library(
        type_prefix="m",
        element_string="Pd",
        sigmas=(2, 5),
        angles=(50, 70, 90),
        bond_ks=(5e5,),
        angle_ks=(5e2,),
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
    opt3_mol_file = os.path.join(output_dir, f"{name}_opted3.mol")
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")

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

        molecule = opt.optimize(molecule)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)
    first_energy = opt.calculate_energy(molecule)

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
            temperature=300,
            random_seed=1000,
        )
        trajectory = opt.run_dynamics(molecule)
        traj_log = trajectory.get_data()
        min_energy = 1e24
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            if pot_energy < min_energy:
                logging.info(
                    f"new lowest E conformer {timestep}: "
                    f"{pot_energy} kJ/mol-1"
                )
                min_energy = pot_energy
                molecule = molecule.with_position_matrix(
                    conformer.molecule.get_position_matrix(),
                )
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
        logging.info(f"optimising {name}")
        molecule = opt.optimize(molecule)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt3_mol_file)
    final_energy = opt.calculate_energy(molecule)

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

    if first_energy < final_energy:
        logging.info("energy after opt1 < energy after MD search.")
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    molecule.write(fina_mol_file)

    return molecule


def analyse_cage(
    molecule,
    name,
    output_dir,
    bead_set,
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
        fin_energy = 0
        fin_gnorm = 0

        raise SystemExit("get run data")

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

        # Always want to extract target torions.
        temp_custom_torsion_set = target_torsions(
            bead_set=bead_set,
            custom_torsion_option=None,
        )

        custom_torsion_atoms = [
            bead_set[j].element_string
            for i in temp_custom_torsion_set
            for j in i
        ]
        raise SystemExit("add Rg or avg sphericity of points")
        g_measure = GeomMeasure(custom_torsion_atoms)
        bond_data = g_measure.calculate_bonds(molecule)
        angle_data = g_measure.calculate_angles(molecule)
        dihedral_data = g_measure.calculate_dihedrals(molecule)
        min_b2b_distance = g_measure.calculate_minb2b(molecule)

        res_dict = {
            "optimised": True,
            "fin_energy": fin_energy,
            "fin_gnorm": fin_gnorm,
            "opt_pore_data": opt_pore_data,
            "lig_shape_measures": lig_shape_measures,
            "node_shape_measures": node_shape_measures,
            "bond_data": bond_data,
            "angle_data": angle_data,
            "dihedral_data": dihedral_data,
            "min_b2b_distance": min_b2b_distance,
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
    (c_key,) = (i for i in bead_set if i[0] == "c")
    (t_key_1,) = (i for i in bead_set if i[0] == "a")
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

    # Define list of topology functions.
    cage_3p2_topologies = cage_topology_options("2p3")
    cage_4p2_topologies = cage_topology_options("2p4")

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

    for popn in populations:
        logging.info(f"building {popn} population")
        popn_iterator = itertools.product(
            populations[popn]["t"],
            populations[popn]["c2"],
            populations[popn]["cl"],
        )
        count = 0
        for iteration in popn_iterator:
            for custom_torsion in custom_torsion_options:
                for custom_vdw in custom_vdw_options:
                    cage_topo_str, bb2_str, bbl_str = iteration
                    name = (
                        f"{cage_topo_str}_{bbl_str}_{bb2_str}_"
                        f"{custom_torsion}_{custom_vdw}"
                    )
                    bb2, bb2_bead_set = populations[popn]["c2"][bb2_str]
                    bbl, bbl_bead_set = populations[popn]["cl"][bbl_str]

                    bead_set = bb2_bead_set.copy()
                    bead_set.update(bbl_bead_set)
                    (ba_bead,) = (
                        bb2_bead_set[i]
                        for i in bb2_bead_set
                        if "a" in i
                    )
                    # bite_angle = (ba_bead.angle_centered - 90) * 2
                    if custom_torsion_options[custom_torsion] is None:
                        custom_torsion_set = None
                    # elif bite_angle == 180:
                    #     # There are no torsions for bite angle == 180.
                    #     custom_torsion_set = None
                    else:
                        tors_option = custom_torsion_options[
                            custom_torsion
                        ]
                        custom_torsion_set = target_torsions(
                            bead_set=bead_set,
                            custom_torsion_option=tors_option,
                        )

                    custom_vdw_set = custom_vdw_options[custom_vdw]

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
                    )

                    if cage is not None:
                        cage.write(
                            str(struct_output / f"{name}_optc.mol")
                        )
                    continue
                    analyse_cage(
                        molecule=cage,
                        name=name,
                        output_dir=calculation_output,
                        bead_set=bead_set,
                        custom_torsion_set=custom_torsion_set,
                        custom_vdw_set=custom_vdw_set,
                    )
                    count += 1
                    raise SystemExit(
                        "set force values for bonds, angles, vdw, torsions "
                        "to be strict"
                    )

        logging.info(f"{count} {popn} cages built.")


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
