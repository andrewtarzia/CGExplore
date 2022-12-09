#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

Author: Andrew Tarzia

"""

import sys
import stk
import os
import json
import logging
import numpy as np
import itertools

from shape import ShapeMeasure
from pore import PoreMeasure
from env_set import cages
from utilities import check_directory, get_atom_distance
from gulp_optimizer import (
    CGGulpOptimizer,
    CGGulpMD,
    MDEmptyTrajcetoryError,
)

from cage_construction.topologies import cage_topology_options

from precursor_db.topologies import TwoC1Arm, ThreeC1Arm, FourC1Arm

from beads import CgBead, bead_library_check


def produce_bead_library(
    type_prefix,
    element_string,
    sigmas,
    angles,
    bond_ks,
    angle_ks,
):
    return {
        f"{type_prefix}{i}{j}{k}{l}": CgBead(
            element_string=element_string,
            bead_type=f"{type_prefix}{i}{j}{k}{l}",
            sigma=sigma,
            angle_centered=angle,
            bond_k=bond_k,
            angle_k=angle_k,
        )
        for (i, sigma), (j, angle), (k, bond_k), (
            l,
            angle_k,
        ) in itertools.product(
            enumerate(sigmas),
            enumerate(angles),
            enumerate(bond_ks),
            enumerate(angle_ks),
        )
    }


def core_2c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        sigmas=range(1, 10, 1),
        angles=(180,),
        bond_ks=(10,),
        angle_ks=(20,),
    )


def arm_2c_beads():
    return produce_bead_library(
        type_prefix="a",
        element_string="Ba",
        sigmas=(1,),
        angles=range(90, 181, 10),
        bond_ks=(10,),
        angle_ks=(20,),
    )


def binder_beads():
    return produce_bead_library(
        type_prefix="b",
        element_string="Pb",
        sigmas=(1,),
        angles=(180,),
        bond_ks=(10,),
        angle_ks=(20,),
    )


def beads_3c():
    return produce_bead_library(
        type_prefix="n",
        element_string="C",
        sigmas=range(1, 16, 1),
        angles=(60, 90, 109.5, 120),
        bond_ks=(10,),
        angle_ks=(20,),
    )


def beads_4c():
    return produce_bead_library(
        type_prefix="m",
        element_string="Pd",
        sigmas=range(1, 16, 1),
        angles=((50, 130), (70, 130), (90, 130)),
        bond_ks=(10,),
        angle_ks=(20,),
    )


def get_shape_calculation_molecule(const_mol, name):
    splits = name.split("_")
    topo_str = splits[0]
    bbs = list(const_mol.get_building_blocks())
    old_position_matrix = const_mol.get_position_matrix()

    large_c_bb = bbs[0]
    atoms = []
    position_matrix = []
    # print(name)

    cl_name = splits[1]
    cl_topo_str = cl_name[:3]
    if cl_topo_str in ("3C1", "4C1"):
        target_id = 1
    else:
        target_id = 0
    for ai in const_mol.get_atom_infos():
        if ai.get_building_block() == large_c_bb:
            # The atom to use is always the first in the building
            # block.
            if ai.get_building_block_atom().get_id() == target_id:
                a = ai.get_atom()
                new_atom = stk.Atom(
                    id=len(atoms),
                    atomic_number=a.get_atomic_number(),
                    charge=a.get_charge(),
                )
                atoms.append(new_atom)
                position_matrix.append(old_position_matrix[a.get_id()])

    if topo_str in ("TwoPlusThree", "M2L4", "TwoPlusFour"):
        c2_name = splits[2]
        c2_topo_str = c2_name[:3]
        if c2_topo_str == "2C0":
            target_id = 1
        else:
            target_id = 0
        two_c_bb = bbs[1]
        for ai in const_mol.get_atom_infos():
            # print(two_c_bb == ai.get_building_block())
            if ai.get_building_block() == two_c_bb:
                # The atom to use is always the first in the building
                # block.
                # print(0, ai.get_building_block_atom().get_id())
                if ai.get_building_block_atom().get_id() == target_id:
                    a = ai.get_atom()
                    new_atom = stk.Atom(
                        id=len(atoms),
                        atomic_number=a.get_atomic_number(),
                        charge=a.get_charge(),
                    )
                    atoms.append(new_atom)
                    position_matrix.append(
                        old_position_matrix[a.get_id()]
                    )

    num_atoms = len(atoms)
    # print(num_atoms)
    # print(topo_str)
    if topo_str == "TwoPlusThree" and num_atoms != 5:
        raise ValueError(
            f"{topo_str} needs 5 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "FourPlusSix" and num_atoms != 4:
        raise ValueError(
            f"{topo_str} needs 4 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "FourPlusSix2" and num_atoms != 4:
        raise ValueError(
            f"{topo_str} needs 4 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "SixPlusNine" and num_atoms != 6:
        raise ValueError(
            f"{topo_str} needs 6 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "EightPlusTwelve" and num_atoms != 8:
        raise ValueError(
            f"{topo_str} needs 8 atoms, not {num_atoms}; name={name}"
        )

    if topo_str == "M2L4" and num_atoms != 6:
        raise ValueError(
            f"{topo_str} needs 6 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "M3L6" and num_atoms != 3:
        raise ValueError(
            f"{topo_str} needs 3 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "M4L8" and num_atoms != 4:
        raise ValueError(
            f"{topo_str} needs 4 atoms, not {num_atoms}; name={name}"
        )
    if topo_str == "M6L12" and num_atoms != 6:
        raise ValueError(
            f"{topo_str} needs 6 atoms, not {num_atoms}; name={name}"
        )

    subset_molecule = stk.BuildingBlock.init(
        atoms=atoms,
        bonds=(),
        position_matrix=np.array(position_matrix),
    )
    return subset_molecule


def optimise_ligand(molecule, name, output_dir, full_bead_library):

    opt_xyz_file = os.path.join(output_dir, f"{name}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")

    # Does optimisation.
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}...")
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=2000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)

    return molecule


def check_long_distances(molecule, name, max_distance):

    max_present = 0
    for bond in molecule.get_bonds():
        distance = get_atom_distance(
            molecule=molecule,
            atom1_id=bond.get_atom1().get_id(),
            atom2_id=bond.get_atom2().get_id(),
        )
        max_present = max((distance, max_present))

    if max_present >= max_distance:
        raise ValueError(
            f"a max dist of {max_distance} found for {name}. "
            "Suggests bad optimisation."
        )


def optimise_cage(molecule, name, output_dir, full_bead_library):

    opt_xyz_file = os.path.join(output_dir, f"{name}_opted.xyz")
    mdr_xyz_file = os.path.join(output_dir, f"{name}_final.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{name}_opted2.mol")
    opt3_mol_file = os.path.join(output_dir, f"{name}_opted3.mol")

    # Does optimisation.
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}...")
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=2000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt1_mol_file)

    check_long_distances(molecule, name=name, max_distance=15)

    if os.path.exists(opt2_mol_file):
        molecule = molecule.with_structure_from_file(opt2_mol_file)
    else:
        logging.info(f"optimising {name}...")

        opt = CGGulpMD(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(mdr_xyz_file)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt2_mol_file)

    if os.path.exists(opt3_mol_file):
        molecule = molecule.with_structure_from_file(opt3_mol_file)
    else:
        logging.info(f"optimising {name}...")
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=2000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule = molecule.with_centroid((0, 0, 0))
        molecule.write(opt3_mol_file)

    return molecule


def analyse_cage(molecule, name, output_dir, full_bead_library):

    output_file = os.path.join(output_dir, f"{name}_res.json")
    # xyz_template = os.path.join(output_dir, f"{name}_temp.xyz")
    pm_output_file = os.path.join(output_dir, f"{name}_opted.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        # molecule.write(xyz_template)
        run_data = opt.extract_gulp()
        try:
            fin_energy = run_data["final_energy"]
        except KeyError:
            raise KeyError(
                "final energy not found in run_data, suggests failure "
                f"of {name}"
            )

        raise SystemExit('fix shape calcualtion')
        shape_mol = get_shape_calculation_molecule(molecule, name)
        shape_measures = ShapeMeasure(
            output_dir=(output_dir / f"{name}_shape"),
            target_atmnums=None,
            shape_string=None,
        ).calculate(shape_mol)

        raise SystemExit('check pore calc will be matched to bead type')
        opt_pore_data = PoreMeasure().calculate_pore(
            molecule=molecule,
            output_file=pm_output_file,
        )
        res_dict = {
            "fin_energy": fin_energy,
            "opt_pore_data": opt_pore_data,
            "shape_measures": shape_measures,
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
        print(options)
        option1 = option1_lib[options[0]]
        option2 = option2_lib[options[1]]
        print(option1, option2)
        temp = topology(bead=option1, abead1=option2)
        print(temp.get_name())
        raise SystemExit()
        opt_bb = optimise_ligand(
            molecule=temp.get_building_block(),
            name=temp.get_name(),
            output_dir=calculation_output,
            full_bead_library=full_bead_library,
        )
        opt_bb.write(str(ligand_output / f"{temp.get_name()}_optl.mol"))
        blocks[temp.get_name()] = opt_bb
    return blocks


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = cages() / "structures"
    check_directory(struct_output)
    figure_output = cages() / "figures"
    check_directory(figure_output)
    calculation_output = cages() / "calculations"
    check_directory(calculation_output)
    ligand_output = cages() / "ligands"
    check_directory(ligand_output)

    # Define list of topology functions.
    cage_3p2_topologies = cage_topology_options("2p3")
    cage_4p2_topologies = cage_topology_options("2p4")

    # Define bead libraries.
    beads_core_2c_lib = new_core_2c_beads()
    beads_4c_lib = new_beads_4c()
    beads_3c_lib = new_beads_3c()
    beads_arm_2c_lib = new_arm_2c_beads()
    beads_binder_lib = new_binder_beads()
    full_bead_library = (
        list(beads_3c_lib.values())
        + list(beads_4c_lib.values())
        + list(beads_arm_2c_lib.values())
        + list(beads_core_2c_lib.values())
        + list(beads_binder_lib.values())
    )
    bead_library_check(full_bead_library)

    # For now, just build N options and calculate properties.
    logging.info("building building blocks...")

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



    raise SystemExit("fix angles for 4c BBs")

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

    for popn in populations:
        logging.info(f"building {popn} population...")
        popn_iterator = itertools.product(
            populations[popn]["t"],
            populations[popn]["c2"],
            populations[popn]["cl"],
        )
        count = 0
        for iteration in popn_iterator:
            cage_topo_str, bb2_str, bbl_str = iteration
            name = f"{cage_topo_str}_{bbl_str}_{bb2_str}"
            cage = stk.ConstructedMolecule(
                topology_graph=populations[popn]["t"][cage_topo_str](
                    building_blocks=(
                        populations[popn]["c2"][bb2_str],
                        populations[popn]["cl"][bbl_str],
                    ),
                ),
            )
            try:
                cage = optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_output,
                    full_bead_library=full_bead_library,
                )
            except MDEmptyTrajcetoryError:
                continue
            cage.write(str(struct_output / f"{name}_optc.mol"))
            raise SystemExit(
                "go through everything, changing the bead librari and bbs"
            )
            analyse_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )
            count += 1
            raise SystemExit(
                "go through everything, changing the bead librari and bbs"
            )

        logging.info(f"{count} {popn} cages built.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
