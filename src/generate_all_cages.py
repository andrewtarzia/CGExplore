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
from utilities import check_directory
from gulp_optimizer import CGGulpOptimizer

from cage_construction.topologies import cage_topology_options

from precursor_db.topologies import TwoC1Arm, ThreeC1Arm, FourC1Arm

from beads import CgBead, bead_library_check


def core_2c_beads():
    return (
        CgBead("Ag", sigma=1.0, angle_centered=180),
        CgBead("Bi", sigma=2.0, angle_centered=180),
        CgBead("Ce", sigma=3.0, angle_centered=180),
        CgBead("Eu", sigma=4.0, angle_centered=180),
        CgBead("He", sigma=5.0, angle_centered=180),
        CgBead("Lu", sigma=6.0, angle_centered=180),
        CgBead("Zn", sigma=7.0, angle_centered=180),
    )


def arm_2c_beads():
    return (
        CgBead("Al", sigma=1.0, angle_centered=90),
        CgBead("As", sigma=1.0, angle_centered=100),
        CgBead("Au", sigma=1.0, angle_centered=110),
        CgBead("Ba", sigma=1.0, angle_centered=120),
        CgBead("B", sigma=1.0, angle_centered=130),
        CgBead("Ga", sigma=1.0, angle_centered=140),
        CgBead("Gd", sigma=1.0, angle_centered=150),
        CgBead("Ge", sigma=1.0, angle_centered=160),
        CgBead("La", sigma=1.0, angle_centered=170),
        CgBead("Mn", sigma=1.0, angle_centered=180),
    )


def binder_beads():
    return (CgBead("Pb", sigma=1.0, angle_centered=180),)


def beads_3c():
    return (
        CgBead("Ac", sigma=1.0, angle_centered=120),
        CgBead("Am", sigma=2.0, angle_centered=120),
        CgBead("Cd", sigma=3.0, angle_centered=120),
        CgBead("Cf", sigma=4.0, angle_centered=120),
        CgBead("Cm", sigma=5.0, angle_centered=120),
        CgBead("Co", sigma=6.0, angle_centered=120),
        CgBead("Cu", sigma=7.0, angle_centered=120),
        CgBead("Cr", sigma=8.0, angle_centered=120),
        CgBead("C", sigma=9.0, angle_centered=120),
        CgBead("Er", sigma=10.0, angle_centered=120),
        CgBead("Fe", sigma=11.0, angle_centered=120),
        CgBead("Hf", sigma=12.0, angle_centered=120),
        CgBead("Ho", sigma=1.0, angle_centered=90),
        CgBead("In", sigma=2.0, angle_centered=90),
        CgBead("I", sigma=3.0, angle_centered=90),
        CgBead("Ir", sigma=4.0, angle_centered=90),
        CgBead("Lr", sigma=5.0, angle_centered=90),
        CgBead("Md", sigma=6.0, angle_centered=90),
        CgBead("Ni", sigma=7.0, angle_centered=90),
        CgBead("No", sigma=8.0, angle_centered=90),
        CgBead("Np", sigma=9.0, angle_centered=90),
        CgBead("Pa", sigma=10.0, angle_centered=90),
        CgBead("Po", sigma=11.0, angle_centered=90),
        CgBead("Pr", sigma=12.0, angle_centered=90),
        CgBead("Pu", sigma=1.0, angle_centered=60),
        CgBead("P", sigma=2.0, angle_centered=60),
        CgBead("Re", sigma=3.0, angle_centered=60),
        CgBead("Rh", sigma=4.0, angle_centered=60),
        CgBead("Se", sigma=5.0, angle_centered=60),
        CgBead("Sm", sigma=6.0, angle_centered=60),
        CgBead("S", sigma=7.0, angle_centered=60),
        CgBead("Ti", sigma=8.0, angle_centered=60),
        CgBead("Tm", sigma=9.0, angle_centered=60),
        CgBead("V", sigma=10.0, angle_centered=60),
        CgBead("Y", sigma=11.0, angle_centered=60),
        CgBead("Zr", sigma=12.0, angle_centered=60),
    )


def beads_4c():
    return (
        CgBead("Hg", sigma=1.0, angle_centered=(90, 180, 130)),
        CgBead("Mo", sigma=2.0, angle_centered=(90, 180, 130)),
        CgBead("Nd", sigma=3.0, angle_centered=(90, 180, 130)),
        CgBead("Ne", sigma=4.0, angle_centered=(90, 180, 130)),
        CgBead("Nb", sigma=5.0, angle_centered=(90, 180, 130)),
        CgBead("Os", sigma=6.0, angle_centered=(90, 180, 130)),
        CgBead("Pd", sigma=7.0, angle_centered=(90, 180, 130)),
        CgBead("Pt", sigma=8.0, angle_centered=(90, 180, 130)),
        CgBead("Ru", sigma=9.0, angle_centered=(90, 180, 130)),
        CgBead("Si", sigma=10.0, angle_centered=(90, 180, 130)),
        CgBead("Sn", sigma=11.0, angle_centered=(90, 180, 130)),
        CgBead("Yb", sigma=12.0, angle_centered=(90, 180, 130)),
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

    if topo_str in ("TwoPlusThree", "M2L4"):
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


def optimise_cage(molecule, name, output_dir, full_bead_library):

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
        os.system(f"rm {opt_xyz_file}")

    return molecule


def analyse_cage(molecule, name, output_dir, full_bead_library):

    output_file = os.path.join(output_dir, f"{name}_res.json")
    pm_output_file = os.path.join(output_dir, f"{name}_opted.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=full_bead_library,
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        run_data = opt.extract_gulp()
        fin_energy = run_data["final_energy"]

        shape_mol = get_shape_calculation_molecule(molecule, name)
        shape_measures = ShapeMeasure(
            output_dir=(output_dir / f"{name}_shape"),
            target_atmnums=None,
            shape_string=None,
        ).calculate(shape_mol)

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

    # Define list of topology functions.
    cage_3p2_topologies = cage_topology_options("2p3")
    cage_4p2_topologies = cage_topology_options("2p4")

    # Define bead libraries.
    beads_4c_lib = beads_4c()
    beads_3c_lib = beads_3c()
    beads_core_2c_lib = core_2c_beads()
    beads_arm_2c_lib = arm_2c_beads()
    beads_binder_lib = binder_beads()
    full_bead_library = (
        beads_3c_lib
        + beads_4c_lib
        + beads_arm_2c_lib
        + beads_core_2c_lib
        + beads_binder_lib
    )
    bead_library_check(full_bead_library)

    # For now, just build N options and calculate properties.
    logging.info("building building blocks...")

    c2_blocks = {}
    for c2_options in itertools.product(
        beads_core_2c_lib,
        beads_arm_2c_lib,
    ):
        temp = TwoC1Arm(bead=c2_options[0], abead1=c2_options[1])
        c2_blocks[temp.get_name()] = temp.get_building_block()

    # for c2_options in itertools.product(beads_2c_lib, repeat=3):
    #     c2_topo = two_precursor_topologies["2c-2"]
    #     temp = c2_topo(
    #         bead=c2_options[0],
    #         abead1=c2_options[1],
    #         abead2=c2_options[2],
    #     )
    #     c2_blocks[temp.get_name()] = temp.get_building_block()

    # for c2_options in itertools.product(beads_2c_lib, repeat=4):
    #     c2_topo = two_precursor_topologies["2c-3"]
    #     temp = c2_topo(
    #         bead=c2_options[0],
    #         abead1=c2_options[1],
    #         abead2=c2_options[2],
    #         abead3=c2_options[3],
    #     )
    #     c2_blocks[temp.get_name()] = temp.get_building_block()

    c3_blocks = {}
    for c3_options in itertools.product(
        beads_3c_lib,
        beads_binder_lib,
    ):
        temp = ThreeC1Arm(bead=c3_options[0], abead1=c3_options[1])
        c3_blocks[temp.get_name()] = temp.get_building_block()

    c4_blocks = {}
    for c4_options in itertools.product(
        beads_4c_lib,
        beads_binder_lib,
    ):
        temp = FourC1Arm(bead=c4_options[0], abead1=c4_options[1])
        c4_blocks[temp.get_name()] = temp.get_building_block()

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
                    # optimizer=stk.Collapser(
                    #     scale_steps=False,
                    #     distance_threshold=5,
                    # ),
                ),
            )
            cage = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )
            cage.write(str(struct_output / f"{name}_optc.mol"))
            analyse_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                full_bead_library=full_bead_library,
            )
            count += 1

        logging.info(f"{count} {popn} cages built.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
