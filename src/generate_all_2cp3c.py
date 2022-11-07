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

from precursor_db.precursors import (
    three_precursor_topology_options,
    two_precursor_topology_options,
)

from beads import beads_2c, beads_3c


def get_shape_calculation_molecule(const_mol, name):
    splits = name.split("_")
    topo_str = splits[0]
    c3_name = splits[1]
    c3_topo_str = c3_name[:3]

    central_c3_name = tuple(
        i.element_string
        for i in beads_3c()
        if i.element_string in c3_name.replace(c3_topo_str, "")[:2]
    )[0]
    old_position_matrix = const_mol.get_position_matrix()

    atoms = []
    position_matrix = []
    for a in const_mol.get_atoms():
        if a.__class__.__name__ == central_c3_name:
            new_atom = stk.Atom(
                id=len(atoms),
                atomic_number=a.get_atomic_number(),
                charge=a.get_charge(),
            )
            atoms.append(new_atom)
            position_matrix.append(old_position_matrix[a.get_id()])

    if topo_str in ("TwoPlusThree",):

        bbs = list(const_mol.get_building_blocks())
        two_c_bb = bbs[1]
        for ai in const_mol.get_atom_infos():
            if ai.get_building_block() == two_c_bb:
                # The atom to use is always the first in the building
                # block.
                if ai.get_building_block_atom().get_id() == 0:
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

    subset_molecule = stk.BuildingBlock.init(
        atoms=atoms, bonds=(), position_matrix=np.array(position_matrix)
    )
    return subset_molecule


def optimise_cage(molecule, name, output_dir):

    opt_xyz_file = os.path.join(output_dir, f"{name}_opted.xyz")
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")
    opt2_mol_file = os.path.join(output_dir, f"{name}_opted2.mol")

    # Does optimisation.
    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}...")
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=beads_2c() + beads_3c(),
            max_cycles=1000,
            conjugate_gradient=True,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule.write(opt1_mol_file)
        os.system(f"rm {opt_xyz_file}")

    if os.path.exists(opt2_mol_file):
        molecule = molecule.with_structure_from_file(opt2_mol_file)
    else:
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=beads_2c() + beads_3c(),
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        _ = opt.optimize(molecule)
        molecule = molecule.with_structure_from_file(opt_xyz_file)
        molecule.write(opt2_mol_file)
        os.system(f"rm {opt_xyz_file}")

    return molecule


def analyse_cage(molecule, name, output_dir):

    output_file = os.path.join(output_dir, f"{name}_res.json")
    pm_output_file = os.path.join(output_dir, f"{name}_opted.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        opt = CGGulpOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            param_pool=beads_2c() + beads_3c(),
            max_cycles=1000,
            conjugate_gradient=False,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        run_data = opt.extract_gulp()
        fin_energy = run_data["final_energy"]

        shape_measures = ShapeMeasure(
            output_dir=(output_dir / f"{name}_shape"),
            target_atmnums=None,
            shape_string=None,
        ).calculate(get_shape_calculation_molecule(molecule, name))

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
    first_line = f"Usage: {__file__}.py mode"
    if not len(sys.argv) == 2:
        logging.info(f"{first_line}")
        logging.info(
            "\nmode must be either `run` or `plot`, where `plot` just "
            "outputs known distributions."
        )
        sys.exit()
    else:
        mode = sys.argv[1]

    if mode not in ("run", "plot"):
        raise ValueError(f"mode must be run or plot, not {mode}")

    struct_output = cages() / "structures"
    check_directory(struct_output)
    figure_output = cages() / "figures"
    check_directory(figure_output)
    calculation_output = cages() / "calculations"
    check_directory(calculation_output)

    # Define list of topology functions.
    cage_topologies = cage_topology_options()

    # Define precursor topologies.
    three_precursor_topologies = three_precursor_topology_options()
    two_precursor_topologies = two_precursor_topology_options()

    # Define bead libraries.
    beads_3c_lib = beads_3c()
    beads_2c_lib = beads_2c()

    if mode == "run":
        # For now, just build N options and calculate properties.
        logging.info("building building blocks...")
        c2_blocks = {}
        c2_products = itertools.product(beads_2c_lib, repeat=3)
        for c2_options in c2_products:
            c2_topo = two_precursor_topologies["2c-0"]
            temp = c2_topo(bead=c2_options[0])
            c2_blocks[temp.get_name()] = temp.get_building_block()

            c2_topo = two_precursor_topologies["2c-1"]
            temp = c2_topo(bead=c2_options[0], abead1=c2_options[1])
            c2_blocks[temp.get_name()] = temp.get_building_block()

            c2_topo = two_precursor_topologies["2c-2"]
            temp = c2_topo(
                bead=c2_options[0],
                abead1=c2_options[1],
                abead2=c2_options[2],
            )
            c2_blocks[temp.get_name()] = temp.get_building_block()

        c3_blocks = {}
        for core_bead in beads_3c_lib:
            c3_topo = three_precursor_topologies["3c-0"]
            temp = c3_topo(bead=core_bead)
            c3_blocks[temp.get_name()] = temp.get_building_block()

            for c2 in beads_2c_lib:
                c3_topo = three_precursor_topologies["3c-1"]
                temp = c3_topo(bead=core_bead, abead1=c2)
                c3_blocks[temp.get_name()] = temp.get_building_block()

        logging.info(
            f"there are {len(c2_blocks)} 2-C and "
            f"{len(c3_blocks)} 3-C building blocks."
        )

        logging.info("building population...")
        popn_iterator = itertools.product(
            cage_topologies, c2_blocks, c3_blocks
        )
        count = 0
        for iteration in popn_iterator:
            cage_topo_str, bb2_str, bb3_str = iteration
            name = f"{cage_topo_str}_{bb3_str}_{bb2_str}"
            cage = stk.ConstructedMolecule(
                topology_graph=cage_topologies[cage_topo_str](
                    building_blocks=(
                        c2_blocks[bb2_str],
                        c3_blocks[bb3_str],
                    ),
                ),
            )
            cage = optimise_cage(cage, name, calculation_output)
            cage.write(str(struct_output / f"{name}_optc.mol"))
            analyse_cage(cage, name, calculation_output)
            count += 1

        logging.info(f"{count} cages built.")

    plot_existing_data_distributions(
        calculation_dir=calculation_output,
        figures_dir=figure_output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
