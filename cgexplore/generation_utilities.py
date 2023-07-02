#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for structure generation utilities.

Author: Andrew Tarzia

"""

import os
import logging
import numpy as np
import itertools
from openmm import openmm, OpenMMException

from .openmm_optimizer import CGOMMOptimizer


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


def run_soft_md_cycle(
    name,
    molecule,
    bead_set,
    ensemble,
    output_dir,
    custom_vdw_set,
    custom_torsion_set,
    bonds,
    angles,
    torsions,
    vdw_bond_cutoff,
    num_steps,
    suffix,
    bond_ff_scale,
    angle_ff_scale,
    temperature,
    time_step,
    friction,
    reporting_freq,
    traj_freq,
):
    """
    Run MD exploration with soft potentials.

    """
    soft_bead_set = {}
    for i in bead_set:
        new_bead = replace(bead_set[i])
        new_bead.bond_k = bead_set[i].bond_k / bond_ff_scale
        new_bead.angle_k = bead_set[i].angle_k / angle_ff_scale
        soft_bead_set[i] = new_bead

    md = CGOMMDynamics(
        fileprefix=f"{name}_{suffix}",
        output_dir=output_dir,
        param_pool=soft_bead_set,
        custom_torsion_set=custom_torsion_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw=custom_vdw_set,
        vdw_bond_cutoff=2,
        temperature=temperature,
        random_seed=1000,
        num_steps=num_steps,
        time_step=time_step,
        friction=friction,
        reporting_freq=reporting_freq,
        traj_freq=traj_freq,
    )

    try:
        return md.run_dynamics(molecule)

    except OpenMMException:
        return None


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


def run_constrained_optimisation(
    molecule,
    bead_set,
    name,
    output_dir,
    custom_vdw_set,
    bond_ff_scale,
    angle_ff_scale,
    max_iterations,
):
    """
    Run optimisation with constraints and softened potentials.

    Keywords
    ========

    molecule: stk.Molecule
    bead_set: dict
    name: str
    output_dir: str or Path
    custom_vdw_set: ??
    bond_ff_scale: float
        Scale (divide) the bond terms in the model by this value.
    angle_ff_scale: float
        Scale (divide) the angle terms in the model by this value.
    max_iterations: int
        Num steps to take.

    """

    soft_bead_set = {}
    for i in bead_set:
        new_bead = replace(bead_set[i])
        new_bead.bond_k = bead_set[i].bond_k / bond_ff_scale
        new_bead.angle_k = bead_set[i].angle_k / angle_ff_scale
        soft_bead_set[i] = new_bead

    intra_bb_bonds = []
    for bond_info in molecule.get_bond_infos():
        if bond_info.get_building_block_id() is not None:
            bond = bond_info.get_bond()
            intra_bb_bonds.append(
                (bond.get_atom1().get_id(), bond.get_atom2().get_id())
            )

    constrained_opt = CGOMMOptimizer(
        fileprefix=f"{name}_constrained",
        output_dir=output_dir,
        param_pool=soft_bead_set,
        custom_torsion_set=None,
        bonds=True,
        angles=True,
        torsions=False,
        vdw=custom_vdw_set,
        max_iterations=max_iterations,
        vdw_bond_cutoff=2,
        atom_constraints=intra_bb_bonds,
    )
    logging.info(
        f"optimising {name} with {len(intra_bb_bonds)} constraints"
    )
    return constrained_opt.optimize(molecule)


def run_optimisation(
    molecule,
    bead_set,
    name,
    file_suffix,
    output_dir,
    custom_vdw_set,
    custom_torsion_set,
    bonds,
    angles,
    torsions,
    vdw_bond_cutoff,
    max_iterations=None,
    ensemble=None,
):
    """
    Run optimisation and save outcome to Ensemble.

    Keywords
    ========

    """

    opt = CGOMMOptimizer(
        fileprefix=f"{name}_{file_suffix}",
        output_dir=output_dir,
        param_pool=bead_set,
        custom_torsion_set=custom_torsion_set,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        vdw=custom_vdw_set,
        max_iterations=max_iterations,
        vdw_bond_cutoff=vdw_bond_cutoff,
    )
    molecule = opt.optimize(molecule)
    energy_decomp = opt.read_final_energy_decomposition()
    if ensemble is None:
        confid = None
    else:
        confid = ensemble.get_num_conformers()

    return Conformer(
        molecule=molecule,
        conformer_id=confid,
        energy_decomposition=energy_decomp,
    )
