#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for structure generation utilities.

Author: Andrew Tarzia

"""


import logging
import os
import pathlib
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import stk
from openmm import OpenMMException, openmm

from .beads import CgBead, periodic_table
from .ensembles import Conformer, Ensemble
from .openmm_optimizer import CGOMMDynamics, CGOMMOptimizer, OMMTrajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def optimise_ligand(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    bead_set: dict[str, CgBead],
    platform: str | None,
) -> stk.Molecule:
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")

    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}, no max_iterations")
        opt = CGOMMOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            bead_set=bead_set,
            custom_torsion_set=None,
            custom_vdw_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
            platform=platform,
        )
        molecule = opt.optimize(molecule)
        molecule = molecule.with_centroid(np.array((0, 0, 0)))
        molecule.write(opt1_mol_file)

    return molecule


def random_deform_molecule(
    molecule: stk.Molecule,
    generator: np.random.Generator,
    sigma: float,
) -> stk.Molecule:
    old_pos_mat = molecule.get_position_matrix()

    new_pos_mat = []
    for pos in old_pos_mat:
        move = generator.random((3,)) * sigma
        new_pos = pos - move
        new_pos_mat.append(new_pos)

    return molecule.with_position_matrix(np.array((new_pos_mat)))


def run_mc_cycle(
    name: str,
    molecule: stk.Molecule,
    bead_set: dict[str, CgBead],
    ensemble: Ensemble,
    output_dir: pathlib.Path,
    custom_vdw_set: tuple,
    custom_torsion_set: tuple,
    bonds: bool,
    angles: bool,
    torsions: bool,
    vdw_bond_cutoff: int,
    sigma: float,
    num_steps: int,
    seed: int,
    beta: float,
    suffix: str,
    platform: str,
) -> stk.Molecule:
    """
    Run metropolis MC scheme.

    """

    generator = np.random.default_rng(seed=seed)

    # Run an initial step.
    opt = CGOMMOptimizer(
        fileprefix=f"{name}_{suffix}",
        output_dir=output_dir,
        bead_set=bead_set,
        custom_torsion_set=custom_torsion_set,
        custom_vdw_set=custom_vdw_set,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        vdw=False,
        max_iterations=200,
        vdw_bond_cutoff=vdw_bond_cutoff,
        platform=platform,
    )
    molecule = opt.optimize(molecule)
    energy_decomp = opt.read_final_energy_decomposition()
    current_energy = energy_decomp["total energy"][0]

    num_passed = 0
    num_run = 0
    for step in range(num_steps + 1):
        # Perform deformation.
        test_molecule = random_deform_molecule(
            molecule=molecule,
            generator=generator,
            sigma=sigma,
        )
        test_molecule = opt.optimize(test_molecule)
        test_energy = opt.read_final_energy_decomposition()["total energy"][0]

        passed = False
        if test_energy < current_energy:
            passed = True
        elif (
            np.exp(-beta * (test_energy - current_energy)) > generator.random()
        ):
            passed = True

        # Pass or fail.
        if passed:
            num_passed += 1
            decomp = opt.read_final_energy_decomposition()
            conformer = Conformer(
                molecule=test_molecule.clone().with_centroid(
                    np.array((0, 0, 0))
                ),
                conformer_id=None,
                energy_decomposition=(decomp),
            )

            ensemble.add_conformer(
                conformer=conformer,
                source=suffix,
            )
            molecule = conformer.molecule
            current_energy = decomp["total energy"][0]  # type: ignore[index]

        num_run += 1

    logging.info(f"{num_passed} passed of {num_run} ({num_passed/num_run})")
    ensemble.write_conformers_to_file()
    return molecule


def run_soft_md_cycle(
    name: str,
    molecule: stk.Molecule,
    bead_set: dict[str, CgBead],
    output_dir: pathlib.Path,
    custom_vdw_set: tuple,
    custom_torsion_set: tuple,
    num_steps: int,
    suffix: str,
    bond_ff_scale: float,
    angle_ff_scale: float,
    temperature: openmm.unit.Quantity,
    time_step: openmm.unit.Quantity,
    friction: openmm.unit.Quantity,
    reporting_freq: float,
    traj_freq: float,
    platform: str,
) -> OMMTrajectory | None:
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
        bead_set=soft_bead_set,
        custom_torsion_set=custom_torsion_set,
        custom_vdw_set=custom_vdw_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw=False,
        vdw_bond_cutoff=2,
        temperature=temperature,
        random_seed=1000,
        num_steps=num_steps,
        time_step=time_step,
        friction=friction,
        reporting_freq=reporting_freq,
        traj_freq=traj_freq,
        platform=platform,
    )

    try:
        return md.run_dynamics(molecule)

    except OpenMMException:
        return None


def run_md_cycle(
    name: str,
    molecule: stk.Molecule,
    md_class: CGOMMDynamics,
    expected_num_steps,
    opt_class: CGOMMOptimizer | None = None,
    min_energy: float | None = None,
):
    raise NotImplementedError(
        "Change return to molecule | str, assign str to either"
    )
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
            min_energy = md_class.calculate_energy(molecule).value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

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


def run_constrained_optimisation(
    molecule: stk.ConstructedMolecule,
    bead_set: dict[str, CgBead],
    name: str,
    output_dir: pathlib.Path,
    custom_vdw_set: tuple,
    bond_ff_scale: float,
    angle_ff_scale: float,
    max_iterations: int,
    platform: str,
) -> stk.Molecule:
    """
    Run optimisation with constraints and softened potentials.

    Keywords:

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
        bead_set=soft_bead_set,
        custom_torsion_set=None,
        custom_vdw_set=custom_vdw_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw=False,
        max_iterations=max_iterations,
        vdw_bond_cutoff=2,
        atom_constraints=intra_bb_bonds,
        platform=platform,
    )
    logging.info(f"optimising with {len(intra_bb_bonds)} constraints")
    return constrained_opt.optimize(molecule)


def run_optimisation(
    molecule: stk.Molecule,
    bead_set: dict[str, CgBead],
    name: str,
    file_suffix: str,
    output_dir: pathlib.Path,
    custom_vdw_set: tuple,
    custom_torsion_set: tuple,
    bonds: bool,
    angles: bool,
    torsions: bool,
    vdw_bond_cutoff: int,
    platform: str,
    max_iterations: int | None = None,
    ensemble: Ensemble | None = None,
) -> Conformer:
    """
    Run optimisation and save outcome to Ensemble.

    Keywords:

    """

    opt = CGOMMOptimizer(
        fileprefix=f"{name}_{file_suffix}",
        output_dir=output_dir,
        bead_set=bead_set,
        custom_torsion_set=custom_torsion_set,
        custom_vdw_set=custom_vdw_set,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        vdw=False,
        max_iterations=max_iterations,
        vdw_bond_cutoff=vdw_bond_cutoff,
        platform=platform,
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


def modify_bead(bead_name: str) -> Iterator[str]:
    for i, s in enumerate(bead_name):
        temp_bead_name = list(bead_name)
        if not s.isnumeric():
            continue
        temp_bead_name[i] = str(int(s) - 1)
        yield "".join(temp_bead_name)
        temp_bead_name[i] = str(int(s) + 1)
        yield "".join(temp_bead_name)


def yield_near_models(
    molecule: stk.Molecule,
    name: str,
    bead_set: dict[str, CgBead],
    output_dir: pathlib.Path | str,
) -> Iterator[stk.Molecule]:
    (
        t_str,
        clbb_name,
        c2bb_name,
        torsions,
        vdws,
        run_number,
    ) = name.split("_")

    for bead_name in bead_set:
        for modification in modify_bead(bead_name):
            if bead_name not in clbb_name:
                new_bbl_str = clbb_name
                new_bb2_str = c2bb_name.replace(bead_name, modification)
            elif bead_name not in c2bb_name:
                new_bbl_str = clbb_name.replace(bead_name, modification)
                new_bb2_str = c2bb_name
            new_name = (
                f"{t_str}_{new_bbl_str}_{new_bb2_str}_"
                f"{torsions}_{vdws}_{run_number}"
            )
            new_fina_mol_file = os.path.join(
                output_dir, f"{new_name}_final.mol"
            )
            if os.path.exists(new_fina_mol_file):
                logging.info(f"found neigh: {new_fina_mol_file}")
                yield molecule.with_structure_from_file(new_fina_mol_file)


def shift_beads(
    molecule: stk.Molecule,
    atomic_number: int,
    kick: float,
) -> stk.Molecule:
    old_pos_mat = molecule.get_position_matrix()
    centroid = molecule.get_centroid()

    new_pos_mat = []
    for atom, pos in zip(molecule.get_atoms(), old_pos_mat):
        if atom.get_atomic_number() == atomic_number:
            c_v = centroid - pos
            c_v = c_v / np.linalg.norm(c_v)
            move = c_v * kick
            new_pos = pos - move
        else:
            new_pos = pos
        new_pos_mat.append(new_pos)

    return molecule.with_position_matrix(np.array((new_pos_mat)))


def yield_shifted_models(
    molecule: stk.Molecule,
    bead_set: dict[str, CgBead],
) -> Iterator[stk.Molecule]:
    for bead in bead_set:
        atom_number = periodic_table()[bead_set[bead].element_string]
        for kick in (1, 2, 3, 4):
            yield shift_beads(molecule, atom_number, kick)
