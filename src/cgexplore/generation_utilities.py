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
from copy import deepcopy

import numpy as np
import stk
from openmm import OpenMMException, openmm

from .beads import CgBead, periodic_table
from .ensembles import Conformer, Ensemble
from .forcefield import Forcefield
from .openmm_optimizer import CGOMMDynamics, CGOMMOptimizer, OMMTrajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def optimise_ligand(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    force_field: Forcefield,
    platform: str | None,
) -> stk.Molecule:
    """
    Optimise a building block.

    Keywords:

        molecule:
            The molecule to optimise.

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_opted1.mol` in `output_dir`.

        output_dir:
            Directory to save outputs of optimisation process.

        force_field:
            Define the forces used in the molecule.

        platform:
            Which platform to use with OpenMM optimisation. Options are
            `CPU` or `CUDA`. More are available but may not work well
            out of the box.

    Returns:

        An stk molecule.

    """
    opt1_mol_file = os.path.join(output_dir, f"{name}_opted1.mol")

    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}, no max_iterations")
        opt = CGOMMOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            force_field=force_field,
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
    """
    Randomly deform a molecule by changing the atom positions.

    Keywords:

        molecule:
            The molecule to deform.

        generator:
            Random number generator to use in deformations.

        sigma:
            Scale of deformations.

    Returns:

        An stk molecule.

    """
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
    Run metropolis MC scheme. [Currently in old interface, needs updating.]

    Keywords:

    Returns:

    """
    raise NotImplementedError("[Currently in old interface, needs updating.]")

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


def soften_force_field(
    force_field: Forcefield,
    bond_ff_scale: float,
    angle_ff_scale: float,
    output_dir: pathlib.Path,
    prefix: str,
) -> Forcefield:
    """
    Soften force field by scaling parameters.

    Keywords:

        force_field:
            Define the forces used in the molecule. Will be softened and
            constrained.

        bond_ff_scale:
            Scale (divide) the bond terms in the model by this value.

        angle_ff_scale:
            Scale (divide) the angle terms in the model by this value.

        output_dir:
            Directory to save outputs of optimisation process.

        prefix:
            Prefix to use for writing forcefield to file.

    Returns:

        New Forcefield.

    """
    new_bond_terms = []
    for i in force_field.get_bond_terms():
        new_term = deepcopy(i)
        new_term.bond_k = i.bond_k / bond_ff_scale
        new_bond_terms.append(new_term)

    new_angle_terms = []
    for i in force_field.get_angle_terms():
        new_term = deepcopy(i)
        new_term.angle_k = i.angle_k / angle_ff_scale
        new_angle_terms.append(new_term)

    new_custom_angle_terms = []
    for i in force_field.get_custom_angle_terms():
        new_term = deepcopy(i)
        new_term.angle_k = i.angle_k / angle_ff_scale
        new_custom_angle_terms.append(new_term)

    soft_force_field = Forcefield(
        identifier=f"{prefix}{force_field.get_identifier()}",
        output_dir=output_dir,
        prefix=force_field.get_prefix(),
        present_beads=force_field.get_present_beads(),
        bond_terms=tuple(new_bond_terms),
        angle_terms=tuple(new_angle_terms),
        custom_angle_terms=tuple(new_custom_angle_terms),
        torsion_terms=(),
        custom_torsion_terms=(),
        nonbonded_terms=force_field.get_nonbonded_terms(),
        vdw_bond_cutoff=2,
    )
    return soft_force_field


def run_soft_md_cycle(
    name: str,
    molecule: stk.Molecule,
    output_dir: pathlib.Path,
    force_field: Forcefield,
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

    Keywords:

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_opted1.mol` in `output_dir`.

        molecule:
            The molecule to optimise.

        output_dir:
            Directory to save outputs of optimisation process.

        force_field:
            Define the forces used in the molecule. Will be softened and
            constrained.

        num_steps:
            The number of time steps to run the MD for.

        suffix:
            Suffix to use for naming output files. E.g. produces a file
            `{name}_{suffix}_ff.xml` in `output_dir`. Used to define step
            in process.

        bond_ff_scale:
            Scale (divide) the bond terms in the model by this value.

        angle_ff_scale:
            Scale (divide) the angle terms in the model by this value.

        temperature:
            Simulation temperature. Should be
            an openmm.unit.Quantity with units of openmm.unit.kelvin.

        time_step:
            Simulation timestep to use in Integrator. Should be
            an openmm.unit.Quantity with units of [time],
            e.g., openmm.unit.femtoseconds.

        friction:
            Friction constant to use in LangevinIntegrator. Should be
            an openmm.unit.Quantity with units of [time^-1],
            e.g., / openmm.unit.picosecond.

        reporting_freq:
            How often the simulation properties should be written in
            time steps.

        traj_freq:
            How often the trajectory should be written in time steps.

        platform:
            Which platform to use with OpenMM optimisation. Options are
            `CPU` or `CUDA`. More are available but may not work well
            out of the box.

    Returns:

        An OMMTrajectory containing the data and conformers.

    """
    soft_force_field = soften_force_field(
        force_field=force_field,
        bond_ff_scale=bond_ff_scale,
        angle_ff_scale=angle_ff_scale,
        output_dir=output_dir,
        prefix="softmd",
    )
    soft_force_field.write_xml_file()

    md = CGOMMDynamics(
        fileprefix=f"{name}_{suffix}",
        output_dir=output_dir,
        force_field=soft_force_field,
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
    """
    Run a MD cycle. [Currently in old interface, needs updating.]

    Keywords:

    Returns:

    """
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
    name: str,
    output_dir: pathlib.Path,
    force_field: Forcefield,
    bond_ff_scale: float,
    angle_ff_scale: float,
    max_iterations: int,
    platform: str,
) -> stk.Molecule:
    """
    Run optimisation with constraints and softened potentials.

    Keywords:

        molecule:
            The molecule to optimise.

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_constrained_ff.xml` in `output_dir`.

        output_dir:
            Directory to save outputs of optimisation process.

        force_field:
            Define the forces used in the molecule. Will be softened and
            constrained.

        bond_ff_scale:
            Scale (divide) the bond terms in the model by this value.

        angle_ff_scale:
            Scale (divide) the angle terms in the model by this value.

        max_iterations:
            Number of iterations to use in optimisation. Can be None, for
            until convergence is met.

        platform:
            Which platform to use with OpenMM optimisation. Options are
            `CPU` or `CUDA`. More are available but may not work well
            out of the box.

    Returns:

        An stk molecule.

    """

    soft_force_field = soften_force_field(
        force_field=force_field,
        bond_ff_scale=bond_ff_scale,
        angle_ff_scale=angle_ff_scale,
        output_dir=output_dir,
        prefix="soft",
    )
    soft_force_field.write_xml_file()

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
        force_field=soft_force_field,
        max_iterations=max_iterations,
        atom_constraints=intra_bb_bonds,
        platform=platform,
    )
    logging.info(f"optimising with {len(intra_bb_bonds)} constraints")
    opt_molecule = constrained_opt.optimize(molecule)
    return opt_molecule


def run_optimisation(
    molecule: stk.Molecule,
    name: str,
    file_suffix: str,
    output_dir: pathlib.Path,
    force_field: Forcefield,
    platform: str,
    max_iterations: int | None = None,
    ensemble: Ensemble | None = None,
) -> Conformer:
    """
    Run optimisation.

    Keywords:

        molecule:
            The molecule to optimise.

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_{suffix}_ff.xml` in `output_dir`.

        file_suffix:
            Suffix to use for naming output files. E.g. produces a file
            `{name}_{suffix}_ff.xml` in `output_dir`. Used to define step
            in process.

        output_dir:
            Directory to save outputs of optimisation process.

        force_field:
            Define the forces used in the molecule.

        platform:
            Which platform to use with OpenMM optimisation. Options are
            `CPU` or `CUDA`. More are available but may not work well
            out of the box.

        max_iterations:
            Number of iterations to use in optimisation. Can be None, for
            until convergence is met.

        ensemble:
            Ensemble to get the conformer id from.

    Returns:

        A Conformer.

    """

    opt = CGOMMOptimizer(
        fileprefix=f"{name}_{file_suffix}",
        output_dir=output_dir,
        force_field=force_field,
        max_iterations=max_iterations,
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


def yield_near_models(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path | str,
) -> Iterator[stk.Molecule]:
    """
    Yield structures of cage models with neighbouring force field IDs.

    Keywords:

        molecule:
            The molecule to replace.

        name:
            Name of molecule, holding force field ID.

        output_dir:
            Directory with optimisation outputs saved.

    Returns:

        An stk molecule.

    """
    ff_name = [i for i in name.split("_") if "f" in i][-1]
    ff_num = int(ff_name[1:])
    ff_range = 10

    for ff_shift in range(1, ff_range):
        for ff_option in (ff_num - ff_shift, ff_num + ff_shift):
            if ff_option < 0:
                continue
            new_name = name.replace(ff_name, f"f{ff_option}")
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
    """
    Shift beads away from cage centroid.

    Keywords:

        molecule:
            The molecule to manipulate.

        atomic_number:
            Atomic number of beads to manipulate.

        kick:
            Scale determining size of manipulation.

    Returns:

        An stk molecule.

    """
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
    force_field: Forcefield,
) -> Iterator[stk.Molecule]:
    """
    Yield conformers with atom positions of particular beads shifted.

    Keywords:

        molecule:
            The molecule to manipulate.

        force_field:
            Defines the force field.

    Yields:

        An stk molecule.

    """
    for bead in force_field.get_present_beads():
        atom_number = periodic_table()[bead.element_string]
        for kick in (1, 2, 3, 4):
            yield shift_beads(molecule, atom_number, kick)
