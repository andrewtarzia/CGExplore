#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for structure generation utilities.

Author: Andrew Tarzia

"""


import logging
import os
import pathlib
from collections.abc import Iterator

import numpy as np
import stk
from openmm import OpenMMException, openmm

from .angles import Angle
from .assigned_system import AssignedSystem
from .beads import periodic_table
from .bonds import Bond
from .ensembles import Conformer, Ensemble
from .forcefield import ForceField
from .openmm_optimizer import CGOMMDynamics, CGOMMOptimizer, OMMTrajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def optimise_ligand(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    force_field: ForceField,
    platform: str | None,
) -> stk.Molecule:
    """Optimise a building block.

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
        assigned_system = force_field.assign_terms(
            molecule=molecule,
            name=name,
            output_dir=output_dir,
        )
        opt = CGOMMOptimizer(
            fileprefix=name,
            output_dir=output_dir,
            platform=platform,
        )
        molecule = opt.optimize(assigned_system)
        molecule = molecule.with_centroid(np.array((0, 0, 0)))
        molecule.write(opt1_mol_file)
        energy_decomp = opt.read_final_energy_decomposition()
        logging.info("optimised with energy:")
        for i in energy_decomp:
            e, u = energy_decomp[i]
            logging.info(f"{i}: {round(e, 2)} [{u}]")

    return molecule


def soften_force_field(
    assigned_system: AssignedSystem,
    bond_ff_scale: float,
    angle_ff_scale: float,
    new_name: str,
) -> AssignedSystem:
    """Soften force field by scaling parameters and turning off torsions.

    Keywords:

        assigned_system:
            Molecule with force field terms assigned that will be modified.

        bond_ff_scale:
            Scale (divide) the bond terms in the model by this value.

        angle_ff_scale:
            Scale (divide) the angle terms in the model by this value.

        new_name:
            New name for system xml.

    Returns:
        New assigned system.

    """
    new_force_field_terms = {
        "bond": tuple(
            Bond(
                atom_names=i.atom_names,
                atom_ids=i.atom_ids,
                bond_r=i.bond_r,
                bond_k=i.bond_k / bond_ff_scale,
                atoms=i.atoms,
                force=i.force,
            )
            for i in assigned_system.force_field_terms["bond"]
        ),
        "angle": tuple(
            Angle(
                atom_names=i.atom_names,
                atom_ids=i.atom_ids,
                angle=i.angle,
                angle_k=i.angle_k / angle_ff_scale,
                atoms=i.atoms,
                force=i.force,
            )
            for i in assigned_system.force_field_terms["angle"]
        ),
        "torsion": (),
        "nonbonded": assigned_system.force_field_terms["nonbonded"],
    }

    new_system_xml = pathlib.Path(
        str(assigned_system.system_xml).replace(
            "_syst.xml", f"_{new_name}syst.xml"
        )
    )

    return AssignedSystem(
        molecule=assigned_system.molecule,
        force_field_terms=new_force_field_terms,
        system_xml=new_system_xml,
        topology_xml=assigned_system.topology_xml,
        bead_set=assigned_system.bead_set,
        vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
    )


def run_soft_md_cycle(
    name: str,
    assigned_system: AssignedSystem,
    output_dir: pathlib.Path,
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
    """Run MD exploration with soft potentials.

    Keywords:

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_opted1.mol` in `output_dir`.

        assigned_system:
            The system to optimise with force field assigned.

        output_dir:
            Directory to save outputs of optimisation process.

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
    soft_assigned_system = soften_force_field(
        assigned_system=assigned_system,
        bond_ff_scale=bond_ff_scale,
        angle_ff_scale=angle_ff_scale,
        new_name="softmd",
    )

    md = CGOMMDynamics(
        fileprefix=f"{name}_{suffix}",
        output_dir=output_dir,
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
        return md.run_dynamics(soft_assigned_system)

    except OpenMMException:
        return None


def run_constrained_optimisation(
    assigned_system: AssignedSystem,
    name: str,
    output_dir: pathlib.Path,
    bond_ff_scale: float,
    angle_ff_scale: float,
    max_iterations: int,
    platform: str,
) -> stk.Molecule:
    """Run optimisation with constraints and softened potentials.

    Keywords:

        assigned_system:
            The system to optimise with force field assigned.

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_constrained_ff.xml` in `output_dir`.

        output_dir:
            Directory to save outputs of optimisation process.

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
    soft_assigned_system = soften_force_field(
        assigned_system=assigned_system,
        bond_ff_scale=bond_ff_scale,
        angle_ff_scale=angle_ff_scale,
        new_name="const",
    )

    intra_bb_bonds = []
    if isinstance(assigned_system.molecule, stk.ConstructedMolecule):
        for bond_info in assigned_system.molecule.get_bond_infos():
            if bond_info.get_building_block_id() is not None:
                bond = bond_info.get_bond()
                intra_bb_bonds.append(
                    (bond.get_atom1().get_id(), bond.get_atom2().get_id())
                )

    constrained_opt = CGOMMOptimizer(
        fileprefix=f"{name}_constrained",
        output_dir=output_dir,
        max_iterations=max_iterations,
        atom_constraints=intra_bb_bonds,
        platform=platform,
    )
    logging.info(f"optimising with {len(intra_bb_bonds)} constraints")
    return constrained_opt.optimize(soft_assigned_system)


def run_optimisation(
    assigned_system: AssignedSystem,
    name: str,
    file_suffix: str,
    output_dir: pathlib.Path,
    platform: str,
    max_iterations: int | None = None,
    ensemble: Ensemble | None = None,
) -> Conformer:
    """Run optimisation.

    Keywords:

        assigned_system:
            The system to optimise with force field assigned.

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_{suffix}_ff.xml` in `output_dir`.

        file_suffix:
            Suffix to use for naming output files. E.g. produces a file
            `{name}_{suffix}_ff.xml` in `output_dir`. Used to define step
            in process.

        output_dir:
            Directory to save outputs of optimisation process.

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
        max_iterations=max_iterations,
        platform=platform,
    )
    molecule = opt.optimize(assigned_system)
    energy_decomp = opt.read_final_energy_decomposition()
    confid = None if ensemble is None else ensemble.get_num_conformers()

    return Conformer(
        molecule=molecule,
        conformer_id=confid,
        energy_decomposition=energy_decomp,
    )


def yield_near_models(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path | str,
    neighbour_library: list,
) -> Iterator[stk.Molecule]:
    """Yield structures of models with neighbouring force field parameters.

    Keywords:

        molecule:
            The molecule to modify the position matrix of.

        name:
            Name of molecule, holding force field ID.

        output_dir:
            Directory with optimisation outputs saved.

        neighbour_library:
            IDs of force fields with nearby parameters, defined in
            `define_forcefields.py`.

    Returns:
        An stk molecule.

    """
    ff_name = [i for i in name.split("_") if "f" in i][-1]

    for new_ff_id in neighbour_library:
        if new_ff_id < 0:
            continue

        new_name = name.replace(ff_name, f"f{new_ff_id}")
        new_fina_mol_file = os.path.join(output_dir, f"{new_name}_final.mol")
        if os.path.exists(new_fina_mol_file):
            logging.info(f"found neigh: {new_fina_mol_file}")
            yield molecule.with_structure_from_file(new_fina_mol_file)


def shift_beads(
    molecule: stk.Molecule,
    atomic_number: int,
    kick: float,
) -> stk.Molecule:
    """Shift beads away from cage centroid.

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
    for atom, pos in zip(molecule.get_atoms(), old_pos_mat, strict=True):
        if atom.get_atomic_number() == atomic_number:
            c_v = centroid - pos
            c_v = c_v / np.linalg.norm(c_v)
            move = c_v * kick
            new_pos = pos - move
        else:
            new_pos = pos
        new_pos_mat.append(new_pos)

    return molecule.with_position_matrix(np.array(new_pos_mat))


def yield_shifted_models(
    molecule: stk.Molecule,
    force_field: ForceField,
    kicks: tuple[int],
) -> Iterator[stk.Molecule]:
    """Yield conformers with atom positions of particular beads shifted.

    Keywords:

        molecule:
            The molecule to manipulate.

        force_field:
            Defines the force field.

        kicks:
            Defines the kicks in Angstrom to apply.

    Yields:
        An stk molecule.

    """
    for bead in force_field.get_present_beads():
        atom_number = periodic_table()[bead.element_string]
        for kick in kicks:
            yield shift_beads(molecule, atom_number, kick)
