#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG OpenMM optimizer.

Author: Andrew Tarzia

Inspired by https://bitbucket.org/4dnucleome/md_soft/src/master/

"""

import logging
import pathlib
import time
import typing

import numpy as np
import pandas as pd
import stk
from openmm import app, openmm
from openmmtools import integrators

from .beads import CgBead, get_cgbead_from_element
from .ensembles import Timestep
from .forcefield import Forcefield
from .optimizer import CGOptimizer
from .utilities import get_atom_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class OMMTrajectory:
    def __init__(
        self,
        base_molecule: stk.Molecule,
        data_path: pathlib.Path,
        traj_path: pathlib.Path,
        forcefield_path: pathlib.Path,
        output_path: pathlib.Path,
        temperature: float,
        random_seed: int,
        num_steps: int,
        time_step: float,
        friction: float | None,
        reporting_freq: float,
        traj_freq: float,
    ) -> None:
        self._base_molecule = base_molecule
        self._data_path = data_path
        self._traj_path = traj_path
        self._output_path = output_path
        self._temperature = temperature
        self._num_steps = num_steps
        self._time_step = time_step
        self._reporting_freq = reporting_freq
        self._traj_freq = traj_freq
        self._num_confs = int(self._num_steps / self._traj_freq)

        self._forcefield_path = forcefield_path
        self._random_seed = random_seed
        self._friction = friction

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self._data_path)

    def get_base_molecule(self) -> stk.Molecule:
        return self._base_molecule

    def yield_conformers(self) -> typing.Iterator[Timestep]:
        num_atoms = self._base_molecule.get_num_atoms()
        start_trigger = "MODEL"
        triggered = False
        end_trigger = "ENDMDL"
        model_number = 0
        new_pos_mat: list = []
        atom_trigger = "HETATM"

        with open(self._traj_path, "r") as f:
            for line in f.readlines():
                if end_trigger in line:
                    if len(new_pos_mat) != num_atoms:
                        raise ValueError(
                            f"num atoms ({num_atoms}) does not match "
                            "size of collected position matrix "
                            f"({len(new_pos_mat)})."
                        )

                    yield Timestep(
                        molecule=(
                            self._base_molecule.with_position_matrix(
                                np.array(new_pos_mat)
                            )
                        ),
                        timestep=model_number * self._traj_freq,
                    )
                    new_pos_mat = []
                    triggered = False

                if start_trigger in line:
                    model_number = int(line.strip().split()[-1])
                    triggered = True

                if triggered:
                    if atom_trigger in line:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        new_pos_mat.append([x, y, z])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(steps={self._num_steps}, "
            f"conformers={self._num_confs})"
        )

    def __repr__(self) -> str:
        return str(self)


class MDEmptyTrajcetoryError(Exception):
    ...


class CGOMMOptimizer(CGOptimizer):
    def __init__(
        self,
        fileprefix: str,
        output_dir: pathlib.Path,
        force_field: Forcefield,
        max_iterations: int | None = None,
        atom_constraints: typing.Iterable[tuple[int, int]] | None = None,
        platform: str | None = None,
    ) -> None:
        self._forcefield = force_field
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._topology_path = output_dir / f"{fileprefix}_ff.xml"
        self._output_path = output_dir / f"{fileprefix}_omm.out"
        self._output_string = ""
        self._properties: dict | None

        if max_iterations is None:
            self._max_iterations = 0
        else:
            self._max_iterations = max_iterations

        self._tolerance = 1e-6 * openmm.unit.kilojoules_per_mole

        if platform is not None:
            self._platform = openmm.Platform.getPlatformByName(platform)
            if platform == "CUDA":
                self._properties = {"CudaPrecision": "mixed"}
            else:
                self._properties = None
        else:
            self._platform = None
            self._properties = None

        self._atom_constraints = atom_constraints

    def _add_forces(
        self,
        system: openmm.System,
        molecule: stk.Molecule,
    ) -> openmm.System:
        if self._bonds:
            system = self._add_bonds(system, molecule)
        if self._angles:
            system = self._add_angles(system, molecule)
        if self._torsions:
            system = self._add_torsions(system, molecule)
        if self._custom_torsion_set is not None:
            system = self._add_custom_torsions(system, molecule)
        if self._atom_constraints is not None:
            system = self._add_atom_constraints(system, molecule)
        return system

    def _group_forces(self, system: openmm.System) -> dict:
        """
        Method to group forces.

        From https://github.com/XiaojuanHu/CTPOL_MD/pull/4/files

        Use these two methods as:
        `fgrps=forcegroupify(system)`
        `
        tt= getEnergyDecomposition(simulation.context, fgrps)
        for idd in tt.keys():
            print(idd,tt[idd])
        print('\n')
        `

        """

        forcegroups = {}
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force.setForceGroup(i)
            forcegroups[force] = i
        return forcegroups

    def _get_energy_decomposition(
        self,
        context: openmm.Context,
        forcegroups: dict,
    ) -> dict:
        """
        Method to decompose energies.

        From https://github.com/XiaojuanHu/CTPOL_MD/pull/4/files

        """

        energies = {}
        for f, i in forcegroups.items():
            energies[f.__class__.__name__] = context.getState(
                getEnergy=True,
                groups=2**i,
            ).getPotentialEnergy()
        return energies

    def _add_bonds(
        self,
        system: openmm.System,
        molecule: stk.Molecule,
    ) -> openmm.System:
        force = openmm.HarmonicBondForce()
        system.addForce(force)

        for bond_info in self._yield_bonds(molecule):
            name1, name2, id1, id2, bond_k, bond_r = bond_info
            force.addBond(
                particle1=id1,
                particle2=id2,
                length=bond_r / 10,
                k=bond_k,
            )

        return system

    def _add_angles(
        self,
        system: openmm.System,
        molecule: stk.Molecule,
    ) -> openmm.System:
        force = openmm.HarmonicAngleForce()
        system.addForce(force)

        for angle_info in self._yield_angles(molecule):
            (
                centre_name,
                outer_name1,
                outer_name2,
                centre_id,
                outer_id1,
                outer_id2,
                angle_k,
                angle_theta,
            ) = angle_info

            force.addAngle(
                particle1=outer_id1,
                particle2=centre_id,
                particle3=outer_id2,
                angle=np.radians(angle_theta),
                k=angle_k,
            )

        return system

    def _add_atom_constraints(
        self,
        system: openmm.System,
        molecule: stk.Molecule,
    ) -> openmm.System:
        if self._atom_constraints is None:
            return system

        self._output_string += "   constraints applied:\n"
        for constraint in self._atom_constraints:
            current_distance = get_atom_distance(
                molecule=molecule,
                atom1_id=constraint[0],
                atom2_id=constraint[1],
            )
            system.addConstraint(
                particle1=constraint[0],
                particle2=constraint[1],
                distance=current_distance / 10,
            )
            self._output_string += (
                f"{constraint[0]} {constraint[1]} "
                f"{current_distance / 10} nm\n"
            )

        self._output_string += "\n"
        return system

    def _add_torsions(
        self,
        system: openmm.System,
        molecule: stk.Molecule,
    ) -> openmm.System:
        raise NotImplementedError("Use custom torsions at this stage.")
        logging.info("warning: this interface will change in the near future")
        force = openmm.PeriodicTorsionForce()
        system.addForce(force)

        for torsion_info in self._yield_torsions(molecule):
            (
                name1,
                name2,
                name3,
                name4,
                id1,
                id2,
                id3,
                id4,
                torsion_k,
                torsion_n,
                phi0,
            ) = torsion_info

            force.addTorsion(
                particle1=id1,
                particle2=id2,
                particle3=id3,
                particle4=id4,
                periodicity=torsion_n,
                phase=np.radians(phi0),
                k=torsion_k,
            )

        return system

    def _add_custom_torsions(
        self,
        system: openmm.System,
        molecule: stk.Molecule,
    ) -> openmm.System:
        force = openmm.PeriodicTorsionForce()
        system.addForce(force)

        for torsion in self._yield_custom_torsions(molecule):
            force.addTorsion(
                particle1=torsion.atom_ids[0],
                particle2=torsion.atom_ids[1],
                particle3=torsion.atom_ids[2],
                particle4=torsion.atom_ids[3],
                periodicity=torsion.torsion_n,
                phase=np.radians(torsion.phi0),
                k=torsion.torsion_k,
            )

        return system

    def _write_xml_file(self, molecule: stk.Molecule) -> None:
        ff_str = "<ForceField>\n\n"

        at_str = " <AtomTypes>\n"
        re_str = " <Residues>\n"
        re_str += '  <Residue name="ALL">\n'

        present_beads = {}
        for atom in molecule.get_atoms():
            aestring = atom.__class__.__name__
            aid = atom.get_id()
            acgbead = get_cgbead_from_element(
                estring=aestring,
                bead_set=self._forcefield.get_bead_set(),
            )
            atype = acgbead.bead_type
            if atype not in present_beads:
                present_beads[atype] = acgbead
                at_str += (
                    f'  <Type name="{atype}" '
                    f'class="{atype}" element="{aestring}" '
                    f'mass="{self._mass}"/>\n'
                )

            re_str += f'   <Atom name="{aid}" type="{atype}"/>\n'

        for bond in molecule.get_bonds():
            a1id = bond.get_atom1().get_id()
            a2id = bond.get_atom2().get_id()

            re_str += f'   <Bond atomName1="{a1id}" atomName2="{a2id}"/>\n'

        at_str += " </AtomTypes>\n\n"
        re_str += "  </Residue>\n"
        re_str += " </Residues>\n\n"

        ff_str += at_str
        ff_str += re_str

        ff_str += "</ForceField>\n"

        with open(self._topology_path, "w") as f:
            f.write(ff_str)

    def _stk_to_topology(
        self,
        molecule: stk.Molecule,
    ) -> app.topology.Topology:
        topology = app.topology.Topology()
        chain = topology.addChain()
        residue = topology.addResidue(name="ALL", chain=chain)

        atoms_added = {}
        for atom in molecule.get_atoms():
            a_id = atom.get_id()
            a_estring = atom.__class__.__name__
            a_element = app.element.Element.getByAtomicNumber(
                atom.get_atomic_number()
            )
            a_cgbead = get_cgbead_from_element(
                estring=a_estring,
                bead_set=self._forcefield.get_bead_set(),
            )

            omm_atom = topology.addAtom(
                name=a_cgbead.bead_type,
                element=a_element,
                residue=residue,
                id=str(a_id),
            )
            atoms_added[a_id] = omm_atom

        for bond in molecule.get_bonds():
            a1_id = bond.get_atom1().get_id()
            a2_id = bond.get_atom2().get_id()

            topology.addBond(
                atom1=atoms_added[a1_id],
                atom2=atoms_added[a2_id],
            )

        return topology

    def _setup_simulation(
        self,
        molecule: stk.Molecule,
    ) -> tuple[app.Simulation, openmm.System]:
        # Load force field.
        self._write_xml_file(molecule)
        forcefield = app.ForceField(
            self._forcefield.get_path(),
            self._topology_path,
        )

        # Create system.
        topology = self._stk_to_topology(molecule)
        system = forcefield.createSystem(topology)
        # system = self._add_forces(system, molecule)
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)

        # Default integrator.
        time_step = 0.1 * openmm.unit.femtoseconds
        integrator = openmm.VerletIntegrator(time_step)

        # Define simulation.
        simulation = app.Simulation(
            topology,
            system,
            integrator,
            platform=self._platform,
            platformProperties=self._properties,
        )

        # Set positions from structure.
        simulation.context.setPositions(molecule.get_position_matrix() / 10)
        return simulation, system

    def _run_energy_decomp(
        self,
        simulation: app.Simulation,
        system: openmm.System,
    ) -> dict:
        fgroups = self._group_forces(system)
        egroups = self._get_energy_decomposition(
            context=simulation.context,
            forcegroups=fgroups,
        )

        energy_decomp = {
            "tot_energy": openmm.unit.quantity.Quantity(
                value=0,
                unit=openmm.unit.kilojoules_per_mole,
            )
        }
        for idd in egroups.keys():
            energy_decomp[idd] = egroups[idd]
            energy_decomp["tot_energy"] += egroups[idd]

        return energy_decomp

    def _output_energy_decomp(
        self,
        simulation: app.Simulation,
        system: openmm.System,
    ) -> None:
        energy_decomp = self._run_energy_decomp(simulation, system)
        self._output_string += "energy decomposition:\n"

        for idd in energy_decomp:
            if idd == "tot_energy":
                continue
            self._output_string += f"{idd}: {energy_decomp[idd]}\n"
        self._output_string += f"total energy: {energy_decomp['tot_energy']}\n"
        self._output_string += "\n"

    def _get_energy(self, simulation: app.Simulation) -> float:
        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )

        return state.getPotentialEnergy().in_units_of(
            openmm.unit.kilojoules_per_mole
        )

    def _minimize_energy(
        self,
        simulation: app.Simulation,
        system: openmm.System,
    ) -> app.Simulation:
        self._output_energy_decomp(simulation, system)

        self._output_string += "minimizing energy\n\n"
        simulation.minimizeEnergy(
            tolerance=self._tolerance,
            maxIterations=self._max_iterations,
        )
        self._output_energy_decomp(simulation, system)

        return simulation

    def _update_stk_molecule(
        self,
        molecule: stk.Molecule,
        simulation: app.Simulation,
    ) -> stk.Molecule:
        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )
        positions = state.getPositions(asNumpy=True)
        molecule = molecule.with_position_matrix(positions * 10)
        return molecule

    def calculate_energy(self, molecule: stk.Molecule) -> float:
        simulation, _ = self._setup_simulation(molecule)
        return self._get_energy(simulation)

    def read_final_energy_decomposition(self) -> dict:
        decomp_data = (
            self._output_string.split("energy decomposition:")[-1]
            .split("end time:")[0]
            .split("\n")
        )
        decomposition = {}
        for i in decomp_data:
            if i == "":
                continue
            force, value_unit = i.split(":")
            value, unit = value_unit.split()
            value = float(value)  # type: ignore[assignment]
            decomposition[force] = (value, unit)
        return decomposition

    def optimize(self, molecule: stk.Molecule) -> stk.Molecule:
        start_time = time.time()
        self._output_string += f"start time: {start_time}\n"
        self._output_string += f"atoms: {molecule.get_num_atoms()}\n"

        simulation, system = self._setup_simulation(molecule)
        simulation = self._minimize_energy(simulation, system)
        molecule = self._update_stk_molecule(molecule, simulation)

        end_time = time.time()
        self._output_string += f"end time: {end_time}\n"
        total_time = end_time - start_time
        self._output_string += f"total time: {total_time} [s]\n"
        with open(self._output_path, "w") as f:
            f.write(self._output_string)
        return molecule


class CGOMMDynamics(CGOMMOptimizer):
    def __init__(
        self,
        fileprefix: str,
        output_dir: pathlib.Path,
        bead_set: dict[str, CgBead],
        custom_torsion_set: tuple | None,
        custom_vdw_set: tuple | None,
        bonds: bool,
        angles: bool,
        torsions: bool,
        vdw: bool,
        temperature: openmm.unit.Quantity,
        num_steps: int,
        time_step: openmm.unit.Quantity,
        friction: openmm.unit.Quantity,
        reporting_freq: float,
        traj_freq: float,
        random_seed: int | None = None,
        max_iterations: int | None = None,
        vdw_bond_cutoff: int | None = None,
        atom_constraints: typing.Iterable[tuple[int, int]] | None = None,
        platform: str | None = None,
    ) -> None:
        self._bead_set = bead_set
        self._bonds = bonds
        self._angles = angles
        self._torsions = torsions
        self._vdw = vdw
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._custom_torsion_set: tuple | None = custom_torsion_set
        self._custom_vdw_set: tuple | None = custom_vdw_set
        self._forcefield_path = output_dir / f"{fileprefix}_ff.xml"
        self._output_path = output_dir / f"{fileprefix}_omm.out"
        self._trajectory_data = output_dir / f"{fileprefix}_traj.dat"
        self._trajectory_file = output_dir / f"{fileprefix}_traj.pdb"

        self._output_string = ""
        if max_iterations is None:
            self._max_iterations = 0
        else:
            self._max_iterations = max_iterations
        self._tolerance = 1e-6 * openmm.unit.kilojoules_per_mole
        if self._vdw and vdw_bond_cutoff is None:
            raise ValueError("if `vdw` is on, `vdw_bond_cutoff` should be set")
        elif vdw_bond_cutoff is None:
            self._vdw_bond_cutoff = 0
        else:
            self._vdw_bond_cutoff = vdw_bond_cutoff

        self._atom_constraints = atom_constraints

        self._temperature = temperature

        if random_seed is None:
            logging.info("a random random seed is used")
            self._random_seed = np.random.randint(1000)
        else:
            self._random_seed = random_seed

        self._num_steps = num_steps
        self._time_step = time_step
        self._friction = friction
        self._reporting_freq = reporting_freq
        self._traj_freq = traj_freq

        if platform is not None:
            self._platform = openmm.Platform.getPlatformByName(platform)
            if platform == "CUDA":
                self._properties = {"CudaPrecision": "mixed"}
            else:
                self._properties = None
        else:
            self._platform = None
            self._properties = None

    def _add_trajectory_reporter(
        self,
        simulation: app.Simulation,
    ) -> app.Simulation:
        simulation.reporters.append(
            app.PDBReporter(
                file=str(self._trajectory_file),
                reportInterval=self._traj_freq,
            )
        )
        return simulation

    def _add_reporter(self, simulation: app.Simulation) -> app.Simulation:
        simulation.reporters.append(
            app.StateDataReporter(
                file=str(self._trajectory_data),
                reportInterval=self._reporting_freq,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=False,
                temperature=True,
                volume=False,
                density=False,
                progress=False,
                remainingTime=False,
                speed=False,
                totalSteps=self._num_steps,
                separator=",",
            )
        )
        return simulation

    def _setup_simulation(
        self,
        molecule: stk.Molecule,
    ) -> tuple[app.Simulation, openmm.System]:
        # Load force field.
        self._write_ff_file(molecule)
        forcefield = app.ForceField(self._forcefield_path)

        # Create system.
        topology = self._stk_to_topology(molecule)
        system = forcefield.createSystem(topology)
        system = self._add_forces(system, molecule)

        # Default integrator.
        # logging.info("better integrator?")
        integrator = openmm.LangevinIntegrator(
            self._temperature,
            self._friction,
            self._time_step,
        )
        integrator.setRandomNumberSeed(self._random_seed)

        # Define simulation.
        simulation = app.Simulation(
            topology,
            system,
            integrator,
            platform=self._platform,
            platformProperties=self._properties,
        )

        # Set positions from structure.
        simulation.context.setPositions(molecule.get_position_matrix() / 10)
        return simulation, system

    def _run_molecular_dynamics(
        self,
        simulation: app.Simulation,
        system: openmm.System,
    ) -> app.Simulation:
        self._output_string += "simulation parameters:\n"
        self._output_string += (
            f"initalise velocities at T={self._temperature}\n"
        )
        simulation.context.setVelocitiesToTemperature(
            self._temperature,
            self._random_seed,
        )

        total_time = self._num_steps * self._time_step
        tt_in_ns = total_time.in_units_of(openmm.unit.nanoseconds)
        tf_in_ns = self._traj_freq * self._time_step
        self._output_string += (
            f"steps: {self._num_steps}\n"
            f"time step: {self._time_step}\n"
            f"total time: {tt_in_ns}\n"
            f"report frequency: {self._reporting_freq}\n"
            f"trajectory frequency per step: {self._traj_freq}\n"
            f"trajectory frequency per ns: {tf_in_ns}\n"
            f"seed: {self._random_seed}\n"
        )

        self._output_string += "running simulation\n"
        simulation = self._add_reporter(simulation)
        simulation = self._add_trajectory_reporter(simulation)

        start = time.time()
        simulation.step(self._num_steps)
        end = time.time()
        speed = self._num_steps / (end - start)
        self._output_string += (
            f"done in {end-start} s ({round(speed, 2)} steps/s)\n\n"
        )

        self._output_energy_decomp(simulation, system)

        return simulation

    def _get_trajectory(self, molecule: stk.Molecule) -> OMMTrajectory:
        return OMMTrajectory(
            base_molecule=molecule,
            data_path=self._trajectory_data,
            traj_path=self._trajectory_file,
            forcefield_path=self._forcefield_path,
            output_path=self._output_path,
            temperature=self._temperature,
            random_seed=self._random_seed,
            num_steps=self._num_steps,
            time_step=self._time_step,
            friction=self._friction,
            reporting_freq=self._reporting_freq,
            traj_freq=self._traj_freq,
        )

    def run_dynamics(self, molecule: stk.Molecule) -> OMMTrajectory:
        start_time = time.time()
        self._output_string += f"start time: {start_time}\n"
        self._output_string += f"atoms: {molecule.get_num_atoms()}\n"

        simulation, system = self._setup_simulation(molecule)
        simulation = self._minimize_energy(simulation, system)
        simulation = self._run_molecular_dynamics(simulation, system)

        end_time = time.time()
        self._output_string += f"end time: {end_time}\n"
        total_time = end_time - start_time
        self._output_string += f"total time: {total_time} [s]\n"
        with open(self._output_path, "w") as f:
            f.write(self._output_string)

        return self._get_trajectory(molecule)


class CGOMMMonteCarlo(CGOMMDynamics):
    def __init__(
        self,
        fileprefix: str,
        output_dir: pathlib.Path,
        bead_set: dict[str, CgBead],
        custom_torsion_set: tuple | None,
        custom_vdw_set: tuple | None,
        bonds: bool,
        angles: bool,
        torsions: bool,
        vdw: bool,
        temperature: float,
        num_steps: int,
        sigma: float,
        random_seed: int | None = None,
        max_iterations: int | None = None,
        vdw_bond_cutoff: int | None = None,
        atom_constraints: typing.Iterable[tuple[int, int]] | None = None,
        platform: str | None = None,
    ):
        self._bead_set = bead_set
        self._bonds = bonds
        self._angles = angles
        self._torsions = torsions
        self._vdw = vdw
        self._mass = 10
        self._bond_cutoff = 30
        self._angle_cutoff = 30
        self._torsion_cutoff = 30
        self._lj_cutoff = 10
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._custom_torsion_set: tuple | None = custom_torsion_set
        self._custom_vdw_set: tuple | None = custom_vdw_set
        self._forcefield_path = output_dir / f"{fileprefix}_ff.xml"
        self._output_path = output_dir / f"{fileprefix}_omc.out"
        self._trajectory_data = output_dir / f"{fileprefix}_traj.dat"
        self._trajectory_file = output_dir / f"{fileprefix}_traj.pdb"

        self._output_string = ""
        if max_iterations is None:
            self._max_iterations = 0
        else:
            self._max_iterations = max_iterations
        self._tolerance = 1e-6 * openmm.unit.kilojoules_per_mole
        if self._vdw and vdw_bond_cutoff is None:
            raise ValueError("if `vdw` is on, `vdw_bond_cutoff` should be set")
        elif vdw_bond_cutoff is None:
            self._vdw_bond_cutoff = 0
        else:
            self._vdw_bond_cutoff = vdw_bond_cutoff

        self._atom_constraints = atom_constraints

        self._temperature = temperature

        if random_seed is None:
            logging.info("a random random seed is used")
            self._random_seed = np.random.randint(1000)
        else:
            self._random_seed = random_seed

        self._num_steps = num_steps
        self._sigma = sigma * openmm.unit.angstroms

        # Artificial.
        self._time_step: openmm.unit.Quantity = 1.0 * openmm.unit.femtoseconds
        self._reporting_freq = 1
        self._traj_freq = 1

        if platform is not None:
            self._platform = openmm.Platform.getPlatformByName(platform)
            if platform == "CUDA":
                self._properties = {"CudaPrecision": "mixed"}
            else:
                self._properties = None
        else:
            self._platform = None
            self._properties = None

    def _setup_simulation(
        self,
        molecule: stk.Molecule,
    ) -> tuple[app.Simulation, openmm.System]:
        # Load force field.
        self._write_ff_file(molecule)
        forcefield = app.ForceField(self._forcefield_path)

        # Create system.
        topology = self._stk_to_topology(molecule)
        system = forcefield.createSystem(topology)
        system = self._add_forces(system, molecule)

        integrator = integrators.MetropolisMonteCarloIntegrator(
            self._temperature,
            self._sigma,
            # Artificial.
            self._time_step,
        )
        self._output_string += "simulation parameters:\n"
        self._output_string += f"T={self._temperature}\n"
        self._output_string += f"kT={integrator.kT}\n"

        integrator.setRandomNumberSeed(self._random_seed)

        # Define simulation.
        simulation = app.Simulation(
            topology,
            system,
            integrator,
            platform=self._platform,
            platformProperties=self._properties,
        )

        # Set positions from structure.
        simulation.context.setPositions(molecule.get_position_matrix() / 10)
        return simulation, system

    def _run_monte_carlo(
        self,
        simulation: app.Simulation,
        system: openmm.System,
    ) -> app.Simulation:
        total_time = self._num_steps * self._time_step
        tt_in_ns = total_time.in_units_of(openmm.unit.nanoseconds)
        tf_in_ns = self._traj_freq * self._time_step
        self._output_string += (
            f"steps: {self._num_steps}\n"
            f"time step: {self._time_step}\n"
            f"total time: {tt_in_ns}\n"
            f"report frequency: {self._reporting_freq}\n"
            f"trajectory frequency per step: {self._traj_freq}\n"
            f"trajectory frequency per ns: {tf_in_ns}\n"
            f"seed: {self._random_seed}\n"
        )

        self._output_string += "running simulation\n"
        simulation = self._add_reporter(simulation)
        simulation = self._add_trajectory_reporter(simulation)

        start = time.time()
        simulation.step(self._num_steps)
        end = time.time()
        speed = self._num_steps / (end - start)
        self._output_string += (
            f"done in {end-start} s ({round(speed, 2)} steps/s)\n\n"
        )

        self._output_energy_decomp(simulation, system)

        return simulation

    def _get_trajectory(self, molecule: stk.Molecule) -> OMMTrajectory:
        return OMMTrajectory(
            base_molecule=molecule,
            data_path=self._trajectory_data,
            traj_path=self._trajectory_file,
            forcefield_path=self._forcefield_path,
            output_path=self._output_path,
            temperature=self._temperature,
            random_seed=self._random_seed,
            num_steps=self._num_steps,
            time_step=self._time_step,
            friction=None,
            reporting_freq=self._reporting_freq,
            traj_freq=self._traj_freq,
        )

    def run_dynamics(self, molecule: stk.Molecule) -> OMMTrajectory:
        start_time = time.time()
        self._output_string += f"start time: {start_time}\n"
        self._output_string += f"atoms: {molecule.get_num_atoms()}\n"

        simulation, system = self._setup_simulation(molecule)
        simulation = self._run_monte_carlo(simulation, system)

        end_time = time.time()
        self._output_string += f"end time: {end_time}\n"
        total_time = end_time - start_time
        self._output_string += f"total time: {total_time} [s]\n"
        with open(self._output_path, "w") as f:
            f.write(self._output_string)

        return self._get_trajectory(molecule)
