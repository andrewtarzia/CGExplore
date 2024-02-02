# Distributed under the terms of the MIT License.

"""Module for CG OpenMM optimizer.

Inspired by https://bitbucket.org/4dnucleome/md_soft/src/master/

"""

import logging
import pathlib
import time
from collections import abc

import numpy as np
import pandas as pd
import stk
from openmm import app, openmm

from .assigned_system import ForcedSystem
from .ensembles import Timestep
from .utilities import get_atom_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class OMMTrajectory:
    """Class for holding trajectory information from OpenMM."""

    def __init__(  # noqa: PLR0913
        self,
        base_molecule: stk.Molecule,
        data_path: pathlib.Path,
        traj_path: pathlib.Path,
        output_path: pathlib.Path,
        temperature: float,
        random_seed: int,
        num_steps: int,
        time_step: float,
        friction: float | None,
        reporting_freq: float,
        traj_freq: float,
    ) -> None:
        """Initialize OMMTrajectory."""
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

        self._random_seed = random_seed
        self._friction = friction

    def get_data(self) -> pd.DataFrame:
        """Get Trajectory data."""
        return pd.read_csv(self._data_path)

    def get_base_molecule(self) -> stk.Molecule:
        """Get the base molecule of this trajectory."""
        return self._base_molecule

    def yield_conformers(self) -> abc.Iterator[Timestep]:
        """Yield conformers from trajectory."""
        num_atoms = self._base_molecule.get_num_atoms()
        start_trigger = "MODEL"
        triggered = False
        end_trigger = "ENDMDL"
        model_number = 0
        new_pos_mat: list = []
        atom_trigger = "HETATM"

        with open(self._traj_path) as f:
            for line in f.readlines():
                if end_trigger in line:
                    if len(new_pos_mat) != num_atoms:
                        msg = (
                            f"num atoms ({num_atoms}) does not match size of "
                            f"collected position matrix ({len(new_pos_mat)})."
                        )
                        raise ValueError(msg)

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

                if triggered and atom_trigger in line:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    new_pos_mat.append([x, y, z])

    def __str__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return (
            f"{self.__class__.__name__}(steps={self._num_steps}, "
            f"conformers={self._num_confs})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return str(self)


class CGOMMOptimizer:
    """Optimiser of CG models using OpenMM."""

    def __init__(  # noqa: PLR0913
        self,
        fileprefix: str,
        output_dir: pathlib.Path,
        max_iterations: int | None = None,
        atom_constraints: abc.Iterable[tuple[int, int]] | None = None,
        platform: str | None = None,
    ) -> None:
        """Initialize CGOMMOptimizer."""
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._output_path = output_dir / f"{fileprefix}_omm.out"
        self._output_string = ""
        self._properties: dict | None

        if max_iterations is None:
            self._max_iterations = 0
        else:
            self._max_iterations = max_iterations

        self._tolerance = 1e-6

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

        # Default integrator.
        time_step = 0.1 * openmm.unit.femtoseconds
        self._integrator = openmm.VerletIntegrator(time_step)

    def _group_forces(self, system: openmm.System) -> dict:
        """Method to group forces.

        From https://github.com/XiaojuanHu/CTPOL_MD/pull/4/files

        Use these two methods as:

        ```
            fgrps=forcegroupify(system)

            tt= getEnergyDecomposition(simulation.context, fgrps)
            for idd in tt.keys():
                print(idd,tt[idd])
        ```

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
        """Method to decompose energies.

        From https://github.com/XiaojuanHu/CTPOL_MD/pull/4/files

        """
        energies = {}
        for f, i in forcegroups.items():
            energies[(i, f.__class__.__name__)] = context.getState(
                getEnergy=True,
                groups=2**i,
            ).getPotentialEnergy()
        return energies

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

    def _setup_simulation(
        self,
        assigned_system: ForcedSystem,
    ) -> tuple[app.Simulation, openmm.System]:
        system = assigned_system.get_openmm_system()
        topology = assigned_system.get_openmm_topology()
        if self._atom_constraints is not None:
            system = self._add_atom_constraints(
                system, assigned_system.molecule
            )

        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)

        # Define simulation.
        simulation = app.Simulation(
            topology,
            system,
            self._integrator,
            platform=self._platform,
            platformProperties=self._properties,
        )

        # Set positions from structure.
        simulation.context.setPositions(
            assigned_system.molecule.get_position_matrix() / 10
        )

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
        for idd in egroups:
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
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True)
        return molecule.with_position_matrix(positions * 10)

    def calculate_energy(self, assigned_system: ForcedSystem) -> float:
        """Calculate energy of a system."""
        simulation, _ = self._setup_simulation(assigned_system)
        return self._get_energy(simulation)

    def read_final_energy_decomposition(self) -> dict:
        """Read the final energy decomposition in an output file."""
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

    def optimize(self, assigned_system: ForcedSystem) -> stk.Molecule:
        """Optimize a molecule."""
        start_time = time.time()
        self._output_string += f"start time: {start_time}\n"
        self._output_string += (
            f"atoms: {assigned_system.molecule.get_num_atoms()}\n"
        )

        simulation, system = self._setup_simulation(assigned_system)
        simulation = self._minimize_energy(simulation, system)
        molecule = self._update_stk_molecule(
            assigned_system.molecule, simulation
        )

        end_time = time.time()
        self._output_string += f"end time: {end_time}\n"
        total_time = end_time - start_time
        self._output_string += f"total time: {total_time} [s]\n"
        with open(self._output_path, "w") as f:
            f.write(self._output_string)
        return molecule


class CGOMMDynamics(CGOMMOptimizer):
    """Optimiser of CG models using OpenMM and Molecular Dynamics."""

    def __init__(  # noqa: PLR0913
        self,
        fileprefix: str,
        output_dir: pathlib.Path,
        temperature: openmm.unit.Quantity,
        num_steps: int,
        time_step: openmm.unit.Quantity,
        friction: openmm.unit.Quantity,
        reporting_freq: float,
        traj_freq: float,
        random_seed: int | None = None,
        max_iterations: int | None = None,
        atom_constraints: abc.Iterable[tuple[int, int]] | None = None,
        platform: str | None = None,
    ) -> None:
        """Initialize CGOMMDynamics."""
        self._fileprefix = fileprefix
        self._output_dir = output_dir
        self._output_path = output_dir / f"{fileprefix}_omm.out"
        self._trajectory_data = output_dir / f"{fileprefix}_traj.dat"
        self._trajectory_file = output_dir / f"{fileprefix}_traj.pdb"

        self._output_string = ""
        if max_iterations is None:
            self._max_iterations = 0
        else:
            self._max_iterations = max_iterations

        self._tolerance = 1e-6

        self._atom_constraints = atom_constraints

        self._temperature = temperature

        if random_seed is None:
            logging.info("a random random seed is used")
            rng = np.random.default_rng()
            self._random_seed = rng.integers(low=1, high=4000)
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

        # Default integrator.
        self._integrator = openmm.LangevinIntegrator(
            self._temperature,
            self._friction,
            self._time_step,
        )
        self._integrator.setRandomNumberSeed(self._random_seed)

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

    def _run_molecular_dynamics(
        self,
        simulation: app.Simulation,
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

        return simulation

    def _get_trajectory(self, molecule: stk.Molecule) -> OMMTrajectory:
        return OMMTrajectory(
            base_molecule=molecule,
            data_path=self._trajectory_data,
            traj_path=self._trajectory_file,
            output_path=self._output_path,
            temperature=self._temperature,
            random_seed=self._random_seed,
            num_steps=self._num_steps,
            time_step=self._time_step,
            friction=self._friction,
            reporting_freq=self._reporting_freq,
            traj_freq=self._traj_freq,
        )

    def run_dynamics(self, assigned_system: ForcedSystem) -> OMMTrajectory:
        """Run dynamics on an assigned system."""
        start_time = time.time()
        self._output_string += f"start time: {start_time}\n"
        self._output_string += (
            f"atoms: {assigned_system.molecule.get_num_atoms()}\n"
        )

        simulation, system = self._setup_simulation(assigned_system)
        simulation = self._minimize_energy(simulation, system)
        simulation = self._run_molecular_dynamics(simulation)

        self._output_energy_decomp(simulation, system)

        end_time = time.time()
        self._output_string += f"end time: {end_time}\n"
        total_time = end_time - start_time
        self._output_string += f"total time: {total_time} [s]\n"
        with open(self._output_path, "w") as f:
            f.write(self._output_string)

        return self._get_trajectory(assigned_system.molecule)
