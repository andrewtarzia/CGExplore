#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG Gulp optimizer.

Author: Andrew Tarzia

Inspired by https://bitbucket.org/4dnucleome/md_soft/src/master/

"""

import logging
import time
import stk
import pandas as pd
import numpy as np
from openmm import openmm, app
from dataclasses import dataclass

from optimizer import CGOptimizer


@dataclass
class Conformer:
    molecule: stk.Molecule
    timestep: int


class Trajectory:
    def __init__(
        self,
        base_molecule,
        data_path,
        traj_path,
        forcefield_path,
        output_path,
        temperature,
        random_seed,
        num_steps,
        time_step,
        friction,
        reporting_freq,
        traj_freq,
    ):
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

    def get_data(self):
        return pd.read_csv(self._data_path)

    def yield_conformers(self):
        raise NotImplementedError()

    def get_base_molecule(self):
        return self._base_molecule

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(steps={self._num_steps}, "
            f"conformers={self._num_confs})"
        )

    def __repr__(self) -> str:
        return str(self)


class OMMTrajectory(Trajectory):
    def __init__(
        self,
        base_molecule,
        data_path,
        traj_path,
        forcefield_path,
        output_path,
        temperature,
        random_seed,
        num_steps,
        time_step,
        friction,
        reporting_freq,
        traj_freq,
    ):
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

    def yield_conformers(self):
        num_atoms = self._base_molecule.get_num_atoms()
        start_trigger = "MODEL"
        triggered = False
        end_trigger = "ENDMDL"
        model_number = 0
        new_pos_mat = []
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

                    yield Conformer(
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


class MDEmptyTrajcetoryError(Exception):
    ...


class CGOMMOptimizer(CGOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        param_pool,
        custom_torsion_set,
        bonds,
        angles,
        torsions,
        vdw,
        max_iterations=None,
        vdw_bond_cutoff=None,
    ):
        super().__init__(
            fileprefix,
            output_dir,
            param_pool,
            bonds,
            angles,
            torsions,
            vdw,
        )
        self._custom_torsion_set = custom_torsion_set
        self._forcefield_path = output_dir / f"{fileprefix}_ff.xml"
        self._output_path = output_dir / f"{fileprefix}_omm.out"
        self._output_string = ""
        if max_iterations is None:
            self._max_iterations = 0
        else:
            self._max_iterations = max_iterations
        self._tolerance = 1e-6 * openmm.unit.kilojoules_per_mole
        if self._vdw and vdw_bond_cutoff is None:
            raise ValueError(
                "if `vdw` is on, `vdw_bond_cutoff` should be set"
            )
        elif vdw_bond_cutoff is None:
            self._vdw_bond_cutoff = 0
        else:
            self._vdw_bond_cutoff = vdw_bond_cutoff

    def _add_forces(self, system, molecule):
        if self._bonds:
            system = self._add_bonds(system, molecule)
        if self._angles:
            system = self._add_angles(system, molecule)
        if self._torsions:
            system = self._add_torsions(system, molecule)
        if self._custom_torsion_set:
            system = self._add_custom_torsions(system, molecule)
        return system

    def _group_forces(self, system):
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

    def _get_energy_decomposition(self, context, forcegroups):
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

    def _add_bonds(self, system, molecule):
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

    def _add_angles(self, system, molecule):
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

    def _yield_custom_torsions(self, molecule, chain_length=5):
        if self._custom_torsion_set is None:
            return ""

        torsions = self._get_new_torsions(molecule, chain_length)
        for torsion in torsions:
            atom1, atom2, atom3, atom4, atom5 = torsion
            names = list(
                f"{i.__class__.__name__}{i.get_id()+1}" for i in torsion
            )
            ids = list(i.get_id() for i in torsion)

            atom_estrings = list(i.__class__.__name__ for i in torsion)
            cgbeads = list(
                self._get_cgbead_from_element(i) for i in atom_estrings
            )
            cgbead_types = tuple(i.bead_type for i in cgbeads)
            if cgbead_types in self._custom_torsion_set:
                phi0 = self._custom_torsion_set[cgbead_types][0]
                torsion_k = self._custom_torsion_set[cgbead_types][1]
                torsion_n = 1
                yield (
                    names[0],
                    names[1],
                    names[3],
                    names[4],
                    ids[0],
                    ids[1],
                    ids[3],
                    ids[4],
                    torsion_k,
                    torsion_n,
                    phi0,
                )
            continue

    def _add_torsions(self, system, molecule):
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

    def _add_custom_torsions(self, system, molecule):
        force = openmm.PeriodicTorsionForce()
        system.addForce(force)

        for torsion_info in self._yield_custom_torsions(molecule):
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

    def _get_vdw_string(self, molecule, present_beads):
        nb_eqn = "sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12"
        nb_str = (
            f' <CustomNonbondedForce energy="{nb_eqn}" '
            f'bondCutoff="{self._vdw_bond_cutoff}">\n'
        )
        nb_str += '  <PerParticleParameter name="sigma"/>\n'
        nb_str += '  <PerParticleParameter name="epsilon"/>\n'
        for atype in present_beads:
            acgbead = present_beads[atype]
            nb_str += (
                f'  <Atom type="{atype}" sigma="{acgbead.sigma/10}" '
                f'epsilon="{acgbead.epsilon}"/>\n'
            )
        nb_str += " </CustomNonbondedForce>\n\n"

        return nb_str

    def _write_ff_file(self, molecule):
        ff_str = "<ForceField>\n\n"

        # logging.info("much redundancy here, can fix.")
        # logging.info(
        #     "if you use BBs as templates, not whole mol, then you "
        #     "need to change the ID"
        # )
        # logging.info(
        #     "if you use BBs as templates, not whole mol, then you "
        #     "need external bonds and to change the ID"
        # )

        at_str = " <AtomTypes>\n"
        re_str = " <Residues>\n"
        re_str += '  <Residue name="ALL">\n'

        present_beads = {}
        for atom in molecule.get_atoms():
            aestring = atom.__class__.__name__
            aid = atom.get_id()
            acgbead = self._get_cgbead_from_element(aestring)
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

            re_str += (
                f'   <Bond atomName1="{a1id}" atomName2="{a2id}"/>\n'
            )

        at_str += " </AtomTypes>\n\n"
        re_str += "  </Residue>\n"
        re_str += " </Residues>\n\n"

        ff_str += at_str
        ff_str += re_str
        # if self._bonds:
        #     ff_str += self._get_bond_string(molecule)
        # if self._angles:
        #     ff_str += self._get_angle_string(molecule)
        # if self._torsions:
        #     ff_str += self._get_torsion_string(molecule)
        if self._vdw:
            ff_str += self._get_vdw_string(molecule, present_beads)
        ff_str += "</ForceField>\n"

        with open(self._forcefield_path, "w") as f:
            f.write(ff_str)

    def _stk_to_topology(self, molecule):
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
            a_cgbead = self._get_cgbead_from_element(a_estring)

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

    def _setup_simulation(self, molecule):

        # Load force field.
        self._write_ff_file(molecule)
        forcefield = app.ForceField(self._forcefield_path)

        # Create system.
        topology = self._stk_to_topology(molecule)
        system = forcefield.createSystem(topology)
        system = self._add_forces(system, molecule)

        # Default integrator.
        time_step = 0.1 * openmm.unit.femtoseconds
        integrator = openmm.VerletIntegrator(time_step)

        # Define simulation.
        simulation = app.Simulation(topology, system, integrator)

        # Set positions from structure.
        simulation.context.setPositions(
            molecule.get_position_matrix() / 10
        )
        return simulation, system

    def _run_energy_decomp(self, simulation, system):
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

    def _output_energy_decomp(self, simulation, system):
        energy_decomp = self._run_energy_decomp(simulation, system)
        self._output_string += "energy decomposition:\n"

        for idd in energy_decomp:
            if idd == "tot_energy":
                continue
            self._output_string += f"{idd}: {energy_decomp[idd]}\n"
        self._output_string += (
            f"total energy: {energy_decomp['tot_energy']}\n"
        )
        self._output_string += "\n"

    def _get_energy(self, simulation, system):
        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )

        return state.getPotentialEnergy().in_units_of(
            openmm.unit.kilojoules_per_mole
        )

    def _minimize_energy(self, simulation, system):

        self._output_energy_decomp(simulation, system)

        self._output_string += "minimizing energy\n\n"
        simulation.minimizeEnergy(
            tolerance=self._tolerance,
            maxIterations=self._max_iterations,
        )
        self._output_energy_decomp(simulation, system)

        return simulation

    def _update_stk_molecule(self, molecule, simulation):
        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )
        positions = state.getPositions(asNumpy=True)
        molecule = molecule.with_position_matrix(positions * 10)
        return molecule

    def calculate_energy(self, molecule):
        simulation, system = self._setup_simulation(molecule)
        return self._get_energy(simulation, system)

    def calculate_energy_decomposed(self, molecule):
        simulation, system = self._setup_simulation(molecule)
        return self._run_energy_decomp(simulation, system)

    def optimize(self, molecule):
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
        fileprefix,
        output_dir,
        param_pool,
        custom_torsion_set,
        bonds,
        angles,
        torsions,
        vdw,
        temperature,
        num_steps,
        time_step,
        friction,
        reporting_freq,
        traj_freq,
        random_seed=None,
        max_iterations=None,
        vdw_bond_cutoff=None,
    ):
        super(CGOMMOptimizer, self).__init__(
            fileprefix,
            output_dir,
            param_pool,
            bonds,
            angles,
            torsions,
            vdw,
        )
        self._custom_torsion_set = custom_torsion_set
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
            raise ValueError(
                "if `vdw` is on, `vdw_bond_cutoff` should be set"
            )
        elif vdw_bond_cutoff is None:
            self._vdw_bond_cutoff = 0
        else:
            self._vdw_bond_cutoff = vdw_bond_cutoff

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

    def _add_trajectory_reporter(self, simulation):
        simulation.reporters.append(
            app.PDBReporter(
                file=str(self._trajectory_file),
                reportInterval=self._traj_freq,
            )
        )
        return simulation

    def _add_reporter(self, simulation):
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

    def _setup_simulation(self, molecule):

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
        simulation = app.Simulation(topology, system, integrator)

        # Set positions from structure.
        simulation.context.setPositions(
            molecule.get_position_matrix() / 10
        )
        return simulation, system

    def _run_molecular_dynamics(self, simulation, system):

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

    def _get_trajectory(self, molecule):

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

    def run_dynamics(self, molecule):
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
