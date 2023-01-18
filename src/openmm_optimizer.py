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
import numpy as np
from openmm import openmm, app

from optimizer import CGOptimizer


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

        logging.info("can add attractions in the future.")
        return nb_str

    def _write_ff_file(self, molecule):
        ff_str = "<ForceField>\n\n"

        logging.info("much redundancy here, can fix.")
        logging.info(
            "if you use BBs as templates, not whole mol, then you "
            "need to change the ID"
        )
        logging.info(
            "if you use BBs as templates, not whole mol, then you "
            "need external bonds and to change the ID"
        )

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
        logging.info(
            "figure out if its quicker to have many small residues or "
            "one big residue?"
        )

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
        logging.info("explicit set units here?")

        # Load force field.
        self._write_ff_file(molecule)
        forcefield = app.ForceField(self._forcefield_path)

        # Create system.
        topology = self._stk_to_topology(molecule)
        system = forcefield.createSystem(topology)
        system = self._add_forces(system, molecule)

        # Default integrator.
        logging.info("better integrator?")
        # random_seed = np.random.randint(1000)
        time_step = 0.25 * openmm.unit.femtoseconds
        temperature = 300 * openmm.unit.kelvin
        friction = 1 / openmm.unit.picosecond
        integrator = openmm.LangevinIntegrator(
            temperature, friction, time_step
        )

        # Define simulation.
        simulation = app.Simulation(topology, system, integrator)

        # Set positions from structure.
        simulation.context.setPositions(
            molecule.get_position_matrix() / 10
        )
        return simulation, system

    def _run_energy_decomp(self, simulation, system):
        self._output_string += "energy decomposition:\n"
        fgroups = self._group_forces(system)
        egroups = self._get_energy_decomposition(
            context=simulation.context,
            forcegroups=fgroups,
        )
        for idd in egroups.keys():
            self._output_string += f"{idd}: {egroups[idd]}\n"

        self._output_string += "\n"

    def _get_energy(self, simulation, system):
        self._run_energy_decomp(simulation, system)

        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )

        return state.getPotentialEnergy().in_units_of(
            openmm.unit.kilojoules_per_mole
        )

    def _minimize_energy(self, simulation, system):

        self._run_energy_decomp(simulation, system)
        simulation.minimizeEnergy(
            tolerance=self._tolerance,
            maxIterations=self._max_iterations,
        )
        self._run_energy_decomp(simulation, system)

        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )

        return state

    def _update_stk_molecule(self, molecule, state):
        positions = state.getPositions(asNumpy=True)
        molecule = molecule.with_position_matrix(positions * 10)
        return molecule

    def calculate_energy(self, molecule):
        simulation, system = self._setup_simulation(molecule)
        return self._get_energy(simulation, system)

    def optimize(self, molecule):
        start_time = time.time()
        self._output_string += f"start time: {start_time}\n"
        self._output_string += f"atoms: {molecule.get_num_atoms()}\n"

        simulation, system = self._setup_simulation(molecule)
        opt_state = self._minimize_energy(simulation, system)
        molecule = self._update_stk_molecule(molecule, opt_state)

        end_time = time.time()
        self._output_string += f"end time: {end_time}\n"
        total_time = end_time - start_time
        self._output_string += f"total time: {total_time} [s]\n"
        with open(self._output_path, "w") as f:
            f.write(self._output_string)
        return molecule
