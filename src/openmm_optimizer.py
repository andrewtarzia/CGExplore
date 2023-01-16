#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG Gulp optimizer.

Author: Andrew Tarzia

Inspired by https://bitbucket.org/4dnucleome/md_soft/src/master/

"""

import logging
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
        # max_iterations,
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
        # self._max_iterations = max_iterations
        self._tolerance = 1 * openmm.unit.kilojoule / openmm.unit.mole

    def _add_forces(self, system, molecule):
        logging.info("might need this for custom dihedrals.")
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
            name1, name2, cgbead1, cgbead2, bond_k, bond_r = bond_info
            id1 = int(name1[-1]) - 1
            id2 = int(name2[-1]) - 1
            force.addBond(
                particle1=id1,
                particle2=id2,
                length=bond_r / 10,
                k=bond_k,
            )

        return system

    def _get_angle_string(self, system, molecule):
        ha_str = " <HarmonicAngleForce>\n"
        raise SystemExit(
            " you need rewrite everything to use the AddForce API, not "
            "the FF because of the issue with smae types in 4C bbs."
        )
        raise SystemExit("rewrite code, then rerun parameterisation")
        done_items = set()
        for angle_info in self._yield_angles(molecule):
            (
                centre_name,
                outer_name1,
                outer_name2,
                centre_cgbead,
                outer_cgbead1,
                outer_cgbead2,
                angle_k,
                angle_theta,
            ) = angle_info

            outer1_type = outer_cgbead1.bead_type
            centre_type = centre_cgbead.bead_type
            outer2_type = outer_cgbead2.bead_type
            if (outer1_type, centre_type, outer2_type) in done_items:
                continue

            angle_theta_rad = np.radians(angle_theta)
            ha_str += (
                f'  <Angle type1="{outer1_type}" type2="{centre_type}" '
                f' type3="{outer2_type}" angle="{angle_theta_rad}" '
                f'k="{angle_k}"/>\n'
            )
            done_items.add((outer1_type, centre_type, outer2_type))

        ha_str += " </HarmonicAngleForce>\n\n"
        return ha_str

    def _yield_custom_torsions(self, molecule, chain_length=5):
        logging.info(
            "with custom torsions for omm, I think you need to fake a "
            "bond in the template. Or just add a force using the "
            "AddForce separate?"
        )
        raise SystemExit()
        if self._custom_torsion_set is None:
            return ""

        torsions = self._get_new_torsions(molecule, chain_length)
        for torsion in torsions:
            atom1, atom2, atom3, atom4, atom5 = torsion
            names = list(
                f"{i.__class__.__name__}{i.get_id()+1}" for i in torsion
            )

            atom_estrings = list(i.__class__.__name__ for i in torsion)
            cgbeads = list(
                self._get_cgbead_from_element(i) for i in atom_estrings
            )
            cgbead_types = tuple(i.bead_type for i in cgbeads)
            if cgbead_types in self._custom_torsion_set:
                phi0 = self._custom_torsion_set[cgbead_types][0]
                torsion_k = self._custom_torsion_set[cgbead_types][1]
                torsion_n = -1
                yield (
                    names[0],
                    names[1],
                    names[3],
                    names[4],
                    torsion_k,
                    torsion_n,
                    phi0,
                )
            continue

    def _get_torsion_string(self, system, molecule):
        pt_str = " <PeriodicTorsionForce>\n"

        done_items = set()
        for torsion_info in self._yield_torsions(molecule):
            (
                name1,
                name2,
                name3,
                name4,
                cgbead1,
                cgbead2,
                cgbead3,
                cgbead4,
                torsion_k,
                torsion_n,
                phi0,
            ) = torsion_info

            type1 = cgbead1.bead_type
            type2 = cgbead2.bead_type
            type3 = cgbead3.bead_type
            type4 = cgbead4.bead_type
            if (type1, type2, type3, type4) in done_items:
                continue

            pt_str += (
                f'  <Proper class1="{type1}" class2="{type2}" '
                f'class3="{type3}" class4="{type4}" '
                f'periodicity1="{torsion_n}" phase1="{phi0}" '
                f'k1="{torsion_k}"/>\n'
            )
            done_items.add((type1, type2, type3, type4))

        for torsion_info in self._yield_custom_torsions(molecule):
            (
                name1,
                name2,
                name3,
                name4,
                type1,
                type2,
                type3,
                type4,
                torsion_k,
                torsion_n,
                phi0,
            ) = torsion_info
            pt_str += (
                f'  <Proper class1="{type1}" class2="{type2}" '
                f'class3="{type3}" class4="{type4}" '
                f'periodicity1="{torsion_n}" phase1="{phi0}" '
                f'k1="{torsion_k}"/>\n'
            )

        pt_str += " </PeriodicTorsionForce>\n\n"
        return pt_str

    def _get_vdw_string(self, molecule, present_beads):
        nb_eqn = "sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12"
        nb_str = (
            f' <CustomNonbondedForce energy="{nb_eqn}" '
            'bondCutoff="0">\n'
        )
        nb_str += '  <PerParticleParameter name="sigma"/>\n'
        nb_str += '  <PerParticleParameter name="epsilon"/>\n'
        for atype in present_beads:
            acgbead = present_beads[atype]
            nb_str += (
                f'  <Atom type="{atype}" sigma="{acgbead.sigma/10}" '
                f'epsilon="{acgbead.bond_k}"/>\n'
            )
        nb_str += " </CustomNonbondedForce>\n\n"

        logging.info("can add attractions in the future.")
        return nb_str

    def _write_ff_file(self, molecule):
        ff_str = "<ForceField>\n\n"

        logging.info("much redundancy here, can fix.")

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

            logging.info(
                "if you use BBs as templates, not whole mol, then you "
                "need to change the ID"
            )
            re_str += f'   <Atom name="{aid}" type="{atype}"/>\n'

        for bond in molecule.get_bonds():
            logging.info(
                "if you use BBs as templates, not whole mol, then you "
                "need external bonds and to change the ID"
            )
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
        # raise SystemExit("better integrator?")
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
        logging.info("energy decomposition")
        fgroups = self._group_forces(system)
        egroups = self._get_energy_decomposition(
            context=simulation.context,
            forcegroups=fgroups,
        )
        for idd in egroups.keys():
            logging.info(f"{idd}: {egroups[idd]}")

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
            # maxIterations=self._max_iterations,
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
        simulation, system = self._setup_simulation(molecule)
        opt_state = self._minimize_energy(simulation, system)
        molecule = self._update_stk_molecule(molecule, opt_state)
        return molecule
