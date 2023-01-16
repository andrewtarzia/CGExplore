#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for CG Gulp optimizer.

Author: Andrew Tarzia

"""

import logging
import numpy as np
from openmm import openmm, app

from optimizer import CGOptimizer, lorentz_berthelot_sigma_mixing
from utilities import get_all_angles, get_all_torsions


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

    def _add_forces(self, system, molecule):
        logging.info("might need this for dihedrals.")
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
            print("force", force)

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

    def _yield_bonds(self, mol):
        logging.info("merge")
        if self._bonds is False:
            return ""

        bonds = list(mol.get_bonds())
        for bond in bonds:
            atom1 = bond.get_atom1()
            name1 = f"{atom1.__class__.__name__}{atom1.get_id()+1}"
            atom2 = bond.get_atom2()
            name2 = f"{atom2.__class__.__name__}{atom2.get_id()+1}"
            estring1 = atom1.__class__.__name__
            estring2 = atom2.__class__.__name__

            try:
                cgbead1 = self._get_cgbead_from_element(estring1)
                cgbead2 = self._get_cgbead_from_element(estring2)
                bond_r = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.sigma,
                    sigma2=cgbead2.sigma,
                )
                bond_k = lorentz_berthelot_sigma_mixing(
                    sigma1=cgbead1.bond_k,
                    sigma2=cgbead2.bond_k,
                )
                yield (
                    cgbead1.bead_type,
                    cgbead2.bead_type,
                    bond_k,
                    bond_r,
                )
            except KeyError:
                logging.info(
                    f"OPT: {(name1, name2)} bond not assigned."
                )
                continue

    def _yield_angles(self, mol):
        logging.info("merge")
        if self._angles is False:
            return ""

        angles = get_all_angles(mol)

        saved_angles = {}
        for angle in angles:
            outer_atom1, centre_atom, outer_atom2 = angle
            outer_name1 = (
                f"{outer_atom1.__class__.__name__}"
                f"{outer_atom1.get_id()+1}"
            )
            centre_name = (
                f"{centre_atom.__class__.__name__}"
                f"{centre_atom.get_id()+1}"
            )
            outer_name2 = (
                f"{outer_atom2.__class__.__name__}"
                f"{outer_atom2.get_id()+1}"
            )
            outer_estring1 = outer_atom1.__class__.__name__
            centre_estring = centre_atom.__class__.__name__
            outer_estring2 = outer_atom2.__class__.__name__

            try:
                outer_cgbead1 = self._get_cgbead_from_element(
                    estring=outer_estring1,
                )
                centre_cgbead = self._get_cgbead_from_element(
                    estring=centre_estring,
                )
                outer_cgbead2 = self._get_cgbead_from_element(
                    estring=outer_estring2,
                )

                if centre_cgbead.coordination == 4:
                    raise NotImplementedError("fix")
                    if centre_name not in saved_angles:
                        saved_angles[centre_name] = []
                    saved_angles[centre_name].append(
                        (
                            centre_cgbead.angle_centered,
                            centre_cgbead.angle_k,
                            outer_atom1.bead_type,
                            outer_atom2.bead_type,
                            centre_atom.bead_type,
                        )
                    )
                    continue

                angle_theta = centre_cgbead.angle_centered
                angle_k = centre_cgbead.angle_k
                yield (
                    centre_cgbead.bead_type,
                    outer_cgbead1.bead_type,
                    outer_cgbead2.bead_type,
                    angle_k,
                    angle_theta,
                )

            except KeyError:
                logging.info(
                    f"OPT: {(outer_name1, centre_name, outer_name2)} "
                    f"angle not assigned (centered on {centre_name})."
                )
                continue

    def _yield_torsions(self, mol):
        logging.info("merge")
        if self._torsions is False:
            return ""
        logging.info("OPT: not setting torsion values yet.")
        phi0 = 0
        torsion_k = -5
        torsion_n = 1

        torsions = get_all_torsions(mol)
        for torsion in torsions:
            atom1, atom2, atom3, atom4 = torsion

            atom1_estring = atom1.__class__.__name__
            atom2_estring = atom2.__class__.__name__
            atom3_estring = atom3.__class__.__name__
            atom4_estring = atom4.__class__.__name__

            try:
                cgbead1 = self._get_cgbead_from_element(atom1_estring)
                cgbead2 = self._get_cgbead_from_element(atom2_estring)
                cgbead3 = self._get_cgbead_from_element(atom3_estring)
                cgbead4 = self._get_cgbead_from_element(atom4_estring)

                yield (
                    cgbead1.bead_type,
                    cgbead2.bead_type,
                    cgbead3.bead_type,
                    cgbead4.bead_type,
                    torsion_k,
                    torsion_n,
                    phi0,
                )

            except KeyError:
                continue

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

        hb_str = " <HarmonicBondForce>\n"
        for bond_info in self._yield_bonds(molecule):
            type1, type2, bond_k, bond_r = bond_info
            hb_str += (
                f'  <Bond type1="{type1}" type2="{type2}" '
                f'length="{bond_r/10}" k="{bond_k}"/>\n'
            )
        hb_str += " </HarmonicBondForce>\n\n"

        ha_str = " <HarmonicAngleForce>\n"
        for angle_info in self._yield_angles(molecule):
            (
                centre_type,
                outer1_type,
                outer2_type,
                angle_k,
                angle_theta,
            ) = angle_info
            angle_theta_deg = np.radians(angle_theta)
            ha_str += (
                f'  <Angle type1="{outer1_type}" type2="{centre_type}" '
                f' type3="{outer2_type}" angle="{angle_theta_deg}" '
                f'k="{angle_k}"/>\n'
            )
        ha_str += " </HarmonicAngleForce>\n\n"

        pt_str = " <PeriodicTorsionForce>\n"
        for torsion_info in self._yield_torsions(molecule):
            (
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

        ff_str += at_str
        ff_str += re_str
        if self._bonds:
            ff_str += hb_str
        if self._angles:
            ff_str += ha_str
        if self._torsions:
            ff_str += pt_str
        if self._vdw:
            ff_str += nb_str
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
        for bond in molecule.get_bonds():
            atom1 = bond.get_atom1()
            a1_id = atom1.get_id()
            a1_estring = atom1.__class__.__name__
            a1_element = app.element.Element.getByAtomicNumber(
                atom1.get_atomic_number()
            )
            a1_cgbead = self._get_cgbead_from_element(a1_estring)

            atom2 = bond.get_atom2()
            a2_id = atom2.get_id()
            a2_estring = atom2.__class__.__name__
            a2_element = app.element.Element.getByAtomicNumber(
                atom2.get_atomic_number()
            )

            a2_cgbead = self._get_cgbead_from_element(a2_estring)

            if a1_id not in atoms_added:
                omm_atom1 = topology.addAtom(
                    name=a1_cgbead.bead_type,
                    element=a1_element,
                    residue=residue,
                    id=str(a1_id),
                )
                atoms_added[a1_id] = omm_atom1

            if a2_id not in atoms_added:
                omm_atom2 = topology.addAtom(
                    name=a2_cgbead.bead_type,
                    element=a2_element,
                    residue=residue,
                    id=str(a2_id),
                )
                atoms_added[a2_id] = omm_atom2

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
        simulation.minimizeEnergy()
        self._run_energy_decomp(simulation, system)

        state = simulation.context.getState(
            getPositions=True,
            getEnergy=True,
        )

        return state

    def _update_stk_molecule(self, molecule, state):
        positions = state.getPositions(asNumpy=True)
        return molecule.with_position_matrix(positions * 10)

    def calculate_energy(self, molecule):
        simulation, system = self._setup_simulation(molecule)
        return self._get_energy(simulation, system)

    def optimize(self, molecule):

        molecule.write("t1.mol")
        simulation, system = self._setup_simulation(molecule)
        opt_state = self._minimize_energy(simulation, system)
        molecule = self._update_stk_molecule(molecule, opt_state)
        molecule.write("t2.mol")

        raise SystemExit("better to have sep dir or not?")
        return molecule
