#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for system classes with assigned terms.

Author: Andrew Tarzia

"""

import os
import pathlib
from dataclasses import dataclass

import stk
from openmm import OpenMMException, app, openmm

from .beads import CgBead, get_cgbead_from_element
from .errors import ForcefieldUnavailableError, ForcefieldUnitError
from .utilities import (
    cosine_periodic_angle_force,
    custom_excluded_volume_force,
)


@dataclass(frozen=True)
class AssignedSystem:
    molecule: stk.Molecule
    force_field_terms: dict[str, tuple]
    system_xml: pathlib.Path
    topology_xml: pathlib.Path
    bead_set: dict[str, CgBead]
    vdw_bond_cutoff: int
    mass: float = 10

    def _get_topology_xml_string(self, molecule: stk.Molecule) -> str:
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
                bead_set=self.bead_set,
            )
            atype = acgbead.bead_type
            aclass = acgbead.bead_class

            if atype not in present_beads:
                present_beads[atype] = acgbead
                at_str += (
                    f'  <Type name="{atype}" '
                    f'class="{aclass}" element="{aestring}" '
                    f'mass="{self.mass}"/>\n'
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
        return ff_str

    def _write_topology_xml(self, molecule: stk.Molecule) -> None:
        ff_str = self._get_topology_xml_string(molecule)

        with open(self.topology_xml, "w") as f:
            f.write(ff_str)

    def get_openmm_topology(self) -> app.topology.Topology:
        topology = app.topology.Topology()
        chain = topology.addChain()
        residue = topology.addResidue(name="ALL", chain=chain)

        atoms_added = {}
        for atom in self.molecule.get_atoms():
            a_id = atom.get_id()
            a_estring = atom.__class__.__name__
            a_element = app.element.Element.getByAtomicNumber(
                atom.get_atomic_number()
            )
            a_cgbead = get_cgbead_from_element(
                estring=a_estring,
                bead_set=self.bead_set,
            )

            omm_atom = topology.addAtom(
                name=a_cgbead.bead_type,
                element=a_element,
                residue=residue,
                id=str(a_id),
            )
            atoms_added[a_id] = omm_atom

        for bond in self.molecule.get_bonds():
            a1_id = bond.get_atom1().get_id()
            a2_id = bond.get_atom2().get_id()

            topology.addBond(
                atom1=atoms_added[a1_id],
                atom2=atoms_added[a2_id],
            )

        return topology

    def _available_forces(self, force_type: str) -> openmm.Force:
        available = {
            "HarmonicBondForce": openmm.HarmonicBondForce(),
            "HarmonicAngleForce": openmm.HarmonicAngleForce(),
            "PeriodicTorsionForce": openmm.PeriodicTorsionForce(),
            "custom-excl-vol": custom_excluded_volume_force(),
            "CosinePeriodicAngleForce": cosine_periodic_angle_force(),
        }
        if force_type not in available:
            msg = f"{force_type} not in {available.keys()}"
            raise ForcefieldUnavailableError(msg)
        return available[force_type]

    def _add_bonds(self, system: openmm.System) -> openmm.System:
        forces = self.force_field_terms["bond"]
        force_types = {i.force for i in forces}
        for force_type in force_types:
            force_function = self._available_forces(force_type)
            system.addForce(force_function)
            for assigned_force in forces:
                if assigned_force.force != force_type:
                    continue
                try:
                    force_function.addBond(
                        particle1=assigned_force.atom_ids[0],
                        particle2=assigned_force.atom_ids[1],
                        length=assigned_force.bond_r.value_in_unit(
                            openmm.unit.nanometer
                        ),
                        k=assigned_force.bond_k.value_in_unit(
                            openmm.unit.kilojoule
                            / openmm.unit.mole
                            / openmm.unit.nanometer**2
                        ),
                    )
                except AttributeError:
                    msg = f"{assigned_force} in bonds does not have units"
                    raise ForcefieldUnitError(msg)

        return system

    def _add_angles(self, system: openmm.System) -> openmm.System:
        forces = self.force_field_terms["angle"]
        force_types = {i.force for i in forces}
        for force_type in force_types:
            force_function = self._available_forces(force_type)
            system.addForce(force_function)
            for assigned_force in forces:
                if assigned_force.force != force_type:
                    continue
                try:
                    if force_type == "CosinePeriodicAngleForce":
                        print(
                            [
                                assigned_force.angle_k.value_in_unit(
                                    openmm.unit.kilojoule / openmm.unit.mole
                                ),
                                assigned_force.n,
                                assigned_force.b,
                            ],
                        )
                        force_function.addAngle(
                            assigned_force.atom_ids[0],
                            assigned_force.atom_ids[1],
                            assigned_force.atom_ids[2],
                            # Order is important here!
                            [
                                assigned_force.angle_k.value_in_unit(
                                    openmm.unit.kilojoule / openmm.unit.mole
                                ),
                                assigned_force.n,
                                assigned_force.b,
                                (-1) ** assigned_force.n,
                            ],
                        )
                    elif force_type == "HarmonicAngleForce":
                        force_function.addAngle(
                            particle1=assigned_force.atom_ids[0],
                            particle2=assigned_force.atom_ids[1],
                            particle3=assigned_force.atom_ids[2],
                            angle=assigned_force.angle.value_in_unit(
                                openmm.unit.radian
                            ),
                            k=assigned_force.angle_k.value_in_unit(
                                openmm.unit.kilojoule
                                / openmm.unit.mole
                                / openmm.unit.radian**2
                            ),
                        )
                except AttributeError:
                    msg = f"{assigned_force} in angles does not have units"
                    raise ForcefieldUnitError(msg)

        return system

    def _add_torsions(self, system: openmm.System) -> openmm.System:
        forces = self.force_field_terms["torsion"]
        force_types = {i.force for i in forces}
        for force_type in force_types:
            force_function = self._available_forces(force_type)
            system.addForce(force_function)
            for assigned_force in forces:
                if assigned_force.force != force_type:
                    continue
                try:
                    force_function.addTorsion(
                        particle1=assigned_force.atom_ids[0],
                        particle2=assigned_force.atom_ids[1],
                        particle3=assigned_force.atom_ids[2],
                        particle4=assigned_force.atom_ids[3],
                        periodicity=assigned_force.torsion_n,
                        phase=assigned_force.phi0.value_in_unit(
                            openmm.unit.radian
                        ),
                        k=assigned_force.torsion_k.value_in_unit(
                            openmm.unit.kilojoules_per_mole
                        ),
                    )
                except AttributeError:
                    msg = f"{assigned_force} in torsions does not have units"
                    raise ForcefieldUnitError(msg)

        return system

    def _add_nonbondeds(self, system: openmm.System) -> openmm.System:
        exclusion_bonds = [
            (i.get_atom1().get_id(), i.get_atom2().get_id())
            for i in self.molecule.get_bonds()
        ]
        forces = self.force_field_terms["nonbonded"]
        force_types = {i.force for i in forces}
        for force_type in force_types:
            force_function = self._available_forces(force_type)
            system.addForce(force_function)
            for assigned_force in forces:
                if assigned_force.force != force_type:
                    continue
                try:
                    force_function.addParticle(
                        [
                            assigned_force.sigma.value_in_unit(
                                openmm.unit.nanometer
                            ),
                            assigned_force.epsilon.value_in_unit(
                                openmm.unit.kilojoules_per_mole
                            ),
                        ],
                    )

                except AttributeError:
                    msg = f"{assigned_force} in nonbondeds does not have units"
                    raise ForcefieldUnitError(msg)

            try:
                # This method MUST be after terms are assigned.
                force_function.createExclusionsFromBonds(
                    exclusion_bonds,
                    self.vdw_bond_cutoff,
                )
            except OpenMMException:
                msg = f"{force_type} is missing a definition for a particle."
                raise ForcefieldUnitError(msg)

        return system

    def _add_forces(self, system: openmm.System) -> openmm.System:
        system = self._add_bonds(system)
        system = self._add_angles(system)
        system = self._add_torsions(system)
        return self._add_nonbondeds(system)

    def get_openmm_system(self) -> openmm.System:
        if os.path.exists(self.system_xml):
            with open(self.system_xml) as f:
                return openmm.XmlSerializer.deserialize(f.read())

        self._write_topology_xml(self.molecule)
        forcefield = app.ForceField(self.topology_xml)
        topology = self.get_openmm_topology()
        system = forcefield.createSystem(topology)
        system = self._add_forces(system)

        with open(self.system_xml, "w") as f:
            f.write(openmm.XmlSerializer.serialize(system))

        return system
