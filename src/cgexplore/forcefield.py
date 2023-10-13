#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for containing forcefields.

Author: Andrew Tarzia

"""

import itertools
import logging
import pathlib
from heapq import nsmallest

import numpy as np
import stk
from openmm import openmm

from .angles import Angle, TargetAngle, find_angles
from .assigned_system import AssignedSystem
from .beads import CgBead, get_cgbead_from_element
from .bonds import Bond, TargetBond
from .errors import ForcefieldUnitError
from .nonbonded import Nonbonded, TargetNonbonded
from .torsions import TargetTorsion, Torsion, find_torsions
from .utilities import angle_between, convert_pyramid_angle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class ForceFieldLibrary:
    def __init__(
        self,
        bead_library: tuple[CgBead],
        vdw_bond_cutoff: int,
        prefix: str,
    ) -> None:
        self._bead_library = bead_library
        self._vdw_bond_cutoff = vdw_bond_cutoff
        self._prefix = prefix
        self._bond_ranges: tuple = ()
        self._angle_ranges: tuple = ()
        self._torsion_ranges: tuple = ()
        self._nonbonded_ranges: tuple = ()

    def add_bond_range(self, bond_range: tuple) -> None:
        self._bond_ranges += (bond_range,)

    def add_angle_range(self, angle_range: tuple) -> None:
        self._angle_ranges += (angle_range,)

    def add_torsion_range(self, torsion_range: tuple) -> None:
        self._torsion_ranges += (torsion_range,)

    def add_nonbonded_range(self, nonbonded_range: tuple) -> None:
        self._nonbonded_ranges += (nonbonded_range,)

    def yield_forcefields(self):
        iterations = []

        for bond_range in self._bond_ranges:
            iterations.append(tuple(bond_range.yield_bonds()))

        for angle_range in self._angle_ranges:
            iterations.append(tuple(angle_range.yield_angles()))

        for torsion_range in self._torsion_ranges:
            iterations.append(tuple(torsion_range.yield_torsions()))

        for nonbonded_range in self._nonbonded_ranges:
            iterations.append(tuple(nonbonded_range.yield_nonbondeds()))

        for i, parameter_set in enumerate(itertools.product(*iterations)):
            bond_terms = tuple(
                i for i in parameter_set if "Bond" in i.__class__.__name__
            )
            angle_terms = tuple(
                i
                for i in parameter_set
                if "Angle" in i.__class__.__name__
                # and "Pyramid" not in i.__class__.__name__
            )
            torsion_terms = tuple(
                i
                for i in parameter_set
                if "Torsion" in i.__class__.__name__
                # if len(i.search_string) == 4
            )
            nonbonded_terms = tuple(
                i for i in parameter_set if "Nonbonded" in i.__class__.__name__
            )

            yield Forcefield(
                identifier=str(i),
                prefix=self._prefix,
                present_beads=self._bead_library,
                bond_targets=bond_terms,
                angle_targets=angle_terms,
                torsion_targets=torsion_terms,
                nonbonded_targets=nonbonded_terms,
                vdw_bond_cutoff=self._vdw_bond_cutoff,
            )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  bead_library={self._bead_library},\n"
            f"  bond_ranges={self._bond_ranges},\n"
            f"  angle_ranges={self._angle_ranges},\n"
            f"  torsion_ranges={self._torsion_ranges},\n"
            f"  nonbonded_ranges={self._nonbonded_ranges}"
            "\n)"
        )

    def __repr__(self) -> str:
        return str(self)


class Forcefield:
    def __init__(
        self,
        identifier: str,
        prefix: str,
        present_beads: tuple[CgBead, ...],
        bond_targets: tuple[TargetBond, ...],
        angle_targets: tuple[TargetAngle, ...],
        torsion_targets: tuple[TargetTorsion, ...],
        nonbonded_targets: tuple[TargetNonbonded, ...],
        vdw_bond_cutoff: int,
    ) -> None:
        self._identifier = identifier
        self._prefix = prefix
        self._present_beads = present_beads
        self._bond_targets = bond_targets
        self._angle_targets = angle_targets
        self._torsion_targets = torsion_targets
        self._nonbonded_targets = nonbonded_targets
        self._vdw_bond_cutoff = vdw_bond_cutoff
        self._hrprefix = "ffhr"

    def _assign_bond_terms(self, molecule: stk.Molecule) -> tuple:
        bonds = list(molecule.get_bonds())
        bond_terms = []
        for bond in bonds:
            atoms = (bond.get_atom1(), bond.get_atom2())
            atom_estrings = list(i.__class__.__name__ for i in atoms)
            cgbeads = list(
                get_cgbead_from_element(i, self.get_bead_set())
                for i in atom_estrings
            )
            cgbead_string = tuple(i.bead_type[0] for i in cgbeads)

            for target_term in self._bond_targets:
                if (target_term.class1, target_term.class2) not in (
                    cgbead_string,
                    tuple(reversed(cgbead_string)),
                ):
                    continue
                try:
                    assert isinstance(target_term.bond_r, openmm.unit.Quantity)
                    assert isinstance(target_term.bond_k, openmm.unit.Quantity)
                except AssertionError:
                    raise ForcefieldUnitError(
                        f"{target_term} in bonds does not have units"
                    )

                bond_terms.append(
                    Bond(
                        atoms=atoms,
                        atom_names=tuple(
                            f"{i.__class__.__name__}" f"{i.get_id()+1}"
                            for i in atoms
                        ),
                        atom_ids=tuple(i.get_id() for i in atoms),
                        bond_k=target_term.bond_k,
                        bond_r=target_term.bond_r,
                        force="HarmonicBondForce",
                    )
                )
        return tuple(bond_terms)

    def _assign_angle_terms(self, molecule: stk.Molecule) -> tuple:
        angle_terms = []
        pos_mat = molecule.get_position_matrix()

        pyramid_angles: dict[str, list] = {}
        octahedral_angles: dict[str, list] = {}
        for found_angle in find_angles(molecule):
            atom_estrings = list(
                i.__class__.__name__ for i in found_angle.atoms
            )
            try:
                cgbeads = list(
                    get_cgbead_from_element(i, self.get_bead_set())
                    for i in atom_estrings
                )
            except KeyError:
                logging.info(
                    f"Angle not assigned ({found_angle}; {atom_estrings})."
                )

            cgbead_string = tuple(i.bead_type[0] for i in cgbeads)
            for target_angle in self._angle_targets:
                search_string = (
                    target_angle.class1,
                    target_angle.class2,
                    target_angle.class3,
                )
                if search_string not in (
                    cgbead_string,
                    tuple(reversed(cgbead_string)),
                ):
                    continue

                try:
                    assert isinstance(target_angle.angle, openmm.unit.Quantity)
                    assert isinstance(
                        target_angle.angle_k, openmm.unit.Quantity
                    )
                except AssertionError:
                    raise ForcefieldUnitError(
                        f"{target_angle} in angles does not have"
                        " units for parameters"
                    )

                central_bead = cgbeads[1]
                central_atom = list(found_angle.atoms)[1]
                central_name = f"{atom_estrings[1]}{central_atom.get_id()+1}"
                actual_angle = Angle(
                    atoms=found_angle.atoms,
                    atom_names=tuple(
                        f"{i.__class__.__name__}" f"{i.get_id()+1}"
                        for i in found_angle.atoms
                    ),
                    atom_ids=found_angle.atom_ids,
                    angle=target_angle.angle,
                    angle_k=target_angle.angle_k,
                    force="HarmonicAngleForce",
                )
                if central_bead.coordination == 4:
                    if central_name not in pyramid_angles:
                        pyramid_angles[central_name] = []
                    pyramid_angles[central_name].append(actual_angle)
                elif central_bead.coordination == 6:
                    if central_name not in octahedral_angles:
                        octahedral_angles[central_name] = []
                    octahedral_angles[central_name].append(actual_angle)
                else:
                    angle_terms.append(actual_angle)

        # For four coordinate systems, apply standard angle theta to
        # neighbouring atoms, then compute pyramid angle for opposing
        # interaction.
        for central_name in pyramid_angles:
            angles = pyramid_angles[central_name]
            all_angles_values = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[0].get_id()],
                        v2=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[2].get_id()],
                    )
                )
                for i, X in enumerate(angles)
            }
            four_smallest = nsmallest(
                n=4,
                iterable=all_angles_values,
                key=all_angles_values.get,  # type: ignore[arg-type]
            )
            # Assign smallest the main angle.
            # All others, get opposite angle.
            for angle_id in all_angles_values:
                current_angle = angles[angle_id]
                if angle_id in four_smallest:
                    angle = current_angle.angle
                else:
                    angle = openmm.unit.Quantity(
                        value=convert_pyramid_angle(
                            current_angle.angle.value_in_unit(
                                current_angle.angle.unit
                            )
                        ),
                        unit=current_angle.angle.unit,
                    )
                angle_terms.append(
                    Angle(
                        atoms=None,
                        atom_names=current_angle.atom_names,
                        atom_ids=current_angle.atom_ids,
                        angle=angle,
                        angle_k=current_angle.angle_k,
                        force="HarmonicAngleForce",
                    ),
                )

        # For six coordinate systems, assume octahedral geometry.
        # So 90 degrees with 12 smallest angles, 180 degrees for the rest.
        for central_name in octahedral_angles:
            angles = octahedral_angles[central_name]
            all_angles_values = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[0].get_id()],
                        v2=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[2].get_id()],
                    )
                )
                for i, X in enumerate(angles)
            }
            smallest = nsmallest(
                n=12,
                iterable=all_angles_values,
                key=all_angles_values.get,  # type: ignore[arg-type]
            )

            # Assign smallest the main angle.
            # All others, get opposite angle.
            for angle_id in all_angles_values:
                current_angle = angles[angle_id]
                if angle_id in smallest:
                    angle = current_angle.angle
                else:
                    angle = openmm.unit.Quantity(
                        value=180,
                        unit=current_angle.angle.unit,
                    )
                angle_terms.append(
                    Angle(
                        atoms=None,
                        atom_names=current_angle.atom_names,
                        atom_ids=current_angle.atom_ids,
                        angle=angle,
                        angle_k=current_angle.angle_k,
                        force="HarmonicAngleForce",
                    ),
                )
        return tuple(angle_terms)

    def _assign_torsion_terms(
        self,
        molecule: stk.Molecule,
    ) -> tuple:
        torsion_terms = []

        # Iterate over the different path lengths, and find all torsions
        # for that lengths.
        path_lengths = set(len(i.search_string) for i in self._torsion_targets)
        for pl in path_lengths:
            for found_torsion in find_torsions(molecule, pl):
                atom_estrings = list(
                    i.__class__.__name__ for i in found_torsion.atoms
                )
                cgbeads = list(
                    get_cgbead_from_element(i, self.get_bead_set())
                    for i in atom_estrings
                )
                cgbead_string = tuple(i.bead_type[0] for i in cgbeads)
                for target_torsion in self._torsion_targets:
                    if target_torsion.search_string not in (
                        cgbead_string,
                        tuple(reversed(cgbead_string)),
                    ):
                        continue

                    try:
                        assert isinstance(
                            target_torsion.phi0, openmm.unit.Quantity
                        )
                        assert isinstance(
                            target_torsion.torsion_k, openmm.unit.Quantity
                        )
                    except AssertionError:
                        raise ForcefieldUnitError(
                            f"{target_torsion} in torsions does not have units"
                        )

                    torsion_terms.append(
                        Torsion(
                            atom_names=tuple(
                                f"{found_torsion.atoms[i].__class__.__name__}"
                                f"{found_torsion.atoms[i].get_id()+1}"
                                for i in target_torsion.measured_atom_ids
                            ),
                            atom_ids=tuple(
                                found_torsion.atoms[i].get_id()
                                for i in target_torsion.measured_atom_ids
                            ),
                            phi0=target_torsion.phi0,
                            torsion_n=target_torsion.torsion_n,
                            torsion_k=target_torsion.torsion_k,
                            force="PeriodicTorsionForce",
                        )
                    )
        return tuple(torsion_terms)

    def _assign_nonbonded_terms(
        self,
        molecule: stk.Molecule,
    ) -> tuple:
        nonbonded_terms = []

        for atom in molecule.get_atoms():
            atom_estring = atom.__class__.__name__
            cgbead = get_cgbead_from_element(atom_estring, self.get_bead_set())

            for target_term in self._nonbonded_targets:
                if target_term.bead_class != cgbead.bead_class:
                    continue
                try:
                    assert isinstance(target_term.sigma, openmm.unit.Quantity)
                    assert isinstance(
                        target_term.epsilon, openmm.unit.Quantity
                    )
                except AssertionError:
                    raise ForcefieldUnitError(
                        f"{target_term} in nonbondeds does not have units"
                    )

                nonbonded_terms.append(
                    Nonbonded(
                        atom_id=atom.get_id(),
                        bead_class=cgbead.bead_class,
                        bead_element=atom_estring,
                        sigma=target_term.sigma,
                        epsilon=target_term.epsilon,
                        force=target_term.force,
                    )
                )

        return tuple(nonbonded_terms)

    def assign_terms(
        self,
        molecule: stk.Molecule,
        name: str,
        output_dir: pathlib.Path,
    ) -> AssignedSystem:
        assigned_terms = {
            "bond": self._assign_bond_terms(molecule),
            "angle": self._assign_angle_terms(molecule),
            "torsion": self._assign_torsion_terms(molecule),
            "nonbonded": self._assign_nonbonded_terms(molecule),
        }

        # possible_constraints = ()
        # if isinstance(molecule, stk.ConstructedMolecule):
        #     for bond_info in molecule.get_bond_infos():
        #         if bond_info.get_building_block_id() is not None:
        #             bond = bond_info.get_bond()
        #             possible_constraints += (
        #                 (
        #                     bond.get_atom1().get_id(),
        #                     bond.get_atom2().get_id(),
        #                 ),
        #             )
        # print(len(possible_constraints))

        return AssignedSystem(
            molecule=molecule,
            force_field_terms=assigned_terms,
            system_xml=(
                output_dir
                / f"{name}_{self._prefix}_{self._identifier}_syst.xml"
            ),
            topology_xml=(
                output_dir
                / f"{name}_{self._prefix}_{self._identifier}_topo.xml"
            ),
            bead_set=self.get_bead_set(),
            vdw_bond_cutoff=self._vdw_bond_cutoff,
        )

    def get_bead_set(self) -> dict[str, CgBead]:
        return {i.bead_type: i for i in self._present_beads}

    def get_identifier(self) -> str:
        return self._identifier

    def get_prefix(self) -> str:
        return self._prefix

    def get_present_beads(self) -> tuple:
        return self._present_beads

    def get_vdw_bond_cutoff(self) -> int:
        return self._vdw_bond_cutoff

    def get_targets(self) -> dict:
        return {
            "bonds": self._bond_targets,
            "angles": self._angle_targets,
            "torsions": self._torsion_targets,
            "nonbondeds": self._nonbonded_targets,
        }

    def get_hr_file_name(self) -> str:
        return f"{self._hrprefix}_{self._prefix}_{self._identifier}.txt"

    def write_human_readable(self, output_dir) -> None:
        with open(output_dir / self.get_hr_file_name(), "w") as f:
            f.write(f"prefix: {self._prefix}\n")
            f.write(f"identifier: {self._identifier}\n")
            f.write(f"vdw bond cut off: {self._vdw_bond_cutoff}\n")
            f.write("present beads:\n")
            for i in self._present_beads:
                f.write(f"{i} \n")

            f.write("\nbonds:\n")
            for bt in self._bond_targets:
                f.write(f"{bt.human_readable()} \n")

            f.write("\nangles:\n")
            for at in self._angle_targets:
                f.write(f"{at.human_readable()} \n")

            f.write("\ntorsions:\n")
            for tt in self._torsion_targets:
                f.write(f"{tt.human_readable()} \n")

            f.write("\nnobondeds:\n")
            for nt in self._nonbonded_targets:
                f.write(f"{nt.human_readable()} \n")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  present_beads={self._present_beads}, \n"
            f"  bond_targets={len(self._bond_targets)}, \n"
            f"  angle_targets={len(self._angle_targets)}, \n"
            f"  torsion_targets={len(self._torsion_targets)}, \n"
            f"  nonbonded_targets={len(self._nonbonded_targets)}"
            "\n)"
        )

    def __repr__(self) -> str:
        return str(self)
