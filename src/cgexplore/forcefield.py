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
import typing
from heapq import nsmallest

import numpy as np
import stk
from openmm import openmm

from .angles import Angle, TargetAngle, find_angles
from .beads import CgBead, get_cgbead_from_element
from .bonds import TargetBond
from .errors import ForcefieldUnitError
from .nonbonded import TargetNonbonded
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

    def yield_forcefields(self, output_path: pathlib.Path):
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
                and "Pyramid" not in i.__class__.__name__
            )
            custom_angle_terms = tuple(
                i for i in parameter_set if "Pyramid" in i.__class__.__name__
            )
            torsion_terms = tuple(
                i
                for i in parameter_set
                if "Torsion" in i.__class__.__name__
                if len(i.search_string) == 4
            )
            custom_torsion_terms = tuple(
                i
                for i in parameter_set
                if "Torsion" in i.__class__.__name__
                if len(i.search_string) != 4
            )
            nonbonded_terms = tuple(
                i for i in parameter_set if "Nonbonded" in i.__class__.__name__
            )

            yield Forcefield(
                identifier=i,
                output_dir=output_path,
                prefix=self._prefix,
                present_beads=self._bead_library,
                bond_terms=bond_terms,
                angle_terms=angle_terms,
                custom_angle_terms=custom_angle_terms,
                torsion_terms=torsion_terms,
                custom_torsion_terms=custom_torsion_terms,
                nonbonded_terms=nonbonded_terms,
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
        output_dir: pathlib.Path,
        prefix: str,
        present_beads: tuple[CgBead],
        bond_terms: tuple[TargetBond],
        angle_terms: tuple[TargetAngle],
        custom_angle_terms: tuple[TargetAngle],
        torsion_terms: tuple[TargetTorsion],
        custom_torsion_terms: tuple[TargetTorsion],
        nonbonded_terms: tuple[TargetNonbonded],
        vdw_bond_cutoff: int,
    ) -> None:
        self._identifier = identifier
        self._output_dir = output_dir
        self._prefix = prefix
        self._present_beads = present_beads
        self._bond_terms = bond_terms
        self._angle_terms = angle_terms
        self._custom_angle_terms = custom_angle_terms
        self._torsion_terms = torsion_terms
        self._custom_torsion_terms = custom_torsion_terms
        self._nonbonded_terms = nonbonded_terms
        self._vdw_bond_cutoff = vdw_bond_cutoff

    def get_xml_string(self) -> str:
        ff_str = "<ForceField>\n\n"
        ff_str += self.get_bond_string()
        ff_str += self.get_angle_string()
        ff_str += self.get_torsion_string()
        ff_str += self.get_nonbonded_string()
        ff_str += "</ForceField>\n"
        return ff_str

    def write_xml_file(self) -> None:
        ff_str = self.get_xml_string()

        with open(
            self._output_dir / f"{self._prefix}_ff_{self._identifier}.xml",
            "w",
        ) as f:
            f.write(ff_str)

    def get_bead_set(self) -> dict[str, CgBead]:
        return {i.bead_type: i for i in self._present_beads}

    def get_bond_terms(self) -> tuple:
        return self._bond_terms

    def get_angle_terms(self) -> tuple:
        return self._angle_terms

    def get_custom_angle_terms(self) -> tuple:
        return self._custom_angle_terms

    def get_torsion_terms(self) -> tuple:
        return self._torsion_terms

    def get_custom_torsion_terms(self) -> tuple:
        return self._custom_torsion_terms

    def get_nonbonded_terms(self) -> tuple:
        return self._nonbonded_terms

    def get_identifier(self) -> str:
        return self._identifier

    def get_prefix(self) -> str:
        return self._prefix

    def get_present_beads(self) -> tuple:
        return self._present_beads

    def get_path(self) -> pathlib.Path:
        return self._output_dir / f"{self._prefix}_ff_{self._identifier}.xml"

    def get_bond_string(self) -> str:
        b_str = " <HarmonicBondForce>\n"
        for term in self._bond_terms:
            try:
                length = term.bond_r.value_in_unit(openmm.unit.nanometer)
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in bonds does not have units for bond_r"
                )
            try:
                kvalue = term.bond_k.value_in_unit(
                    openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2
                )
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in bonds does not have units for bond_k"
                )
            b_str += (
                f'  <Bond class1="{term.class1}" class2="{term.class2}"'
                f' length="{length}"'
                f' k="{kvalue}"/>\n'
            )
        b_str += " </HarmonicBondForce>\n\n"

        return b_str

    def get_angle_string(self) -> str:
        b_str = " <HarmonicAngleForce>\n"
        for term in self._angle_terms:
            try:
                angle = term.angle.value_in_unit(openmm.unit.radians)
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in angles does not have units for angle"
                )
            try:
                kvalue = term.angle_k.value_in_unit(
                    openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2
                )
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in angles does not have units for angle_k"
                )
            b_str += (
                f'  <Angle class1="{term.class1}" class2="{term.class2}"'
                f' class3="{term.class3}"'
                f' angle="{angle}"'
                f' k="{kvalue}"/>\n'
            )

        b_str += " </HarmonicAngleForce>\n\n"
        return b_str

    def get_torsion_string(self) -> str:
        b_str = " <PeriodicTorsionForce>\n"
        for term in self._torsion_terms:
            class1, class2, class3, class4 = (
                s
                for i, s in enumerate(term.search_string)
                if i in term.measured_atom_ids
            )
            periodicity1 = term.torsion_n
            try:
                phase1 = term.phi0.value_in_unit(openmm.unit.radian)
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in torsions does not have units for phi0"
                )
            try:
                kvalue1 = term.torsion_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole
                )
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in torsions does not have units for torsion_k"
                )

            b_str += (
                f'  <Proper  class1="{class1}" class2="{class2}"'
                f' class3="{class3}" class4="{class4}"'
                f' periodicity1="{periodicity1}"'
                f' phase1="{phase1}"'
                f' k1="{kvalue1}"/>\n'
            )
        b_str += " </PeriodicTorsionForce>\n\n"
        return b_str

    def get_nonbonded_string(self) -> str:
        nb_eqn = "sqrt(epsilon1*epsilon2)*((sigma1+sigma2)/(2*r))^12"
        nb_str = (
            f' <CustomNonbondedForce energy="{nb_eqn}" '
            f'bondCutoff="{self._vdw_bond_cutoff}">\n'
        )
        nb_str += '  <PerParticleParameter name="sigma"/>\n'
        nb_str += '  <PerParticleParameter name="epsilon"/>\n'
        for term in self._nonbonded_terms:
            try:
                sigma = term.sigma.value_in_unit(openmm.unit.nanometer)
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in nonbondeds does not have units for sigma"
                )
            try:
                epsilon = term.epsilon.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole
                )
            except AttributeError:
                raise ForcefieldUnitError(
                    f"{term} in nonbondeds does not have units for epsilon"
                )
            nb_str += (
                f'  <Atom class="{term.bead_class}" sigma="{sigma}" '
                f'epsilon="{epsilon}"/>\n'
            )
        nb_str += " </CustomNonbondedForce>\n\n"

        return nb_str

    def yield_custom_torsions(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[Torsion]:
        # Iterate over the different path lengths, and find all torsions
        # for that lengths.
        path_lengths = set(
            len(i.search_string) for i in self._custom_torsion_terms
        )
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
                for target_torsion in self._custom_torsion_terms:
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
                            f"{target_torsion} in torsions does not have"
                            " units for parameters"
                        )

                    yield Torsion(
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
                    )

    def yield_custom_angles(
        self,
        molecule: stk.Molecule,
    ) -> typing.Iterator[Angle]:
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
            for target_angle in self._custom_angle_terms:
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
                        f"{target_angle} in custom angles does not have"
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
                    raise ValueError(
                        "Only use custom angles if coordination == 4 or"
                        " 6 at this version"
                    )

        # For four coordinate systems, apply standard angle theta to
        # neighbouring atoms, then compute pyramid angle for opposing
        # interaction.
        for central_name in pyramid_angles:
            found_angles = pyramid_angles[central_name]
            all_angles_values = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[0].get_id()],
                        v2=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[2].get_id()],
                    )
                )
                for i, X in enumerate(found_angles)
            }
            four_smallest = nsmallest(
                n=4,
                iterable=all_angles_values,
                key=all_angles_values.get,  # type: ignore[arg-type]
            )
            # Assign smallest the main angle.
            # All others, get opposite angle.
            for angle_id in all_angles_values:
                found_angle = found_angles[angle_id]
                if angle_id in four_smallest:
                    angle = found_angle.angle
                else:
                    angle = openmm.unit.Quantity(
                        value=convert_pyramid_angle(
                            found_angle.angle.value_in_unit(
                                found_angle.angle.unit
                            )
                        ),
                        unit=found_angle.angle.unit,
                    )
                yield Angle(
                    atoms=None,
                    atom_names=found_angle.atom_names,
                    atom_ids=found_angle.atom_ids,
                    angle=angle,
                    angle_k=found_angle.angle_k,
                )

        # For six coordinate systems, assume octahedral geometry.
        # So 90 degrees with 12 smallest angles, 180 degrees for the rest.
        for central_name in octahedral_angles:
            found_angles = octahedral_angles[central_name]
            all_angles_values = {
                i: np.degrees(
                    angle_between(
                        v1=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[0].get_id()],
                        v2=pos_mat[X.atoms[1].get_id()]
                        - pos_mat[X.atoms[2].get_id()],
                    )
                )
                for i, X in enumerate(found_angles)
            }
            smallest = nsmallest(
                n=12,
                iterable=all_angles_values,
                key=all_angles_values.get,  # type: ignore[arg-type]
            )

            # Assign smallest the main angle.
            # All others, get opposite angle.
            for angle_id in all_angles_values:
                found_angle = found_angles[angle_id]
                if angle_id in smallest:
                    angle = found_angle.angle
                else:
                    angle = openmm.unit.Quantity(
                        value=180,
                        unit=found_angle.angle.unit,
                    )
                yield Angle(
                    atoms=None,
                    atom_names=found_angle.atom_names,
                    atom_ids=found_angle.atom_ids,
                    angle=angle,
                    angle_k=found_angle.angle_k,
                )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  present_beads={self._present_beads}, \n"
            f"  bond_terms={len(self._bond_terms)}, \n"
            f"  angle_terms={len(self._angle_terms)}, \n"
            f"  custom_angle_terms={len(self._custom_angle_terms)}, \n"
            f"  torsion_terms={len(self._torsion_terms)}, \n"
            f"  custom_torsion_terms={len(self._custom_torsion_terms)}, \n"
            f"  nonbonded_terms={len(self._nonbonded_terms)}"
            "\n)"
        )

    def __repr__(self) -> str:
        return str(self)
