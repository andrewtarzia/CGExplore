# Distributed under the terms of the MIT License.

"""Module for containing forcefields.

Author: Andrew Tarzia

"""

import itertools as it
import logging
import pathlib
from collections import abc
from heapq import nsmallest

import numpy as np
import stk
from openmm import openmm

from .angles import (
    Angle,
    CosineAngle,
    TargetAngle,
    TargetCosineAngle,
    TargetMartiniAngle,
    find_angles,
)
from .assigned_system import AssignedSystem, ForcedSystem, MartiniSystem
from .beads import CgBead, get_cgbead_from_element
from .bonds import Bond, TargetBond, TargetMartiniBond
from .errors import ForceFieldUnitError
from .nonbonded import Nonbonded, TargetNonbonded
from .torsions import (
    TargetMartiniTorsion,
    TargetTorsion,
    Torsion,
    find_torsions,
)
from .utilities import angle_between, convert_pyramid_angle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class ForceField:
    """Define a CG ForceField."""

    def __init__(  # noqa: PLR0913:
        self,
        identifier: str,
        prefix: str,
        present_beads: tuple[CgBead, ...],
        bond_targets: tuple[TargetBond | TargetMartiniBond, ...],
        angle_targets: tuple[TargetAngle | TargetMartiniAngle, ...],
        torsion_targets: tuple[TargetTorsion | TargetMartiniTorsion, ...],
        nonbonded_targets: tuple[TargetNonbonded, ...],
        vdw_bond_cutoff: int,
    ) -> None:
        """Initialize ForceField."""
        # Check if you should use MartiniForceField.
        for bt in bond_targets:
            if isinstance(bt, TargetMartiniBond):
                msg = (
                    f"{bt} is Martini type, probably use "
                    "MartiniForceFieldLibrary/MartiniForceField"
                )
                raise TypeError(msg)
        for at in angle_targets:
            if isinstance(at, TargetMartiniAngle):
                msg = (
                    f"{at} is Martini type, probably use "
                    "MartiniForceFieldLibrary/MartiniForceField"
                )
                raise TypeError(msg)
        for tt in torsion_targets:
            if isinstance(tt, TargetMartiniTorsion):
                msg = (
                    f"{tt} is Martini type, probably use "
                    "MartiniForceFieldLibrary/MartiniForceField"
                )
                raise TypeError(msg)

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
        found = set()
        assigned = set()

        bonds = list(molecule.get_bonds())
        bond_terms = []
        for bond in bonds:
            atoms = (bond.get_atom1(), bond.get_atom2())
            atom_estrings = [i.__class__.__name__ for i in atoms]
            cgbeads = [
                get_cgbead_from_element(i, self.get_bead_set())
                for i in atom_estrings
            ]
            cgbead_string = tuple(i.bead_type for i in cgbeads)
            found.add(cgbead_string)
            found.add(tuple(reversed(cgbead_string)))
            for target_term in self._bond_targets:
                if (target_term.type1, target_term.type2) not in (
                    cgbead_string,
                    tuple(reversed(cgbead_string)),
                ):
                    continue
                assigned.add(cgbead_string)
                assigned.add(tuple(reversed(cgbead_string)))
                if not isinstance(target_term.bond_r, openmm.unit.Quantity):
                    msg = f"{target_term} in bonds does not have units"
                    raise ForceFieldUnitError(msg)
                if not isinstance(target_term.bond_k, openmm.unit.Quantity):
                    msg = f"{target_term} in bonds does not have units"
                    raise ForceFieldUnitError(msg)

                if "Martini" in target_term.__class__.__name__:
                    force = "MartiniDefinedBond"
                    funct = target_term.funct
                else:
                    force = "HarmonicBondForce"
                    funct = 0

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
                        force=force,
                        funct=funct,
                    )
                )

        unassigned = sorted(i for i in found if i not in assigned)
        if len(unassigned) > 0:
            logging.info(f"unassigned bond terms: {unassigned}")
        return tuple(bond_terms)

    def _assign_angle_terms(  # noqa: C901, PLR0912, PLR0915
        self,
        molecule: stk.Molecule,
    ) -> tuple[Angle | CosineAngle, ...]:
        angle_terms: list[Angle | CosineAngle] = []
        pos_mat = molecule.get_position_matrix()

        found = set()
        assigned = set()
        pyramid_angles: dict[str, list] = {}
        octahedral_angles: dict[str, list] = {}
        for found_angle in find_angles(molecule):
            atom_estrings = [i.__class__.__name__ for i in found_angle.atoms]
            try:
                cgbeads = [
                    get_cgbead_from_element(i, self.get_bead_set())
                    for i in atom_estrings
                ]
            except KeyError:
                logging.info(
                    f"Angle not assigned ({found_angle}; {atom_estrings})."
                )

            cgbead_string = tuple(i.bead_type for i in cgbeads)
            found.add(cgbead_string)
            found.add(tuple(reversed(cgbead_string)))
            for target_angle in self._angle_targets:
                search_string = (
                    target_angle.type1,
                    target_angle.type2,
                    target_angle.type3,
                )
                if search_string not in (
                    cgbead_string,
                    tuple(reversed(cgbead_string)),
                ):
                    continue

                assigned.add(cgbead_string)
                assigned.add(tuple(reversed(cgbead_string)))
                if isinstance(target_angle, TargetAngle):
                    if not isinstance(
                        target_angle.angle, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_angle} in angles does not have units for"
                            " parameters"
                        )
                        raise ForceFieldUnitError(msg)
                    if not isinstance(
                        target_angle.angle_k, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_angle} in angles does not have units for"
                            " parameters"
                        )
                        raise ForceFieldUnitError(msg)

                    central_bead = cgbeads[1]
                    central_atom = list(found_angle.atoms)[1]
                    central_name = (
                        f"{atom_estrings[1]}{central_atom.get_id()+1}"
                    )
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
                    if central_bead.coordination == 4:  # noqa: PLR2004
                        if central_name not in pyramid_angles:
                            pyramid_angles[central_name] = []
                        pyramid_angles[central_name].append(actual_angle)
                    elif central_bead.coordination == 6:  # noqa: PLR2004
                        if central_name not in octahedral_angles:
                            octahedral_angles[central_name] = []
                        octahedral_angles[central_name].append(actual_angle)
                    else:
                        angle_terms.append(actual_angle)

                elif isinstance(target_angle, TargetCosineAngle):
                    if not isinstance(
                        target_angle.angle_k, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_angle} in angles does not have units for"
                            " parameters"
                        )
                        raise ForceFieldUnitError(msg)

                    central_bead = cgbeads[1]
                    central_atom = list(found_angle.atoms)[1]
                    central_name = (
                        f"{atom_estrings[1]}{central_atom.get_id()+1}"
                    )
                    angle_terms.append(
                        CosineAngle(
                            atoms=found_angle.atoms,
                            atom_names=tuple(
                                f"{i.__class__.__name__}" f"{i.get_id()+1}"
                                for i in found_angle.atoms
                            ),
                            atom_ids=found_angle.atom_ids,
                            n=target_angle.n,
                            b=target_angle.b,
                            angle_k=target_angle.angle_k,
                            force="CosinePeriodicAngleForce",
                        )
                    )

                elif isinstance(target_angle, TargetMartiniAngle):
                    if not isinstance(
                        target_angle.angle, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_angle} in angles does not have units for"
                            " parameters"
                        )
                        raise ForceFieldUnitError(msg)
                    if not isinstance(
                        target_angle.angle_k, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_angle} in angles does not have units for"
                            " parameters"
                        )
                        raise ForceFieldUnitError(msg)

                    central_bead = cgbeads[1]
                    central_atom = list(found_angle.atoms)[1]
                    central_name = (
                        f"{atom_estrings[1]}{central_atom.get_id()+1}"
                    )
                    actual_angle = Angle(
                        atoms=found_angle.atoms,
                        atom_names=tuple(
                            f"{i.__class__.__name__}" f"{i.get_id()+1}"
                            for i in found_angle.atoms
                        ),
                        atom_ids=found_angle.atom_ids,
                        angle=target_angle.angle,
                        angle_k=target_angle.angle_k,
                        force="MartiniDefinedAngle",
                        funct=target_angle.funct,
                    )
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
        unassigned = sorted(i for i in found if i not in assigned)
        if len(unassigned) > 0:
            logging.info(f"unassigned angle terms: {unassigned}")
        return tuple(angle_terms)

    def _assign_torsion_terms(
        self,
        molecule: stk.Molecule,
    ) -> tuple:
        torsion_terms = []
        assigned = set()

        # Iterate over the different path lengths, and find all torsions
        # for that lengths.
        path_lengths = {len(i.search_string) for i in self._torsion_targets}
        for pl in path_lengths:
            for found_torsion in find_torsions(molecule, pl):
                atom_estrings = [
                    i.__class__.__name__ for i in found_torsion.atoms
                ]
                cgbeads = [
                    get_cgbead_from_element(i, self.get_bead_set())
                    for i in atom_estrings
                ]
                cgbead_string = tuple(i.bead_type for i in cgbeads)
                for target_torsion in self._torsion_targets:
                    if target_torsion.search_string not in (
                        cgbead_string,
                        tuple(reversed(cgbead_string)),
                    ):
                        continue

                    assigned.add(cgbead_string)
                    assigned.add(tuple(reversed(cgbead_string)))
                    if not isinstance(
                        target_torsion.phi0, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_torsion} in torsions does not have units"
                        )
                        raise ForceFieldUnitError(msg)
                    if not isinstance(
                        target_torsion.torsion_k, openmm.unit.Quantity
                    ):
                        msg = (
                            f"{target_torsion} in torsions does not have units"
                        )
                        raise ForceFieldUnitError(msg)

                    if "Martini" in target_torsion.__class__.__name__:
                        force = "MartiniDefinedTorsion"
                        funct = target_torsion.funct
                    else:
                        force = "PeriodicTorsionForce"
                        funct = 0

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
                            force=force,
                            funct=funct,
                        )
                    )

        if len(assigned) > 0:
            logging.info(
                f"assigned torsion terms: {sorted(assigned)} "
                f"({len(self._torsion_targets)} targets) "
            )
        return tuple(torsion_terms)

    def _assign_nonbonded_terms(
        self,
        molecule: stk.Molecule,
    ) -> tuple:
        nonbonded_terms = []
        found = set()
        assigned = set()

        for atom in molecule.get_atoms():
            atom_estring = atom.__class__.__name__
            cgbead = get_cgbead_from_element(atom_estring, self.get_bead_set())
            found.add(cgbead.bead_class)
            for target_term in self._nonbonded_targets:
                if target_term.bead_class != cgbead.bead_class:
                    continue
                assigned.add(target_term.bead_class)
                if not isinstance(target_term.sigma, openmm.unit.Quantity):
                    msg = f"{target_term} in nonbondeds does not have units"
                    raise ForceFieldUnitError(msg)
                if not isinstance(target_term.epsilon, openmm.unit.Quantity):
                    msg = f"{target_term} in nonbondeds does not have units"
                    raise ForceFieldUnitError(msg)

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
        unassigned = sorted(i for i in found if i not in assigned)
        if len(unassigned) > 0:
            logging.info(f"unassigned nonbonded terms: {unassigned}")
        return tuple(nonbonded_terms)

    def assign_terms(
        self,
        molecule: stk.Molecule,
        name: str,
        output_dir: pathlib.Path,
    ) -> ForcedSystem:
        """Assign forcefield terms to molecule."""
        assigned_terms = {
            "bond": self._assign_bond_terms(molecule),
            "angle": self._assign_angle_terms(molecule),
            "torsion": self._assign_torsion_terms(molecule),
            "nonbonded": self._assign_nonbonded_terms(molecule),
        }

        # if isinstance(molecule, stk.ConstructedMolecule):
        #     for bond_info in molecule.get_bond_infos():
        #         if bond_info.get_building_block_id() is not None:
        #             possible_constraints += (
        #                 ),

        return AssignedSystem(
            molecule=molecule,
            forcefield_terms=assigned_terms,
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
        """Get beads in forcefield."""
        return {i.bead_type: i for i in self._present_beads}

    def get_identifier(self) -> str:
        """Get forcefield identifier."""
        return self._identifier

    def get_prefix(self) -> str:
        """Get forcefield prefix."""
        return self._prefix

    def get_present_beads(self) -> tuple:
        """Get beads present."""
        return self._present_beads

    def get_vdw_bond_cutoff(self) -> int:
        """Get vdW bond cutoff of forcefield."""
        return self._vdw_bond_cutoff

    def get_targets(self) -> dict:
        """Get targets of forcefield."""
        return {
            "bonds": self._bond_targets,
            "angles": self._angle_targets,
            "torsions": self._torsion_targets,
            "nonbondeds": self._nonbonded_targets,
        }

    def get_hr_file_name(self) -> str:
        """Get human-readable file name."""
        return f"{self._hrprefix}_{self._prefix}_{self._identifier}.txt"

    def write_human_readable(self, output_dir: pathlib.Path) -> None:
        """Write forcefield definition to human-readable file."""
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
        """Return a string representation of the Ensemble."""
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
        """Return a string representation of the Ensemble."""
        return str(self)


class MartiniForceField(ForceField):
    """Class defining a Martini Forcefield."""

    def __init__(  # noqa: PLR0913
        self,
        identifier: str,
        prefix: str,
        present_beads: tuple[CgBead, ...],
        bond_targets: tuple[TargetBond | TargetMartiniBond, ...],
        angle_targets: tuple[TargetAngle | TargetMartiniAngle, ...],
        torsion_targets: tuple[TargetTorsion | TargetMartiniTorsion, ...],
        constraints: tuple[tuple],
        vdw_bond_cutoff: int,
    ) -> None:
        """Initialize MartiniForceField."""
        self._identifier = identifier
        self._prefix = prefix
        self._present_beads = present_beads
        self._bond_targets = bond_targets
        self._angle_targets = angle_targets
        self._torsion_targets = torsion_targets
        self._vdw_bond_cutoff = vdw_bond_cutoff
        self._constraints = constraints
        self._hrprefix = "mffhr"

    def assign_terms(
        self,
        molecule: stk.Molecule,
        name: str,
        output_dir: pathlib.Path,
    ) -> ForcedSystem:
        """Assign forcefield terms to molecule."""
        assigned_terms = {
            "bond": self._assign_bond_terms(molecule),
            "angle": self._assign_angle_terms(molecule),
            "torsion": self._assign_torsion_terms(molecule),
            "nonbonded": (),
            "constraints": self._constraints,
        }

        return MartiniSystem(
            molecule=molecule,
            forcefield_terms=assigned_terms,
            system_xml=(
                output_dir
                / f"{name}_{self._prefix}_{self._identifier}_syst.xml"
            ),
            topology_itp=(
                output_dir
                / f"{name}_{self._prefix}_{self._identifier}_topo.itp"
            ),
            bead_set=self.get_bead_set(),
            vdw_bond_cutoff=self._vdw_bond_cutoff,
        )


class ForceFieldLibrary:
    """Define a library of forcefields with varying parameters."""

    def __init__(
        self,
        bead_library: tuple[CgBead],
        vdw_bond_cutoff: int,
        prefix: str,
    ) -> None:
        """Initialize ForceFieldLibrary."""
        self._bead_library = bead_library
        self._vdw_bond_cutoff = vdw_bond_cutoff
        self._prefix = prefix
        self._bond_ranges: tuple = ()
        self._angle_ranges: tuple = ()
        self._torsion_ranges: tuple = ()
        self._nonbonded_ranges: tuple = ()

    def add_bond_range(self, bond_range: tuple) -> None:
        """Add a range of terms to library."""
        self._bond_ranges += (bond_range,)

    def add_angle_range(self, angle_range: tuple) -> None:
        """Add a range of terms to library."""
        self._angle_ranges += (angle_range,)

    def add_torsion_range(self, torsion_range: tuple) -> None:
        """Add a range of terms to library."""
        self._torsion_ranges += (torsion_range,)

    def add_nonbonded_range(self, nonbonded_range: tuple) -> None:
        """Add a range of terms to library."""
        self._nonbonded_ranges += (nonbonded_range,)

    def _get_iterations(self) -> list:
        iterations = []
        for bond_range in self._bond_ranges:
            iterations.append(tuple(bond_range.yield_bonds()))  # noqa: PERF401

        for angle_range in self._angle_ranges:
            iterations.append(  # noqa: PERF401
                tuple(angle_range.yield_angles())
            )

        for torsion_range in self._torsion_ranges:
            iterations.append(  # noqa: PERF401
                tuple(torsion_range.yield_torsions())
            )

        for nonbonded_range in self._nonbonded_ranges:
            iterations.append(  # noqa: PERF401
                tuple(nonbonded_range.yield_nonbondeds())
            )
        return iterations

    def yield_forcefields(self) -> abc.Iterable[ForceField]:
        """Yield the forcefields in the library."""
        iterations = self._get_iterations()

        for i, parameter_set in enumerate(it.product(*iterations)):
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

            yield ForceField(
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
        """Return a string representation of the Ensemble."""
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
        """Return a string representation of the Ensemble."""
        return str(self)


class MartiniForceFieldLibrary(ForceFieldLibrary):
    """Define a library of forcefields with varying parameters."""

    def __init__(
        self,
        bead_library: tuple[CgBead],
        vdw_bond_cutoff: int,
        prefix: str,
    ) -> None:
        """Initialize MartiniForceFieldLibrary."""
        self._bead_library = bead_library
        self._vdw_bond_cutoff = vdw_bond_cutoff
        self._prefix = prefix
        self._bond_ranges: tuple = ()
        self._angle_ranges: tuple = ()
        self._torsion_ranges: tuple = ()
        self._constraints: tuple[tuple] = ()  # type: ignore[assignment]

    def _get_iterations(self) -> list:
        iterations = []
        for bond_range in self._bond_ranges:
            iterations.append(tuple(bond_range.yield_bonds()))  # noqa: PERF401

        for angle_range in self._angle_ranges:
            iterations.append(  # noqa: PERF401
                tuple(angle_range.yield_angles())
            )

        for torsion_range in self._torsion_ranges:
            iterations.append(  # noqa: PERF401
                tuple(torsion_range.yield_torsions())
            )

        return iterations

    def yield_forcefields(self) -> abc.Iterable[MartiniForceField]:
        """Yield forcefields from library."""
        iterations = self._get_iterations()

        for i, parameter_set in enumerate(it.product(*iterations)):
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
            yield MartiniForceField(
                identifier=str(i),
                prefix=self._prefix,
                present_beads=self._bead_library,
                bond_targets=bond_terms,
                angle_targets=angle_terms,
                torsion_targets=torsion_terms,
                constraints=self._constraints,
                vdw_bond_cutoff=self._vdw_bond_cutoff,
            )

    def __str__(self) -> str:
        """Return a string representation of the Ensemble."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  bead_library={self._bead_library},\n"
            f"  bond_ranges={self._bond_ranges},\n"
            f"  angle_ranges={self._angle_ranges},\n"
            f"  torsion_ranges={self._torsion_ranges},\n"
            "\n)"
        )

    def __repr__(self) -> str:
        """Return a string representation of the Ensemble."""
        return str(self)
