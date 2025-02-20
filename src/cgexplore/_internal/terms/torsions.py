# Distributed under the terms of the MIT License.

"""Module for handling torsions.

Author: Andrew Tarzia

"""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

import stk
from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass(frozen=True, slots=True)
class Torsion:
    """Class containing torsion defintion.

    Parameters:
        atom_names:
            Element and ID+1 of atoms in bond.

        atom_ids:
            ID of atoms in bond.

        phi0:
            Phase offset of torsion force.

        torsion_k:
            `k` quantity of torsion force.

        torsion_n:
            `n` integer of torsion force.

        force:
            The force to apply to a torsion.

        funct:
            For some forcefields (e.g., Martini), this term can change the
            force function.

    """

    atom_names: tuple[str, str, str, str]
    atom_ids: tuple[int, int, int, int]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int
    force: str | None
    funct: int = 0


@dataclass(frozen=True, slots=True)
class TargetTorsion:
    """Defines a target term to search for in a molecule.

    Parameters:
        search_string:
            The string of bonded beads to use to find this torsion. This is
            used by the forcefield to find the term.

        search_estring:
            The element names of the bonded beads, which should match the
            search string. This is only used for conveniance in some reporting.

        measured_atom_ids:
            Which of the searched atoms are set to have the torsion assigned to
            them. Must only be four beads. Ordering matches OpenMM definition.

        phi0:
            Phase offset to apply to torsion force.

        torsion_k:
            `k` quantity to apply to torsion force.

        torsion_n:
            `n` integer to apply to torsion force.

        funct:
            For some forcefields (e.g., Martini), this term can change the
            force function.

    """

    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int
    funct: int = 0

    def vector_key(self) -> str:
        """Return key for vector defining this target term."""
        return "".join(self.search_string)

    def vector(self) -> tuple[float, float, float]:
        """Return vector defining this target term."""
        return (
            self.phi0.value_in_unit(openmm.unit.degrees),
            self.torsion_k.value_in_unit(openmm.unit.kilojoules_per_mole),
            self.torsion_n,
        )

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{''.join(self.search_string)}, "
            f"{''.join(self.search_estring)}, "
            f"{self.measured_atom_ids!s}, "
            f"{self.phi0.in_units_of(openmm.unit.degrees)}, "
            f"{self.torsion_k.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            f"{self.torsion_n}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class TargetTorsionRange:
    """Defines a target term and ranges in parameters to search for.

    Parameters:
        search_string:
            The string of bonded beads to use to find this torsion. This is
            used by the forcefield to find the term.

        search_estring:
            The element names of the bonded beads, which should match the
            search string. This is only used for conveniance in some reporting.

        measured_atom_ids:
            Which of the searched atoms are set to have the torsion assigned to
            them. Must only be four beads. Ordering matches OpenMM definition.

        phi0s:
            Range of phase offsets to apply to torsion forces in library.

        torsion_ks:
            Range of `k` quantitys to apply to torsion forces in library.

        torsion_ns:
            Range of `n` integers to apply to torsion forces in library.

    """

    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0s: tuple[openmm.unit.Quantity]
    torsion_ks: tuple[openmm.unit.Quantity]
    torsion_ns: tuple[int]

    def yield_torsions(self) -> abc.Iterable[TargetTorsion]:
        """Find interactions matching target."""
        for phi0, k, n in it.product(
            self.phi0s, self.torsion_ks, self.torsion_ns
        ):
            yield TargetTorsion(
                search_string=self.search_string,
                search_estring=self.search_estring,
                measured_atom_ids=self.measured_atom_ids,
                phi0=phi0,
                torsion_k=k,
                torsion_n=n,
            )


@dataclass(frozen=True, slots=True)
class FoundTorsion:
    """Define a found forcefield term.

    Parameters:
        atoms:
            `stk.Atom` instances of atoms in torsion.

        atom_ids:
            ID of atoms in bond.

    """

    atoms: tuple[stk.Atom, ...]
    atom_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TargetMartiniTorsion:
    """Defines a target angle to search for in a molecule."""

    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0: openmm.unit.Quantity
    torsion_k: openmm.unit.Quantity
    torsion_n: int
    funct: int

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{''.join(self.search_string)}, "
            f"{''.join(self.search_estring)}, "
            f"{self.measured_atom_ids!s}, "
            f"{self.funct},"
            f"{self.phi0.in_units_of(openmm.unit.degrees)}, "
            f"{self.torsion_k.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            f"{self.torsion_n}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class MartiniTorsionRange:
    """Defines a target torsion and ranges in parameters to search for."""

    search_string: tuple[str, ...]
    search_estring: tuple[str, ...]
    measured_atom_ids: tuple[int, int, int, int]
    phi0s: tuple[openmm.unit.Quantity]
    torsion_ks: tuple[openmm.unit.Quantity]
    torsion_ns: tuple[int]
    funct: int

    def yield_torsions(self) -> abc.Iterable[TargetMartiniTorsion]:
        """Find torsions matching target."""
        msg = "handle torsions"
        raise NotImplementedError(msg)
        for phi0, k, n in it.product(  # type: ignore[unreachable]
            self.phi0s, self.torsion_ks, self.torsion_ns
        ):
            yield TargetMartiniTorsion(
                search_string=self.search_string,
                search_estring=self.search_estring,
                measured_atom_ids=self.measured_atom_ids,
                phi0=phi0,
                torsion_k=k,
                torsion_n=n,
                funct=self.funct,
            )
