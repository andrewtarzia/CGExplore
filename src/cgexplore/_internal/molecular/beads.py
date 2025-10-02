"""Module for beads."""

import logging
import typing
from collections import Counter, abc
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CgBead:
    element_string: str
    bead_type: str
    bead_class: str
    coordination: int


class BeadLibrary:
    """Define a library of beads used in a model."""

    def __init__(self, beads: abc.Sequence[CgBead]) -> None:
        """Initialize a BeadLibrary."""
        self._beads = beads
        # Run a check.
        self._bead_library_check()
        self._from_type_map = {i.bead_type: i for i in self._beads}
        self._from_class_map = {i.bead_class: i for i in self._beads}
        self._from_element_map = {i.element_string: i for i in self._beads}

    @classmethod
    def from_bead_types(cls, bead_types: dict[str, int]) -> typing.Self:
        """Create a BeadLibrary from a dictionary of bead types.

        Parameters:
            bead_types:
                Dictionary mapping between bead types (user nomenclature) and
                their expected coordination.

        """
        possible_enames = {}
        beads = []
        for bt, coord in bead_types.items():
            element_strings = _get_estrings(coord)
            for element_string in element_strings:
                if element_string not in set(possible_enames.values()):
                    possible_enames[bt] = element_string
                    beads.append(
                        CgBead(
                            element_string=element_string,
                            bead_type=bt,
                            bead_class=bt,
                            coordination=coord,
                        )
                    )
                    break
        return cls(beads)

    def _bead_library_check(self) -> None:
        """Check bead library for bad definitions."""
        used_names = tuple(i.bead_class for i in self._beads)
        counts = Counter(used_names)
        for bead_class in counts:
            if counts[bead_class] > 1:
                # Check if they are different classes.
                bead_types = {
                    i.bead_type
                    for i in self._beads
                    if i.bead_class == bead_class
                }
                if len(bead_types) == 1:
                    msg = (
                        f"Bead ({bead_class}) shows twice in your library:"
                        f" {counts}.\n"
                        f"Library: {self._beads} ({len(self._beads)} beads)"
                    )
                    raise ValueError(msg)

    def get_from_type(self, bead_type: str) -> CgBead:
        """Get CgBead from matching to bead type."""
        return self._from_type_map[bead_type]

    def get_from_element(self, estring: str) -> CgBead:
        """Get CgBead from matching to element string."""
        return self._from_element_map[estring]

    def get_from_class(self, bead_class: str) -> CgBead:
        """Get CgBead from matching to bead class."""
        return self._from_class_map[bead_class]

    def get_present_beads(self) -> list[CgBead]:
        """Get a list of present beads."""
        return list(self._beads)

    def __str__(self) -> str:
        """Return a string representation of the Ensemble."""
        bead_string = tuple(
            [i.bead_type + "=" + i.element_string for i in self._beads]
        )
        return f"{self.__class__.__name__}(\n  beads={bead_string}, \n)"

    def __repr__(self) -> str:
        """Return a string representation of the Ensemble."""
        return str(self)


def periodic_table() -> dict[str, int]:
    """Periodic table of elements."""
    return {
        # Reordered to put safe elements at the top.
        "Pd": 46,
        "Ag": 47,
        "Ba": 56,
        "Pb": 82,
        "Fe": 26,
        "Ga": 31,
        "Ni": 28,
        "C": 6,
        "N": 7,
        "O": 8,
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Co": 27,
        "Cu": 29,
        "Zn": 30,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Pa": 91,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Cf": 98,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Db": 105,
    }


def string_to_atom_number(string: str) -> int:
    """Convert atom string to atom number."""
    return periodic_table()[string]


def _get_estrings(coordination: int) -> list[str]:
    """Get element strings for a given coordination."""
    periodic_table_exclusions = {
        0: ("Pd",),
        1: ("Pd",),
        2: ("Pd", "H"),
        3: ("Pd", "H"),
        4: ("H",),
        5: ("Pd", "H"),
        6: ("Pd", "H"),
    }
    return [
        element
        for element in periodic_table()
        if element not in periodic_table_exclusions[coordination]
    ]
