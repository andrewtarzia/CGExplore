# Distributed under the terms of the MIT License.

"""Module for beads."""

import logging
from collections import Counter
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class CgBead:
    """Define a coarse-grained bead."""

    element_string: str
    bead_type: str
    bead_class: str
    coordination: int


def get_cgbead_from_type(
    bead_type: str,
    bead_set: dict[str, CgBead],
) -> CgBead:
    """Get CgBead class from matching to bead type."""
    return bead_set[bead_type]


def get_cgbead_from_element(
    estring: str,
    bead_set: dict[str, CgBead],
) -> CgBead:
    """Get CgBead class from matching to element string."""
    for i in bead_set:
        bead = bead_set[i]
        if bead.element_string == estring:
            return bead
    msg = f"{estring} not found in {bead_set}"
    raise ValueError(msg)


def periodic_table() -> dict[str, int]:
    """Periodic table of elements."""
    return {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
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
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
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
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
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
        "Pb": 82,
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


def bead_library_check(bead_library: tuple[CgBead]) -> None:
    """Check bead library for bad definitions."""
    logging.info(bead_library)
    logging.info(f"there are {len(bead_library)} beads")
    used_names = tuple(i.bead_class for i in bead_library)
    counts = Counter(used_names)
    for bead_class in counts:
        if counts[bead_class] > 1:
            # Check if they are different classes.
            bead_types = {
                i.bead_type for i in bead_library if i.bead_class == bead_class
            }
            if len(bead_types) == 1:
                msg = (
                    f"you used a bead ({bead_class}) twice in your library: "
                    f"{counts}"
                )
                raise ValueError(msg)
