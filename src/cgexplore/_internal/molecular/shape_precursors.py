# Distributed under the terms of the MIT License.

"""Classes of precursors with no bonding.

Author: Andrew Tarzia

"""

from dataclasses import dataclass

import numpy as np
import stk
from scipy.constants import golden

from .beads import CgBead, string_to_atom_number


def beads_to_model(
    present_beads: tuple[CgBead, ...],
    coordinates: np.ndarray,
) -> stk.Molecule:
    atoms = tuple(
        stk.Atom(i, string_to_atom_number(present_beads[i].element_string))
        for i in range(len(present_beads))
    )
    return stk.BuildingBlock.init(
        atoms=tuple(atoms),
        bonds=(),
        position_matrix=np.array(coordinates),
    )


@dataclass
class ShapePrecursor:
    """Define a shape precursor."""

    distance: float
    present_beads: tuple[CgBead, ...]

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


@dataclass
class DiatomShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = [
            np.array([0, 0, 0]),
            np.array([1 * self.distance, 0, 0]),
        ]
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class TriangleShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = self.distance * 2 * np.sqrt(3) / 4
        _y = self.distance * 2
        coordinates = np.array(
            (
                np.array([0, _x, 0]),
                np.array([_y / 2, -_x, 0]),
                np.array([-_y / 2, -_x, 0]),
            )
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class SquareShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = (
            np.array(
                (
                    np.array([1, 1, 0]),
                    np.array([1, -1, 0]),
                    np.array([-1, -1, 0]),
                    np.array([-1, 1, 0]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class StarShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = np.cos(np.radians(60))
        _y = np.sin(np.radians(60))
        _c1 = np.cos(2 * np.pi / 5)
        _c2 = np.cos(np.pi / 5)
        _s1 = np.sin(2 * np.pi / 5)
        _s2 = np.sin(4 * np.pi / 5)

        coordinates = (
            np.array(
                (
                    np.array([0, 1, 0]),
                    np.array([_s1, _c1, 0]),
                    np.array([_s2, -_c2, 0]),
                    np.array([-_s2, -_c2, 0]),
                    np.array([-_s1, _c1, 0]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class HexagonShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = np.cos(np.radians(60))
        _y = np.sin(np.radians(60))
        coordinates = (
            np.array(
                (
                    np.array([1, 0, 0]),
                    np.array([_x, _y, 0]),
                    np.array([-_x, _y, 0]),
                    np.array([-1, 0, 0]),
                    np.array([-_x, -_y, 0]),
                    np.array([_x, -_y, 0]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class TdShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        # Vertices of a tetrahdron so that origin is at the origin. Source:
        # http://tinyurl.com/lc262h8.
        coordinates = (
            np.array(
                (
                    np.array([0, 0, np.sqrt(6) / 2]),
                    np.array([-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
                    np.array([1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
                    np.array([0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class CuShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = (
            np.array(
                (
                    np.array([-1, 1, -1]),
                    np.array([-1, -1, -1]),
                    np.array([1, 1, -1]),
                    np.array([1, -1, -1]),
                    np.array([-1, 1, 1]),
                    np.array([-1, -1, 1]),
                    np.array([1, 1, 1]),
                    np.array([1, -1, 1]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class OcShape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = 1

        coordinates = (
            np.array(
                (
                    np.array([_x, 0, 0]),
                    np.array([0, _x, 0]),
                    np.array([0, 0, _x]),
                    np.array([-_x, 0, 0]),
                    np.array([0, -_x, 0]),
                    np.array([0, 0, -_x]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V2P3Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = self.distance * 2 * np.sqrt(3) / 4
        _y = self.distance * 2
        coordinates = (
            np.array(
                (
                    np.array([0, 0, 1]),
                    np.array([0, 0, -1]),
                    np.array([0, _x, 0]),
                    np.array([_y / 2, -_x, 0]),
                    np.array([-_y / 2, -_x, 0]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V2P4Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = (
            np.array(
                (
                    np.array([0, 0, -1]),
                    np.array([0, 0, 1]),
                    np.array([2, 0, 0]),
                    np.array([-2, 0, 0]),
                    np.array([0, 2, 0]),
                    np.array([0, -2, 0]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V4P62Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = (
            np.array(
                (
                    np.array([1, 0, 1]),
                    np.array([-1, 0, 1]),
                    np.array([1, 0, -1]),
                    np.array([-1, 0, -1]),
                    np.array([0, -1, 1]),
                    np.array([0, 1, 1]),
                    np.array([0, -1, -1]),
                    np.array([0, 1, -1]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V6P9Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = (
            np.array(
                (
                    np.array([-1, -1 / np.sqrt(3), -1]),
                    np.array([-1, -1 / np.sqrt(3), 1]),
                    np.array([1, -1 / np.sqrt(3), -1]),
                    np.array([1, -1 / np.sqrt(3), 1]),
                    np.array([0, 2 / np.sqrt(3), -1]),
                    np.array([0, 2 / np.sqrt(3), 1]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V8P16Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = 2
        coordinates = (
            np.array(
                (
                    np.array([-0.5 * _x, 0.5 * _x, -0.35 * _x]),
                    np.array([-0.5 * _x, -0.5 * _x, -0.35 * _x]),
                    np.array([0.5 * _x, -0.5 * _x, -0.35 * _x]),
                    np.array([0.5 * _x, 0.5 * _x, -0.35 * _x]),
                    np.array([-_x * np.sqrt(2) / 2, 0, _x * 0.35]),
                    np.array([0, -_x * np.sqrt(2) / 2, _x * 0.35]),
                    np.array([_x * np.sqrt(2) / 2, 0, _x * 0.35]),
                    np.array([0, _x * np.sqrt(2) / 2, _x * 0.35]),
                )
            )
            * self.distance
        )

        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V10P20Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _x = 1.5
        _z = _x / 2
        coordinates = (
            np.array(
                (
                    np.array([-_x, _x, -_z]),
                    np.array([-_x, -_x, -_z]),
                    np.array([_x, _x, -_z]),
                    np.array([_x, -_x, -_z]),
                    np.array([-_x, _x, _z]),
                    np.array([-_x, -_x, _z]),
                    np.array([_x, _x, _z]),
                    np.array([_x, -_x, _z]),
                    np.array([0, 0, _x]),
                    np.array([0, 0, -_x]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V12P24Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        coordinates = (
            np.array(
                (
                    np.array([1, 0, 0]),
                    np.array([-1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, -1, 0]),
                    np.array([0.5, 0.5, 0.707]),
                    np.array([0.5, -0.5, 0.707]),
                    np.array([-0.5, 0.5, 0.707]),
                    np.array([-0.5, -0.5, 0.707]),
                    np.array([0.5, 0.5, -0.707]),
                    np.array([0.5, -0.5, -0.707]),
                    np.array([-0.5, 0.5, -0.707]),
                    np.array([-0.5, -0.5, -0.707]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V12P30Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        # Vertices of a regular origin-centred icosahedron
        # Source: http://eusebeia.dyndns.org/4d/icosahedron
        coordinates = (
            np.array(
                (
                    np.array([0, 1, golden]),
                    np.array([0, -1, golden]),
                    np.array([0, 1, -golden]),
                    np.array([0, -1, -golden]),
                    np.array([1, golden, 0]),
                    np.array([-1, golden, 0]),
                    np.array([1, -golden, 0]),
                    np.array([-1, -golden, 0]),
                    np.array([golden, 0, 1]),
                    np.array([-golden, 0, 1]),
                    np.array([golden, 0, -1]),
                    np.array([-golden, 0, -1]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V20P30Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        # Source: http://tinyurl.com/h2dl949
        _phi = (1 + np.sqrt(5)) / 2
        _x = 1.5
        coordinates = (
            np.array(
                (
                    np.array([_x * _phi, 0.0, _x / _phi]),
                    np.array([_x * -_phi, 0.0, _x / _phi]),
                    np.array([_x * -_phi, 0.0, _x / -_phi]),
                    np.array([_x * _phi, 0.0, _x / -_phi]),
                    np.array([_x / _phi, _x * _phi, 0.0]),
                    np.array([_x / _phi, _x * -_phi, 0.0]),
                    np.array([_x / -_phi, _x * -_phi, 0.0]),
                    np.array([_x / -_phi, _x * _phi, 0.0]),
                    np.array([0.0, _x / _phi, _x * _phi]),
                    np.array([0.0, _x / _phi, _x * -_phi]),
                    np.array([0.0, _x / -_phi, _x * -_phi]),
                    np.array([0.0, _x / -_phi, _x * _phi]),
                    np.array([_x, _x, _x]),
                    np.array([_x, -_x, _x]),
                    np.array([-_x, -_x, _x]),
                    np.array([-_x, _x, _x]),
                    np.array([-_x, _x, -_x]),
                    np.array([_x, _x, -_x]),
                    np.array([_x, -_x, -_x]),
                    np.array([-_x, -_x, -_x]),
                )
            )
            * self.distance
        )
        self.building_block = beads_to_model(self.present_beads, coordinates)


@dataclass
class V24P48Shape(ShapePrecursor):
    """Define a shape precursor."""

    def __post_init__(self) -> None:
        _coord1 = 0.621
        _coord2 = -0.621
        _coord3 = _coord1 - (_coord1 - _coord2) / 2
        _coord4 = -1.5
        _coord5 = 1.5
        _coord6 = _coord5 - (_coord5 - _coord1) / 2
        coordinates = (
            np.array(
                (
                    np.array([_coord1, _coord1, _coord4]),
                    np.array([_coord1, _coord1, _coord5]),
                    np.array([_coord1, _coord4, _coord2]),
                    np.array([_coord1, _coord5, _coord2]),
                    np.array([_coord1, _coord2, _coord4]),
                    np.array([_coord1, _coord2, _coord5]),
                    np.array([_coord1, _coord4, _coord1]),
                    np.array([_coord1, _coord5, _coord1]),
                    np.array([_coord2, _coord1, _coord4]),
                    np.array([_coord2, _coord1, _coord5]),
                    np.array([_coord2, _coord2, _coord4]),
                    np.array([_coord2, _coord2, _coord5]),
                    np.array([_coord2, _coord4, _coord1]),
                    np.array([_coord2, _coord5, _coord1]),
                    np.array([_coord2, _coord4, _coord2]),
                    np.array([_coord2, _coord5, _coord2]),
                    np.array([_coord4, _coord1, _coord1]),
                    np.array([_coord4, _coord2, _coord1]),
                    np.array([_coord4, _coord2, _coord2]),
                    np.array([_coord4, _coord1, _coord2]),
                    np.array([_coord5, _coord1, _coord1]),
                    np.array([_coord5, _coord2, _coord1]),
                    np.array([_coord5, _coord2, _coord2]),
                    np.array([_coord5, _coord1, _coord2]),
                )
            )
            * self.distance
        )

        self.building_block = beads_to_model(self.present_beads, coordinates)
