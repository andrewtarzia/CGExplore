# Distributed under the terms of the MIT License.

"""Module for handling angles."""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

import stk
from openmm import openmm

from cgexplore._internal.utilities.errors import ForceFieldUnitError
from cgexplore._internal.utilities.utilities import convert_pyramid_angle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_angle_k_unit = openmm.unit.kilojoules_per_mole / openmm.unit.radian**2


@dataclass(frozen=True, slots=True)
class Angle:
    """Class containing angle defintion."""

    atom_names: abc.Sequence[str]
    atom_ids: abc.Sequence[int]
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity
    atoms: tuple[stk.Atom, ...] | None
    force: str | None
    funct: int = 0


@dataclass(frozen=True, slots=True)
class CosineAngle:
    """Class containing cosine-angle defintion."""

    atom_names: abc.Sequence[str]
    atom_ids: abc.Sequence[int]
    n: int
    b: int
    angle_k: openmm.unit.Quantity
    atoms: tuple[stk.Atom, ...] | None
    force: str | None


@dataclass(frozen=True, slots=True)
class TargetAngle:
    """Defines a target angle to search for in a molecule."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity

    def vector_key(self) -> str:
        """Return key for vector defining this target term."""
        return f"{self.type1}{self.type2}{self.type3}"

    def vector(self) -> tuple[float, float]:
        """Return vector defining this target term."""
        return (
            self.angle.value_in_unit(openmm.unit.degrees),
            self.angle_k.value_in_unit(_angle_k_unit),
        )

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(_angle_k_unit)}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class TargetAngleRange:
    """Defines a target angle and ranges in parameters to search for."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    angles: abc.Sequence[openmm.unit.Quantity]
    angle_ks: abc.Sequence[openmm.unit.Quantity]

    def yield_angles(self) -> abc.Iterable[TargetAngle]:
        """Find angles matching target."""
        for angle, k in it.product(self.angles, self.angle_ks):
            yield TargetAngle(
                type1=self.type1,
                type2=self.type2,
                type3=self.type3,
                element1=self.element1,
                element2=self.element2,
                element3=self.element3,
                angle=angle,
                angle_k=k,
            )


@dataclass(frozen=True, slots=True)
class TargetCosineAngle:
    """Defines a target angle to search for in a molecule."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    n: int
    b: int
    angle_k: openmm.unit.Quantity

    def vector_key(self) -> str:
        """Return key for vector defining this target term."""
        return f"{self.type1}{self.type2}{self.type3}"

    def vector(self) -> tuple[float, float, float]:
        """Return vector defining this target term."""
        return (
            self.n,
            self.b,
            self.angle_k.value_in_unit(openmm.unit.kilojoules_per_mole),
        )

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.n}, {self.b}, "
            f"{self.angle_k.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class TargetCosineAngleRange:
    """Defines a target angle and ranges in parameters to search for."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    ns: abc.Sequence[int]
    bs: abc.Sequence[int]
    angle_ks: abc.Sequence[openmm.unit.Quantity]

    def yield_angles(self) -> abc.Iterable[TargetCosineAngle]:
        """Find angles matching target."""
        for n, b, k in it.product(self.ns, self.bs, self.angle_ks):
            yield TargetCosineAngle(
                type1=self.type1,
                type2=self.type2,
                type3=self.type3,
                element1=self.element1,
                element2=self.element2,
                element3=self.element3,
                n=n,
                b=b,
                angle_k=k,
            )


@dataclass(frozen=True, slots=True)
class TargetPyramidAngle(TargetAngle):
    """Defines a target angle to search for in a molecule."""

    opposite_angle: openmm.unit.Quantity

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.opposite_angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(_angle_k_unit)}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class PyramidAngleRange:
    """Defines a target angle and ranges in parameters to search for."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    angles: abc.Sequence[openmm.unit.Quantity]
    angle_ks: abc.Sequence[openmm.unit.Quantity]

    def yield_angles(self) -> abc.Iterable[TargetPyramidAngle]:
        """Find angles matching target."""
        for angle, k in it.product(self.angles, self.angle_ks):
            try:
                opposite_angle = openmm.unit.Quantity(
                    value=convert_pyramid_angle(
                        angle.value_in_unit(angle.unit)
                    ),
                    unit=angle.unit,
                )
            except AttributeError:
                msg = f"{self} in angles does not have units for parameters"
                raise ForceFieldUnitError(msg)  # noqa: B904

            yield TargetPyramidAngle(
                type1=self.type1,
                type2=self.type2,
                type3=self.type3,
                element1=self.element1,
                element2=self.element2,
                element3=self.element3,
                angle=angle,
                opposite_angle=opposite_angle,
                angle_k=k,
            )


@dataclass(frozen=True, slots=True)
class FoundAngle:
    """Define a found forcefield term."""

    atoms: abc.Sequence[stk.Atom]
    atom_ids: abc.Sequence[int]


@dataclass(frozen=True, slots=True)
class TargetMartiniAngle:
    """Defines a target angle to search for in a molecule."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    funct: int
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        """Return human-readable definition of this target term."""
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.funct}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(_angle_k_unit)}, "
            ")"
        )


@dataclass(frozen=True, slots=True)
class MartiniAngleRange:
    """Defines a target angle and ranges in parameters to search for."""

    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    funct: int
    angles: abc.Sequence[openmm.unit.Quantity]
    angle_ks: abc.Sequence[openmm.unit.Quantity]

    def yield_angles(self) -> abc.Iterable[TargetMartiniAngle]:
        """Find angles matching target."""
        for angle, k in it.product(self.angles, self.angle_ks):
            yield TargetMartiniAngle(
                type1=self.type1,
                type2=self.type2,
                type3=self.type3,
                element1=self.element1,
                element2=self.element2,
                element3=self.element3,
                funct=self.funct,
                angle=angle,
                angle_k=k,
            )
