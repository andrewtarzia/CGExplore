# Distributed under the terms of the MIT License.

"""Module for handling angles."""

import itertools as it
import logging
from collections import abc
from dataclasses import dataclass

import stk
from openmm import openmm
from rdkit.Chem import AllChem as rdkit

from .errors import ForceFieldUnitError
from .utilities import convert_pyramid_angle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def angle_k_unit():
    return openmm.unit.kilojoules_per_mole / openmm.unit.radian**2


@dataclass
class Angle:
    atom_names: tuple[str, ...]
    atom_ids: tuple[int, ...]
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity
    atoms: tuple[stk.Atom, ...] | None
    force: str | None
    funct: int = 0


@dataclass
class CosineAngle:
    atom_names: tuple[str, ...]
    atom_ids: tuple[int, ...]
    n: int
    b: int
    angle_k: openmm.unit.Quantity
    atoms: tuple[stk.Atom, ...] | None
    force: str | None


@dataclass
class TargetAngle:
    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity

    def vector_key(self) -> str:
        return f"{self.type1}{self.type2}{self.type3}"

    def vector(self) -> tuple[float, float]:
        return (
            self.angle.value_in_unit(openmm.unit.degrees),
            self.angle_k.value_in_unit(angle_k_unit()),
        )

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class TargetAngleRange:
    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
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


@dataclass
class TargetPyramidAngle(TargetAngle):
    opposite_angle: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.opposite_angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class TargetCosineAngle:
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
        return f"{self.type1}{self.type2}{self.type3}"

    def vector(self) -> tuple[float, float, float]:
        return (
            self.n,
            self.b,
            self.angle_k.value_in_unit(openmm.unit.kilojoules_per_mole),
        )

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.n}, {self.b}, "
            f"{self.angle_k.in_units_of(openmm.unit.kilojoules_per_mole)}, "
            ")"
        )


@dataclass
class TargetCosineAngleRange:
    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    ns: tuple[int]
    bs: tuple[int]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
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


@dataclass
class PyramidAngleRange:
    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
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
                raise ForceFieldUnitError(msg)

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


@dataclass
class FoundAngle:
    atoms: tuple[stk.Atom, ...]
    atom_ids: tuple[int, ...]


def find_angles(molecule: stk.Molecule) -> abc.Iterator[FoundAngle]:
    paths = rdkit.FindAllPathsOfLengthN(
        mol=molecule.to_rdkit_mol(),
        length=3,
        useBonds=False,
        useHs=True,
    )
    for atom_ids in paths:
        atoms = tuple(molecule.get_atoms(atom_ids=list(atom_ids)))
        yield FoundAngle(
            atoms=atoms,
            atom_ids=tuple(i.get_id() for i in atoms),
        )


@dataclass
class TargetMartiniAngle:
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
        return (
            f"{self.__class__.__name__}("
            f"{self.type1}{self.type2}{self.type3}, "
            f"{self.element1}{self.element2}{self.element3}, "
            f"{self.funct}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class MartiniAngleRange:
    type1: str
    type2: str
    type3: str
    element1: str
    element2: str
    element3: str
    funct: int
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
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
