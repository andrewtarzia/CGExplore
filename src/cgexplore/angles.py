# Distributed under the terms of the MIT License.

"""Module for handling angles."""

import itertools
import logging
from collections import abc
from dataclasses import dataclass

import stk
from openmm import openmm
from rdkit.Chem import AllChem as rdkit

from .errors import ForcefieldUnitError
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
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.class1}{self.class2}{self.class3}, "
            f"{self.eclass1}{self.eclass2}{self.eclass3}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class TargetAngleRange:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
        for angle, k in itertools.product(self.angles, self.angle_ks):
            yield TargetAngle(
                class1=self.class1,
                class2=self.class2,
                class3=self.class3,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                eclass3=self.eclass3,
                angle=angle,
                angle_k=k,
            )


@dataclass
class TargetPyramidAngle(TargetAngle):
    opposite_angle: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.class1}{self.class2}{self.class3}, "
            f"{self.eclass1}{self.eclass2}{self.eclass3}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.opposite_angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class TargetCosineAngle:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    n: float
    b: float
    angle_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.class1}{self.class2}{self.class3}, "
            f"{self.eclass1}{self.eclass2}{self.eclass3}, "
            f"{self.n}, {self.b}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class TargetCosineAngleRange:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    ns: tuple[float]
    bs: tuple[float]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
        for n, b, k in itertools.product(self.ns, self.bs, self.angle_ks):
            yield TargetCosineAngle(
                class1=self.class1,
                class2=self.class2,
                class3=self.class3,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                eclass3=self.eclass3,
                n=n,
                b=b,
                angle_k=k,
            )


@dataclass
class PyramidAngleRange:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
        for angle, k in itertools.product(self.angles, self.angle_ks):
            try:
                opposite_angle = openmm.unit.Quantity(
                    value=convert_pyramid_angle(
                        angle.value_in_unit(angle.unit)
                    ),
                    unit=angle.unit,
                )
            except AttributeError:
                msg = f"{self} in angles does not have units for parameters"
                raise ForcefieldUnitError(msg)

            yield TargetPyramidAngle(
                class1=self.class1,
                class2=self.class2,
                class3=self.class3,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                eclass3=self.eclass3,
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
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    funct: int
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity

    def human_readable(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.class1}{self.class2}{self.class3}, "
            f"{self.eclass1}{self.eclass2}{self.eclass3}, "
            f"{self.funct}, "
            f"{self.angle.in_units_of(openmm.unit.degrees)}, "
            f"{self.angle_k.in_units_of(angle_k_unit())}, "
            ")"
        )


@dataclass
class MartiniAngleRange:
    class1: str
    class2: str
    class3: str
    eclass1: str
    eclass2: str
    eclass3: str
    funct: int
    angles: tuple[openmm.unit.Quantity]
    angle_ks: tuple[openmm.unit.Quantity]

    def yield_angles(self):
        for angle, k in itertools.product(self.angles, self.angle_ks):
            yield TargetMartiniAngle(
                class1=self.class1,
                class2=self.class2,
                class3=self.class3,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                eclass3=self.eclass3,
                funct=self.funct,
                angle=angle,
                angle_k=k,
            )
