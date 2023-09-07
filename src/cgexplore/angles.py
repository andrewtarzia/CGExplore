#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for handling angles.

Author: Andrew Tarzia

"""

import itertools
import logging
from rdkit.Chem import AllChem as rdkit
import stk
from dataclasses import dataclass
import typing
from openmm import openmm

from .utilities import convert_pyramid_angle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Angle:
    atoms: tuple[stk.Atom, stk.Atom, stk.Atom]
    atom_names: tuple[str, str, str]
    atom_ids: tuple[int, int, int]
    angle: openmm.unit.Quantity
    angle_k: openmm.unit.Quantity


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
            yield TargetPyramidAngle(
                class1=self.class1,
                class2=self.class2,
                class3=self.class3,
                eclass1=self.eclass1,
                eclass2=self.eclass2,
                eclass3=self.eclass3,
                angle=angle,
                opposite_angle=openmm.unit.Quantity(
                    value=convert_pyramid_angle(
                        angle.value_in_unit(angle.unit)
                    ),
                    unit=angle.unit,
                ),
                angle_k=k,
            )


@dataclass
class FoundAngle:
    atoms: tuple[stk.Atom, stk.Atom, stk.Atom]
    atom_ids: tuple[int, int, int]


def find_angles(molecule: stk.Molecule) -> typing.Iterator[FoundAngle]:
    paths = rdkit.FindAllPathsOfLengthN(
        mol=molecule.to_rdkit_mol(),
        length=3,
        useBonds=False,
        useHs=True,
    )
    for atom_ids in paths:
        atoms = list(molecule.get_atoms(atom_ids=[i for i in atom_ids]))
        yield FoundAngle(
            atoms=atoms,
            atom_ids=tuple(i.get_id() for i in atoms),
        )
