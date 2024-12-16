# Distributed under the terms of the MIT License.

"""Classes of topologies of precursors.

Author: Andrew Tarzia

"""

from dataclasses import dataclass

import numpy as np
import stk

from .beads import CgBead, string_to_atom_number


@dataclass
class LinearPrecursor:
    composition: tuple[int, ...]
    present_beads: tuple[CgBead, ...]
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]

    def __post_init__(self) -> None:
        atoms = [
            stk.Atom(
                i, string_to_atom_number(self.present_beads[i].element_string)
            )
            for i in range(len(self.present_beads))
        ]

        coordinates = [np.array([0, 0, 0])]
        bonds = []
        for i in range(self.composition[0]):
            coordinates.append(np.array([1 * (i + 1), 0, 0]))
            bonds.append(stk.Bond(atoms[i], atoms[i + 1], order=1))

        model = stk.BuildingBlock.init(
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            position_matrix=np.array(coordinates),
        )

        if self.composition[0] == 0:
            new_fgs = (
                stk.GenericFunctionalGroup(
                    atoms=(atoms[0],),
                    bonders=(atoms[0],),
                    deleters=(),
                    placers=(atoms[0],),
                ),
                stk.GenericFunctionalGroup(
                    atoms=(atoms[0],),
                    bonders=(atoms[0],),
                    deleters=(),
                    placers=(atoms[0],),
                ),
            )
        elif self.composition[0] == 1:
            new_fgs = (
                stk.GenericFunctionalGroup(
                    atoms=(atoms[0], atoms[1]),
                    bonders=(atoms[0],),
                    deleters=(),
                    placers=(atoms[0], atoms[1]),
                ),
                stk.GenericFunctionalGroup(
                    atoms=(atoms[0], atoms[1]),
                    bonders=(atoms[1],),
                    deleters=(),
                    placers=(atoms[0], atoms[1]),
                ),
            )
        else:
            new_fgs = (  # type: ignore[assignment]
                stk.SmartsFunctionalGroupFactory(
                    smarts=(
                        f"[{self.placer_beads[0].element_string}]"
                        f"[{self.binder_beads[0].element_string}]"
                    ),
                    bonders=(1,),
                    deleters=(),
                    placers=(0, 1),
                ),
            )
        self.building_block = stk.BuildingBlock.init_from_molecule(
            molecule=model,
            functional_groups=new_fgs,
        )

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


@dataclass
class TrianglePrecursor:
    present_beads: tuple[CgBead, ...]
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]

    def __post_init__(self) -> None:
        _x = 2 * np.sqrt(3) / 4
        _y = 2
        coordinates = np.array(
            (
                np.array([0, _x, 0]),
                np.array([_y / 2, -_x, 0]),
                np.array([-_y / 2, -_x, 0]),
            )
        )

        atoms = [
            stk.Atom(
                i, string_to_atom_number(self.present_beads[i].element_string)
            )
            for i in range(len(self.present_beads))
        ]
        bonds = [
            stk.Bond(atoms[0], atoms[1], order=1),
            stk.Bond(atoms[1], atoms[2], order=1),
            stk.Bond(atoms[2], atoms[0], order=1),
        ]

        model = stk.BuildingBlock.init(
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            position_matrix=coordinates,
        )

        new_fgs = (
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{self.placer_beads[0].element_string}]"
                    f"[{self.binder_beads[0].element_string}]"
                    f"[{self.placer_beads[1].element_string}]"
                ),
                bonders=(1,),
                deleters=(),
                placers=(0, 1, 2),
            ),
        )
        self.building_block = stk.BuildingBlock.init_from_molecule(
            molecule=model,
            functional_groups=new_fgs,
        )

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


@dataclass
class SquarePrecursor:
    present_beads: tuple[CgBead, ...]
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]

    def __post_init__(self) -> None:
        coordinates = np.array(
            (
                np.array([1, 1, 0]),
                np.array([1, -1, 0]),
                np.array([-1, -1, 0]),
                np.array([-1, 1, 0]),
            )
        )

        atoms = [
            stk.Atom(
                i, string_to_atom_number(self.present_beads[i].element_string)
            )
            for i in range(len(self.present_beads))
        ]
        bonds = [
            stk.Bond(atoms[0], atoms[1], order=1),
            stk.Bond(atoms[1], atoms[2], order=1),
            stk.Bond(atoms[2], atoms[3], order=1),
            stk.Bond(atoms[3], atoms[0], order=1),
        ]

        model = stk.BuildingBlock.init(
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            position_matrix=coordinates,
        )

        new_fgs = (
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{self.placer_beads[0].element_string}]"
                    f"[{self.binder_beads[0].element_string}]"
                    f"[{self.placer_beads[1].element_string}]"
                ),
                bonders=(1,),
                deleters=(),
                placers=(0, 1, 2),
            ),
        )
        self.building_block = stk.BuildingBlock.init_from_molecule(
            molecule=model,
            functional_groups=new_fgs,
        )

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


class Precursor:
    """Define model precursors."""

    def __init__(self) -> None:
        """Initialize a precursor."""
        self._bead_set: dict[str, CgBead]
        self._building_block: stk.BuildingBlock
        self._name: str
        raise NotImplementedError

    def get_bead_set(self) -> dict[str, CgBead]:
        """Get beads in precursor."""
        return self._bead_set

    def get_building_block(self) -> stk.BuildingBlock:
        """Get building block defined by precursor."""
        return self._building_block

    def get_name(self) -> str:
        """Get name of precursor."""
        return self._name


class FourC0Arm(Precursor):
    """A `FourC0Arm` Precursor."""

    def __init__(self, bead: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"4C0{bead.bead_type}"
        self._bead_set = {bead.bead_type: bead}
        four_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])([Br])[Br]",
            position_matrix=np.array(
                [
                    [-2, 0, -1],
                    [0, 0, 1],
                    [0, -2, -1],
                    [2, 0, 1],
                    [0, 2, 1],
                ]
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(f"[{bead.element_string}][Br]"),
            bonders=(0,),
            deleters=(1,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=four_c_bb,
            functional_groups=(new_fgs,),
        )


class FourC1Arm(Precursor):
    """A `FourC1Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = f"4C1{bead.bead_type}{abead1.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
        }

        factories = (stk.BromoFactory(placers=(0, 1)),)
        four_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])([Br])[Br]",
            functional_groups=factories,
            position_matrix=np.array(
                [
                    [-2, 0, -1],
                    [0, 0, 1],
                    [0, -2, -1],
                    [2, 0, -1],
                    [0, 2, -1],
                ]
            ),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.small.NCore(
                core_building_block=four_c_bb,
                arm_building_blocks=stk.BuildingBlock(
                    smiles=f"[{abead1.element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=np.array([[-3, 0, 0], [0, 0, 0]]),
                ),
                repeating_unit="AAAA",
            ),
        )
        new_fgs = (
            stk.SmartsFunctionalGroupFactory(
                smarts=(f"[{bead.element_string}][{abead1.element_string}]"),
                bonders=(1,),
                deleters=(),
            ),
        )

        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=new_fgs,
        )


class ThreeC0Arm(Precursor):
    """A `ThreeC0Arm` Precursor."""

    def __init__(self, bead: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"3C0{bead.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
        }
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{bead.element_string}][Br]",
            bonders=(0,),
            deleters=(1,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=three_c_bb,
            functional_groups=(new_fgs,),
        )


class ThreeC1Arm(Precursor):
    """A `ThreeC1Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = f"3C1{bead.bead_type}{abead1.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
        }
        factories = (stk.BromoFactory(placers=(0, 1)),)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0.5],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.small.NCore(
                core_building_block=three_c_bb,
                arm_building_blocks=stk.BuildingBlock(
                    smiles=f"[{abead1.element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=np.array([[-3, 0, 0], [0, 0, 0]]),
                ),
                repeating_unit="AAA",
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{bead.element_string}][{abead1.element_string}]",
            bonders=(1,),
            deleters=(),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )


class ThreeC2Arm(Precursor):
    """A `ThreeC2Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = f"3C2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
        }
        factories = (stk.BromoFactory(placers=(0, 1)),)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{abead2.element_string}]",
            bonders=(0,),
            deleters=(),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.small.NCore(
                core_building_block=three_c_bb,
                arm_building_blocks=stk.BuildingBlock(
                    smiles=f"[{abead1.element_string}][{abead2.element_string}]",
                    functional_groups=new_fgs,
                    position_matrix=np.array([[-3, 0, 0], [0, 0, 0]]),
                ),
                repeating_unit="AAA",
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{abead2.element_string}]",
            bonders=(1,),
            deleters=(),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )


class TwoC0Arm(Precursor):
    """A `TwoC0Arm` Precursor."""

    def __init__(self, bead: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"2C0{bead.bead_type}"
        self._bead_set = {bead.bead_type: bead}
        core_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}][Br]",
            position_matrix=np.array([[-3, 0, 0], [0, 0, 0], [3, 0, 0]]),
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[Br][{bead.element_string}]",
            bonders=(1,),
            deleters=(0,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=core_c_bb,
            functional_groups=(new_fgs,),
        )


class TwoC1Arm(Precursor):
    """A `TwoC1Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = f"2C1{bead.bead_type}{abead1.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{bead.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=f"[{abead1.element_string}][{bead.element_string}][{abead1.element_string}]",
            functional_groups=new_fgs,
            position_matrix=np.array([[-3, 0, 0], [0, 0, 0], [3, 0, 0]]),
        )


class TwoC2Arm(Precursor):
    """A `TwoC2Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = f"2C2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead2.element_string}X1][{abead1.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{abead1.element_string}]"
                f"[{abead2.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-8, 0, 0],
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                    [8, 0, 0],
                ]
            ),
        )


class TwoC3Arm(Precursor):
    """A `TwoC3Arm` Precursor."""

    def __init__(
        self,
        bead: CgBead,
        abead1: CgBead,
        abead2: CgBead,
        abead3: CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._abead3 = abead3
        self._name = (
            f"2C3{bead.bead_type}{abead1.bead_type}"
            f"{abead2.bead_type}{abead3.bead_type}"
        )
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
            abead3.bead_type: abead3,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead3.element_string}X1][{abead2.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead3.element_string}]"
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{abead1.element_string}]"
                f"[{abead2.element_string}][{abead3.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-12, 0, 0],
                    [-8, 0, 0],
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                    [8, 0, 0],
                    [12, 0, 0],
                ]
            ),
        )


class SixBead(Precursor):
    """A Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = f"6C2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead2.element_string}X1][{abead1.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{bead.element_string}]"
                f"[{abead1.element_string}][{abead2.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-6, 3, 0.2],
                    [-4, 2, 0],
                    [-2, 0.1, 0],
                    [2, 0, 0],
                    [4, 2, 0],
                    [6, 3, 0.2],
                ]
            ),
        )


class StericSixBead(Precursor):
    """A Precursor."""

    def __init__(
        self,
        bead: CgBead,
        abead1: CgBead,
        abead2: CgBead,
        sbead: CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._sbead = sbead
        self._name = (
            f"6S2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
            f"{sbead.bead_type}"
        )
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
            sbead.bead_type: sbead,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead2.element_string}X1][{abead1.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{bead.element_string}]"
                f"([{sbead.element_string}])[{bead.element_string}]"
                f"[{abead1.element_string}][{abead2.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-6, 3, 0.2],
                    [-4, 2, 0],
                    [-2, 0.1, 0],
                    [0, 0.1, 0],
                    [0, 1, 0],
                    [2, 0, 0],
                    [4, 2, 0],
                    [6, 3, 0.2],
                ]
            ),
        )
