#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of precursors.

Author: Andrew Tarzia

"""


import numpy as np
import stk

from .beads import CgBead


class Precursor:
    def __init__(self) -> None:
        self._bead_set: dict[str, CgBead]
        self._building_block: stk.BuildingBlock
        self._name: str
        raise NotImplementedError()

    def get_bead_set(self) -> dict[str, CgBead]:
        return self._bead_set

    def get_building_block(self) -> stk.BuildingBlock:
        return self._building_block

    def get_name(self) -> str:
        return self._name


class FourC0Arm(Precursor):
    def __init__(self, bead: CgBead) -> None:
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
    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
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
    def __init__(self, bead: CgBead) -> None:
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
    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
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
    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
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
    def __init__(self, bead: CgBead) -> None:
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
    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
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
    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
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
    def __init__(
        self,
        bead: CgBead,
        abead1: CgBead,
        abead2: CgBead,
        abead3: CgBead,
    ) -> None:
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


class UnsymmLigand(Precursor):
    def __init__(
        self,
        centre_bead: CgBead,
        lhs_bead: CgBead,
        rhs_bead: CgBead,
        binder_bead: CgBead,
    ) -> None:
        self._centre_bead = centre_bead
        self._lhs_bead = lhs_bead
        self._rhs_bead = rhs_bead
        self._binder_bead = binder_bead
        self._name = (
            f"UL{centre_bead.bead_type}{lhs_bead.bead_type}"
            f"{rhs_bead.bead_type}{binder_bead.bead_type}"
        )
        self._bead_set = {
            centre_bead.bead_type: centre_bead,
            lhs_bead.bead_type: lhs_bead,
            rhs_bead.bead_type: rhs_bead,
            binder_bead.bead_type: binder_bead,
        }

        new_fgs = (
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{binder_bead.element_string}X1]"
                    f"[{rhs_bead.element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{binder_bead.element_string}X1]"
                    f"[{lhs_bead.element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{binder_bead.element_string}]"
                f"[{lhs_bead.element_string}]"
                f"[{centre_bead.element_string}]"
                f"[{rhs_bead.element_string}]"
                f"[{binder_bead.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-10, 0, 0],
                    [-5, 3, 0],
                    [0, 5, 0],
                    [5, 3, 0],
                    [10, 0, 0],
                ]
            ),
        )


class UnsymmBiteLigand(Precursor):
    def __init__(
        self,
        centre_bead: CgBead,
        lhs_bead: CgBead,
        rhs_bead: CgBead,
        binder_bead: CgBead,
    ) -> None:
        self._centre_bead = centre_bead
        self._lhs_bead = lhs_bead
        self._rhs_bead = rhs_bead
        self._binder_bead = binder_bead
        self._name = (
            f"BL{centre_bead.bead_type}{lhs_bead.bead_type}"
            f"{rhs_bead.bead_type}{binder_bead.bead_type}"
        )
        self._bead_set = {
            centre_bead.bead_type: centre_bead,
            lhs_bead.bead_type: lhs_bead,
            rhs_bead.bead_type: rhs_bead,
            binder_bead.bead_type: binder_bead,
        }

        new_fgs = (
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{binder_bead.element_string}X1]"
                    f"[{rhs_bead.element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{binder_bead.element_string}X1]"
                    f"[{lhs_bead.element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{binder_bead.element_string}]"
                f"[{lhs_bead.element_string}]"
                f"[{centre_bead.element_string}]"
                f"[{rhs_bead.element_string}]"
                f"[{binder_bead.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-10, 0, 0],
                    [-5, 3, 0],
                    [0, 5, 0],
                    [5, 3, 0],
                    [10, 0, 0],
                ]
            ),
        )