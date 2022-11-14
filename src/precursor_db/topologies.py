#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of precursors.

Author: Andrew Tarzia

"""

import stk

from stk.molecular.topology_graphs.cage.vertices import _CageVertex


class TerminalVertex(_CageVertex):
    def place_building_block(self, building_block, edges):
        building_block = building_block.with_centroid(
            position=self._position,
            atom_ids=building_block.get_core_atom_ids(),
        )
        # return building_block.get_position_matrix()
        (fg,) = building_block.get_functional_groups()
        fg_centroid = building_block.get_centroid(
            atom_ids=fg.get_placer_ids(),
        )
        core_centroid = building_block.get_centroid(
            atom_ids=building_block.get_core_atom_ids(),
        )
        edge_centroid = sum(
            edge.get_position() for edge in edges
        ) / len(edges)
        return building_block.with_rotation_between_vectors(
            start=(fg_centroid - core_centroid),
            # _cap_direction is defined by a subclass.
            target=edge_centroid - self._position,
            origin=self._position,
        ).get_position_matrix()

    def map_functional_groups_to_edges(self, building_block, edges):

        return {
            fg_id: edge.get_id() for fg_id, edge in enumerate(edges)
        }


class Core1Arm(stk.cage.Cage):
    # Special vertex definitions to be maintained.
    # Vertex ID, position.
    # Note the classes are Cage specific!
    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, (0, 0, 0)),
        TerminalVertex(
            id=1,
            position=(2, 0, 0),
            use_neighbor_placement=False,
        ),
        TerminalVertex(
            id=2,
            position=(-1.2, 1.0, 0),
            use_neighbor_placement=False,
        ),
        TerminalVertex(
            id=3,
            position=(-1.2, -1.0, 0),
            use_neighbor_placement=False,
        ),
    )

    # But Edges are not!
    _edge_prototypes = (
        # Edge ID, connected vertices by ID above.
        stk.Edge(
            id=0,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[1],
        ),
        stk.Edge(
            id=1,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[2],
        ),
        stk.Edge(
            id=2,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[3],
        ),
    )


class Core2Arm(stk.cage.Cage):
    # Special vertex definitions to be maintained.
    # Vertex ID, position.
    # Note the classes are Cage specific!
    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, (0, 0, 0)),
        stk.cage.LinearVertex(
            id=1,
            position=(2, 0, 0),
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=2,
            position=(-1.2, 1.0, 0),
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=3,
            position=(-1.2, -1.0, 0),
            use_neighbor_placement=False,
        ),
        TerminalVertex(
            id=4,
            position=(4, 0, 0),
            use_neighbor_placement=False,
        ),
        TerminalVertex(
            id=5,
            position=(-2.4, 2.0, 0),
            use_neighbor_placement=False,
        ),
        TerminalVertex(
            id=6,
            position=(-2.4, -2.0, 0),
            use_neighbor_placement=False,
        ),
    )

    # But Edges are not!
    _edge_prototypes = (
        # Edge ID, connected vertices by ID above.
        stk.Edge(
            id=0,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[1],
        ),
        stk.Edge(
            id=1,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[2],
        ),
        stk.Edge(
            id=2,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[3],
        ),
        stk.Edge(
            id=3,
            vertex1=_vertex_prototypes[4],
            vertex2=_vertex_prototypes[1],
        ),
        stk.Edge(
            id=4,
            vertex1=_vertex_prototypes[5],
            vertex2=_vertex_prototypes[2],
        ),
        stk.Edge(
            id=5,
            vertex1=_vertex_prototypes[6],
            vertex2=_vertex_prototypes[3],
        ),
    )


class FourC0Arm:
    def __init__(self, bead):
        self._bead = bead
        self._name = f"4C0{bead.element_string}"
        four_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])([Br])[Br]",
            position_matrix=[
                [-2, 0, 0],
                [0, 0, 0],
                [0, -2, 0],
                [2, 0, 0],
                [0, 2, 0],
            ],
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(f"[{bead.element_string}]" f"[Br]"),
            bonders=(0,),
            deleters=(1,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=four_c_bb,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class ThreeC0Arm:
    def __init__(self, bead):
        self._bead = bead
        self._name = f"3C0{bead.element_string}"
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            position_matrix=[
                [-1.8, 0, 0],
                [0, 0, 0],
                [-1.2, 1, 0],
                [-1.2, -1, 0],
            ],
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(f"[{bead.element_string}]" f"[Br]"),
            bonders=(0,),
            deleters=(1,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=three_c_bb,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class ThreeC1Arm:
    def __init__(self, bead, abead1):
        self._bead = bead
        self._abead1 = abead1
        self._name = f"3C1{bead.element_string}{abead1.element_string}"
        factories = (stk.BromoFactory(placers=(0, 1)),)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=[
                [-2, 0, 0],
                [0, 0, 0],
                [-1.2, 1, 0],
                [-1.2, -1, 0],
            ],
        )
        bb_tuple = (
            three_c_bb,
            stk.BuildingBlock(
                smiles=f"[{abead1.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0]],
            ),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=Core1Arm(bb_tuple)
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{bead.element_string}]" f"[{abead1.element_string}]"
            ),
            bonders=(1,),
            deleters=(),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class ThreeC2Arm:
    def __init__(self, bead, abead1, abead2):
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = (
            f"3C2{bead.element_string}{abead1.element_string}"
            f"{abead2.element_string}"
        )
        factories = (stk.BromoFactory(placers=(0, 1)),)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=[
                [-2, 0, 0],
                [0, 0, 0],
                [-1.2, 1, 0],
                [-1.2, -1, 0],
            ],
        )
        bb_tuple = (
            three_c_bb,
            stk.BuildingBlock(
                smiles=f"[Br][{abead1.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
            ),
            stk.BuildingBlock(
                smiles=f"[{abead2.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0]],
            ),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=Core2Arm(bb_tuple)
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{abead1.element_string}]"
                f"[{abead2.element_string}]"
            ),
            bonders=(1,),
            deleters=(),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class TwoC0Arm:
    def __init__(self, bead):
        self._bead = bead
        self._name = f"2C0{bead.element_string}"
        core_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}][Br]",
            position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(f"[Br][{bead.element_string}]"),
            bonders=(1,),
            deleters=(0,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=core_c_bb,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class TwoC1Arm:
    def __init__(self, bead, abead1):
        self._bead = bead
        self._abead1 = abead1
        self._name = f"2C1{bead.element_string}{abead1.element_string}"
        factories = (stk.BromoFactory(placers=(0, 1)),)
        core_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}][Br]",
            functional_groups=factories,
            position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
        )
        bb_tuple = (
            stk.BuildingBlock(
                smiles=f"[{abead1.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0]],
            ),
            core_c_bb,
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{abead1.element_string}][{bead.element_string}]"
            ),
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=bb_tuple,
                repeating_unit="ABA",
                num_repeating_units=1,
            )
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class TwoC2Arm:
    def __init__(self, bead, abead1, abead2):
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = (
            f"2C2{bead.element_string}{abead1.element_string}"
            f"{abead2.element_string}"
        )
        factories = (stk.BromoFactory(placers=(0, 1)),)
        core_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}][Br]",
            functional_groups=factories,
            position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
        )
        bb_tuple = (
            stk.BuildingBlock(
                smiles=f"[{abead2.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0]],
            ),
            stk.BuildingBlock(
                smiles=f"[Br][{abead1.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
            ),
            core_c_bb,
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{abead2.element_string}X1]"
                f"[{abead1.element_string}]"
            ),
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=bb_tuple,
                repeating_unit="ABCBA",
                num_repeating_units=1,
            )
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name


class TwoC3Arm:
    def __init__(self, bead, abead1, abead2, abead3):
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._abead3 = abead3
        self._name = (
            f"2C3{bead.element_string}{abead1.element_string}"
            f"{abead2.element_string}{abead3.element_string}"
        )
        factories = (stk.BromoFactory(placers=(0, 1)),)
        core_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}][Br]",
            functional_groups=factories,
            position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
        )
        bb_tuple = (
            stk.BuildingBlock(
                smiles=f"[{abead3.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0]],
            ),
            stk.BuildingBlock(
                smiles=f"[Br][{abead2.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
            ),
            stk.BuildingBlock(
                smiles=f"[Br][{abead1.element_string}][Br]",
                functional_groups=factories,
                position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
            ),
            core_c_bb,
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(
                f"[{abead3.element_string}X1]"
                f"[{abead2.element_string}]"
            ),
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=bb_tuple,
                repeating_unit="ABCDCBA",
                num_repeating_units=1,
            )
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )

    def get_building_block(self):
        return self._building_block

    def get_name(self):
        return self._name
