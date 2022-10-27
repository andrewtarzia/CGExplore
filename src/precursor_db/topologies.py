#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Classes of topologies of precursors.

Author: Andrew Tarzia

"""

import stk
import random

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


class ThreeCPrecursorTemplate:
    def __init__(self, arm_length):
        self._arm_length = arm_length
        if arm_length == 1:
            self._internal_topology = Core1Arm
        elif arm_length == 2:
            self._internal_topology = Core2Arm

    def get_building_block(self, bead_2c_lib, bead_3c_lib):
        factories = (stk.BromoFactory(placers=(0, 1)),)

        three_c = random.choice(bead_3c_lib)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{three_c.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=[
                [-2, 0, 0],
                [0, 0, 0],
                [-1.2, 1, 0],
                [-1.2, -1, 0],
            ],
        )

        if self._arm_length == 1:
            two_cs = random.choice(bead_2c_lib)
            bb_tuple = (
                three_c_bb,
                stk.BuildingBlock(
                    smiles=f"[{two_cs.element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0]],
                ),
            )
            new_fgs = stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{three_c.element_string}]"
                    f"[{two_cs.element_string}]"
                ),
                bonders=(1,),
                deleters=(),
            )

        elif self._arm_length == 2:
            two_cs = random.choices(bead_2c_lib, k=2)
            bb_tuple = (
                three_c_bb,
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs[0].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
                stk.BuildingBlock(
                    smiles=f"[{two_cs[1].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0]],
                ),
            )
            new_fgs = stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{two_cs[0].element_string}]"
                    f"[{two_cs[1].element_string}]"
                ),
                bonders=(1,),
                deleters=(),
            )

        const_mol = stk.ConstructedMolecule(
            topology_graph=self._internal_topology(bb_tuple)
        )

        building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )
        return building_block


class TwoCPrecursorTemplate:
    def __init__(self, arm_length):
        self._arm_length = arm_length
        self._internal_topology = stk.polymer.Linear

    def get_building_block(self, bead_1c_lib, bead_2c_lib):
        factories = (stk.BromoFactory(placers=(0, 1)),)

        one_c = random.choice(bead_1c_lib)
        one_c_bb = stk.BuildingBlock(
            smiles=f"[{one_c.element_string}][Br]",
            functional_groups=factories,
            position_matrix=[[-3, 0, 0], [0, 0, 0]],
        )
        if self._arm_length == 1:
            repeating_unit = "ABA"
            two_cs = random.choice(bead_2c_lib)
            bb_tuple = (
                one_c_bb,
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs.element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
            )
            new_fgs = stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{one_c.element_string}]"
                    f"[{two_cs.element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            )

        elif self._arm_length == 2:
            repeating_unit = "ABCBA"
            two_cs = random.choices(bead_2c_lib, k=2)
            bb_tuple = (
                one_c_bb,
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs[0].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs[1].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
            )
            new_fgs = stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{one_c.element_string}]"
                    f"[{two_cs[0].element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            )

        elif self._arm_length == 3:
            repeating_unit = "ABCDCBA"
            two_cs = random.choices(bead_2c_lib, k=3)
            bb_tuple = (
                one_c_bb,
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs[0].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs[1].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
                stk.BuildingBlock(
                    smiles=f"[Br][{two_cs[2].element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=[[-3, 0, 0], [0, 0, 0], [3, 0, 0]],
                ),
            )
            new_fgs = stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{one_c.element_string}]"
                    f"[{two_cs[0].element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            )

        const_mol = stk.ConstructedMolecule(
            topology_graph=self._internal_topology(
                building_blocks=bb_tuple,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
            )
        )

        building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )
        return building_block
