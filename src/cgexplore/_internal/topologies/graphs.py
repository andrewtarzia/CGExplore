"""New stk topologies of cages."""

from collections import abc

import numpy as np
import stk


class M4L82(stk.cage.Cage):
    """New topology definition."""

    _non_linears = (
        stk.cage.NonLinearVertex(0, np.array([0, 0, np.sqrt(6) / 2])),
        stk.cage.NonLinearVertex(
            1, np.array([-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6])
        ),
        stk.cage.NonLinearVertex(
            2, np.array([1, -np.sqrt(3) / 3, -np.sqrt(6) / 6])
        ),
        stk.cage.NonLinearVertex(
            3, np.array([0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6])
        ),
    )

    paired_wall_1_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[0], _non_linears[1])
        )
        / 2
    )
    wall_1_shift = np.array((0.2, 0.2, 0))

    paired_wall_2_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[2], _non_linears[3])
        )
        / 2
    )
    wall_2_shift = np.array((0.2, 0.2, 0))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            id=4,
            position=paired_wall_1_coord + wall_1_shift,
        ),
        stk.cage.LinearVertex.init_at_center(
            id=5,
            vertices=(_non_linears[0], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=6,
            vertices=(_non_linears[0], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=7,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=8,
            vertices=(_non_linears[1], _non_linears[3]),
        ),
        stk.cage.LinearVertex(
            id=9,
            position=paired_wall_2_coord + wall_2_shift,
        ),
        stk.cage.LinearVertex(
            id=10,
            position=paired_wall_1_coord - wall_1_shift,
        ),
        stk.cage.LinearVertex(
            id=11,
            position=paired_wall_2_coord - wall_2_shift,
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[10]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[7]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[5]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[11]),
    )


class CGM4L8(stk.cage.M4L8):
    """New topology definition."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([2, 0, 0])),
        stk.cage.NonLinearVertex(1, np.array([0, 2, 0])),
        stk.cage.NonLinearVertex(2, np.array([-2, 0, 0])),
        stk.cage.NonLinearVertex(3, np.array([0, -2, 0])),
        stk.cage.LinearVertex(
            4, np.array([1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            5, np.array([1, 1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            6, np.array([1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            7, np.array([1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            8, np.array([-1, -1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            9, np.array([-1, -1, -0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            10, np.array([-1, 1, 0.5]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            11, np.array([-1, 1, -0.5]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[8]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
    )


class CGM12L24(stk.cage.M12L24):
    """New topology definition."""

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(0, np.array([1.25, 0, 0])),
        stk.cage.NonLinearVertex(1, np.array([-1.25, 0, 0])),
        stk.cage.NonLinearVertex(2, np.array([0, 1.25, 0])),
        stk.cage.NonLinearVertex(3, np.array([0, -1.25, 0])),
        stk.cage.NonLinearVertex(4, np.array([0.625, 0.625, 0.88])),
        stk.cage.NonLinearVertex(5, np.array([0.625, -0.625, 0.88])),
        stk.cage.NonLinearVertex(6, np.array([-0.625, 0.625, 0.88])),
        stk.cage.NonLinearVertex(7, np.array([-0.625, -0.625, 0.88])),
        stk.cage.NonLinearVertex(8, np.array([0.625, 0.625, -0.88])),
        stk.cage.NonLinearVertex(9, np.array([0.625, -0.625, -0.88])),
        stk.cage.NonLinearVertex(10, np.array([-0.625, 0.625, -0.88])),
        stk.cage.NonLinearVertex(11, np.array([-0.625, -0.625, -0.88])),
        stk.cage.LinearVertex(
            12, np.array([0.9, 0.31, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            13, np.array([0.9, 0.31, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            14, np.array([0.9, -0.31, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            15, np.array([0.9, -0.31, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            16, np.array([-0.9, 0.31, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            17, np.array([-0.9, 0.31, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            18, np.array([-0.9, -0.31, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            19, np.array([-0.9, -0.31, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            20, np.array([0.31, 0.9, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            21, np.array([0.31, 0.9, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            22, np.array([-0.31, 0.9, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            23, np.array([-0.31, 0.9, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            24, np.array([0.31, -0.9, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            25, np.array([0.31, -0.9, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            26, np.array([-0.31, -0.9, 0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            27, np.array([-0.31, -0.9, -0.31]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            28, np.array([0.58, 0, 0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            29, np.array([-0.58, 0, 0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            30, np.array([0, 0.58, 0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            31, np.array([0, -0.58, 0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            32, np.array([0.58, 0, -0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            33, np.array([-0.58, 0, -0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            34, np.array([0, 0.58, -0.82]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            35, np.array([0, -0.58, -0.82]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[12]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[13]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[14]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[15]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[16]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[17]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[18]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[19]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[20]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[21]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[22]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[23]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[24]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[25]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[26]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[27]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[28]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[30]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[12]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[20]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[14]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[24]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[28]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[31]),
        stk.Edge(24, _vertex_prototypes[6], _vertex_prototypes[16]),
        stk.Edge(25, _vertex_prototypes[6], _vertex_prototypes[29]),
        stk.Edge(26, _vertex_prototypes[6], _vertex_prototypes[30]),
        stk.Edge(27, _vertex_prototypes[6], _vertex_prototypes[22]),
        stk.Edge(28, _vertex_prototypes[7], _vertex_prototypes[18]),
        stk.Edge(29, _vertex_prototypes[7], _vertex_prototypes[26]),
        stk.Edge(30, _vertex_prototypes[7], _vertex_prototypes[31]),
        stk.Edge(31, _vertex_prototypes[7], _vertex_prototypes[29]),
        stk.Edge(32, _vertex_prototypes[8], _vertex_prototypes[13]),
        stk.Edge(33, _vertex_prototypes[8], _vertex_prototypes[32]),
        stk.Edge(34, _vertex_prototypes[8], _vertex_prototypes[34]),
        stk.Edge(35, _vertex_prototypes[8], _vertex_prototypes[21]),
        stk.Edge(36, _vertex_prototypes[9], _vertex_prototypes[15]),
        stk.Edge(37, _vertex_prototypes[9], _vertex_prototypes[32]),
        stk.Edge(38, _vertex_prototypes[9], _vertex_prototypes[35]),
        stk.Edge(39, _vertex_prototypes[9], _vertex_prototypes[25]),
        stk.Edge(40, _vertex_prototypes[10], _vertex_prototypes[17]),
        stk.Edge(41, _vertex_prototypes[10], _vertex_prototypes[23]),
        stk.Edge(42, _vertex_prototypes[10], _vertex_prototypes[34]),
        stk.Edge(43, _vertex_prototypes[10], _vertex_prototypes[33]),
        stk.Edge(44, _vertex_prototypes[11], _vertex_prototypes[19]),
        stk.Edge(45, _vertex_prototypes[11], _vertex_prototypes[33]),
        stk.Edge(46, _vertex_prototypes[11], _vertex_prototypes[27]),
        stk.Edge(47, _vertex_prototypes[11], _vertex_prototypes[35]),
    )

    @staticmethod
    def _get_scale(
        building_block_vertices: dict[  # noqa: ARG004
            stk.BuildingBlock, abc.Sequence[stk.Vertex]
        ],
        scale_multiplier: float,  # noqa: ARG004
    ) -> float:
        return 10

    def get_vertex_alignments(self) -> dict[int, int]:
        """Get the vertex alignments.

        Returns:
            The vertex alignments.

        """
        return self._vertex_alignments
