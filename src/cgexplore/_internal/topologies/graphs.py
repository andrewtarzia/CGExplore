"""New stk topologies of cages."""

from collections import abc

import numpy as np
import stk


def stoich_map(tstr: str) -> int:
    """Stoichiometry maps to the number of building blocks."""
    return {
        "2P3": 5,
        "4P6": 10,
        "4P62": 10,
        "6P9": 15,
        "8P12": 20,
        "2P4": 6,
        "3P6": 9,
        "4P8": 12,
        "4P82": 12,
        "6P12": 18,
        "8P16": 24,
        "6P122": 18,
        "8P162": 24,
        "12P24": 36,
        "6P8": 14,
    }[tstr]


class UnalignedM1L2(stk.cage.Cage):
    """New topology definition."""

    _vertex_prototypes = (
        stk.cage.UnaligningVertex(0, np.array([0, 0, 0])),
        stk.cage.UnaligningVertex(
            1, np.array([-3, 0, 0]), use_neighbor_placement=False
        ),
        stk.cage.UnaligningVertex(
            2, np.array([3, 0, 0]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[1]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[2]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[1]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[2]),
    )


class M4L82(stk.cage.Cage):
    """Cage topology."""

    _non_linears = (
        stk.cage.NonLinearVertex(0, [0, 0, np.sqrt(6) / 2]),
        stk.cage.NonLinearVertex(1, [-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
        stk.cage.NonLinearVertex(2, [1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
        stk.cage.NonLinearVertex(3, [0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6]),
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
            use_neighbor_placement=False,
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
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=10,
            position=paired_wall_1_coord - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=11,
            position=paired_wall_2_coord - wall_2_shift,
            use_neighbor_placement=False,
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
        stk.cage.NonLinearVertex(0, [2, 0, 0]),
        stk.cage.NonLinearVertex(1, [0, 2, 0]),
        stk.cage.NonLinearVertex(2, [-2, 0, 0]),
        stk.cage.NonLinearVertex(3, [0, -2, 0]),
        stk.cage.LinearVertex(4, [1, 1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(5, [1, 1, -0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(6, [1, -1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(7, [1, -1, -0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(8, [-1, -1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(9, [-1, -1, -0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(10, [-1, 1, 0.5], use_neighbor_placement=False),
        stk.cage.LinearVertex(11, [-1, 1, -0.5], use_neighbor_placement=False),
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
        stk.cage.NonLinearVertex(0, [1.25, 0, 0]),
        stk.cage.NonLinearVertex(1, [-1.25, 0, 0]),
        stk.cage.NonLinearVertex(2, [0, 1.25, 0]),
        stk.cage.NonLinearVertex(3, [0, -1.25, 0]),
        stk.cage.NonLinearVertex(4, [0.625, 0.625, 0.88]),
        stk.cage.NonLinearVertex(5, [0.625, -0.625, 0.88]),
        stk.cage.NonLinearVertex(6, [-0.625, 0.625, 0.88]),
        stk.cage.NonLinearVertex(7, [-0.625, -0.625, 0.88]),
        stk.cage.NonLinearVertex(8, [0.625, 0.625, -0.88]),
        stk.cage.NonLinearVertex(9, [0.625, -0.625, -0.88]),
        stk.cage.NonLinearVertex(10, [-0.625, 0.625, -0.88]),
        stk.cage.NonLinearVertex(11, [-0.625, -0.625, -0.88]),
        stk.cage.LinearVertex(
            12, [0.9, 0.31, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            13, [0.9, 0.31, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            14, [0.9, -0.31, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            15, [0.9, -0.31, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            16, [-0.9, 0.31, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            17, [-0.9, 0.31, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            18, [-0.9, -0.31, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            19, [-0.9, -0.31, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            20, [0.31, 0.9, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            21, [0.31, 0.9, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            22, [-0.31, 0.9, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            23, [-0.31, 0.9, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            24, [0.31, -0.9, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            25, [0.31, -0.9, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            26, [-0.31, -0.9, 0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            27, [-0.31, -0.9, -0.31], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            28, [0.58, 0, 0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            29, [-0.58, 0, 0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            30, [0, 0.58, 0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            31, [0, -0.58, 0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            32, [0.58, 0, -0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            33, [-0.58, 0, -0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            34, [0, 0.58, -0.82], use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            35, [0, -0.58, -0.82], use_neighbor_placement=False
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
        """Get the vertex alignments."""
        return self._vertex_alignments


class M6L122(stk.cage.Cage):
    """Cage topology."""

    _x = 2 * np.sqrt(3) / 4
    _y = 2
    _non_linears = (
        stk.cage.NonLinearVertex(0, [0, _x, 1]),
        stk.cage.NonLinearVertex(1, [_y / 2, -_x, 1]),
        stk.cage.NonLinearVertex(2, [-_y / 2, -_x, 1]),
        stk.cage.NonLinearVertex(3, [0, _x, -1]),
        stk.cage.NonLinearVertex(4, [_y / 2, -_x, -1]),
        stk.cage.NonLinearVertex(5, [-_y / 2, -_x, -1]),
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

    paired_wall_3_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[4], _non_linears[5])
        )
        / 2
    )
    wall_3_shift = np.array((0.2, 0.2, 0))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            6,
            np.array([0, _x, 0]) + wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            7,
            np.array([0, _x, 0]) - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            8,
            np.array([_y / 2, -_x, 0]) + wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            9,
            np.array([_y / 2, -_x, 0]) - wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            10,
            np.array([-_y / 2, -_x, 0]) + wall_3_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            11,
            np.array([-_y / 2, -_x, 0]) - wall_3_shift,
            use_neighbor_placement=False,
        ),
        ##
        stk.cage.LinearVertex.init_at_center(
            id=12,
            vertices=(_non_linears[0], _non_linears[1]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=13,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=14,
            vertices=(_non_linears[2], _non_linears[0]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=15,
            vertices=(_non_linears[3], _non_linears[4]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=16,
            vertices=(_non_linears[4], _non_linears[5]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=17,
            vertices=(_non_linears[5], _non_linears[3]),
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[12]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[14]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[7]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[12]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[13]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[9]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[14]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[10]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[15]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[17]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[7]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[15]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[16]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[8]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[9]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[16]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[17]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[10]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[11]),
    )


class M8L162(stk.cage.Cage):
    """Cage topology."""

    _non_linears = (
        stk.cage.NonLinearVertex(0, [1, 1, 1]),
        stk.cage.NonLinearVertex(1, [1, -1, 1]),
        stk.cage.NonLinearVertex(2, [-1, -1, 1]),
        stk.cage.NonLinearVertex(3, [-1, 1, 1]),
        stk.cage.NonLinearVertex(4, [1, 1, -1]),
        stk.cage.NonLinearVertex(5, [1, -1, -1]),
        stk.cage.NonLinearVertex(6, [-1, -1, -1]),
        stk.cage.NonLinearVertex(7, [-1, 1, -1]),
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

    paired_wall_3_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[4], _non_linears[5])
        )
        / 2
    )
    wall_3_shift = np.array((0.2, 0.2, 0))

    paired_wall_4_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[6], _non_linears[7])
        )
        / 2
    )
    wall_4_shift = np.array((0.2, 0.2, 0))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            8,
            np.array([1, 1, 0]) + wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            9,
            np.array([1, 1, 0]) - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            10,
            np.array([1, -1, 0]) + wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            11,
            np.array([1, -1, 0]) - wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            12,
            np.array([-1, -1, 0]) + wall_3_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            13,
            np.array([-1, -1, 0]) - wall_3_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            14,
            np.array([-1, 1, 0]) + wall_4_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            15,
            np.array([-1, 1, 0]) - wall_4_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex.init_at_center(
            id=16,
            vertices=(_non_linears[0], _non_linears[1]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=17,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=18,
            vertices=(_non_linears[2], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=19,
            vertices=(_non_linears[3], _non_linears[0]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=20,
            vertices=(_non_linears[4], _non_linears[5]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=21,
            vertices=(_non_linears[5], _non_linears[6]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=22,
            vertices=(_non_linears[6], _non_linears[7]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=23,
            vertices=(_non_linears[7], _non_linears[4]),
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[16]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[19]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[8]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[9]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[16]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[17]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[11]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[17]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[18]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[12]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[13]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[18]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[19]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[14]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[15]),
        stk.Edge(16, _vertex_prototypes[4], _vertex_prototypes[20]),
        stk.Edge(17, _vertex_prototypes[4], _vertex_prototypes[23]),
        stk.Edge(18, _vertex_prototypes[4], _vertex_prototypes[8]),
        stk.Edge(19, _vertex_prototypes[4], _vertex_prototypes[9]),
        stk.Edge(20, _vertex_prototypes[5], _vertex_prototypes[20]),
        stk.Edge(21, _vertex_prototypes[5], _vertex_prototypes[21]),
        stk.Edge(22, _vertex_prototypes[5], _vertex_prototypes[10]),
        stk.Edge(23, _vertex_prototypes[5], _vertex_prototypes[11]),
        stk.Edge(24, _vertex_prototypes[6], _vertex_prototypes[21]),
        stk.Edge(25, _vertex_prototypes[6], _vertex_prototypes[22]),
        stk.Edge(26, _vertex_prototypes[6], _vertex_prototypes[12]),
        stk.Edge(27, _vertex_prototypes[6], _vertex_prototypes[13]),
        stk.Edge(28, _vertex_prototypes[7], _vertex_prototypes[22]),
        stk.Edge(29, _vertex_prototypes[7], _vertex_prototypes[23]),
        stk.Edge(30, _vertex_prototypes[7], _vertex_prototypes[14]),
        stk.Edge(31, _vertex_prototypes[7], _vertex_prototypes[15]),
    )
