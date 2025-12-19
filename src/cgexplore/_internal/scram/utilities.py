"""Utilities module."""

from collections import abc

import agx
import numpy as np
import stk


def points_on_sphere(
    sphere_radius: float,
    num_points: int,
    angle_rotation: float,
) -> np.ndarray:
    """Get the points on a sphere."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(num_points)
    z = np.linspace(
        1 - 1.0 / num_points,
        1.0 / num_points - 1.0,
        num_points,
    )
    radius = np.sqrt(1 - z * z)
    points = np.zeros((3, num_points))
    points[0, :] = sphere_radius * np.cos(theta) * radius
    points[1, :] = sphere_radius * np.sin(theta) * radius
    points[2, :] = z * sphere_radius

    axis = np.array((1.0, 0.0, 0.0))
    moving_points = points.T

    rot_mat = stk.rotation_matrix_arbitrary_axis(
        angle=np.radians(angle_rotation),
        axis=axis,
    )
    new_points = rot_mat @ moving_points.T
    new_points = new_points.T

    return np.array(new_points, dtype=np.float64)


def get_stk_topology_code(
    tfun: abc.Callable,
) -> tuple[agx.TopologyCode, list[np.ndarray]]:
    """Get the default stk graph."""
    vps = tfun._vertex_prototypes  # type: ignore[attr-defined] # noqa: SLF001
    eps = tfun._edge_prototypes  # type: ignore[attr-defined] # noqa: SLF001

    combination = [(i.get_vertex1_id(), i.get_vertex2_id()) for i in eps]
    tc = agx.TopologyCode(id=0, vertex_map=combination)
    positions = [i.get_position() for i in vps]

    return tc, positions
