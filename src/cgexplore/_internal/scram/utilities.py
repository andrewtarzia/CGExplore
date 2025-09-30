"""Utilities module."""

import logging
from collections import abc

import numpy as np
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def vmap_to_str(vertex_map: abc.Sequence[tuple[int, int]]) -> str:
    """Convert vertex map to str."""
    strs = sorted([f"{i[0]}-{i[1]}" for i in vertex_map])
    return "_".join(strs)


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
