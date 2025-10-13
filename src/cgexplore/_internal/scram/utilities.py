"""Utilities module."""

from collections import defaultdict

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


def generate_graph_type(
    stoichiometry_map: dict[str, int],
    multiplier: int,
    bb_library: dict[str, stk.BuildingBlock],
) -> str:
    """Automatically get the graph type to match the new naming convention."""
    fgcounts = defaultdict(int)
    for name, stoich in stoichiometry_map.items():
        fgcounts[bb_library[name].get_num_functional_groups()] += (
            stoich * multiplier
        )

    string = ""
    for fgtype, fgnum in sorted(fgcounts.items(), reverse=True):
        string += f"{fgnum}-{fgtype}FG_"
    return string.rstrip("_")
