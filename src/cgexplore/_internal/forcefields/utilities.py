# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging

import numpy as np
import numpy.typing as npt
from openmm import openmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def unit_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Returns the unit vector of the vector.

    https://stackoverflow.com/questions/2827393/
    angles-between-two-n-dimensional-vectors-in-python/
    13849249#13849249

    """
    return vector / np.linalg.norm(vector)


def angle_between(
    v1: npt.NDArray[np.float64],
    v2: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64] | None = None,
) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'.

    Defined as ::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

    https://stackoverflow.com/questions/2827393/
    angles-between-two-n-dimensional-vectors-in-python/
    13849249#13849249

    If normal is given, the angle polarity is determined using the
    cross product of the two vectors.

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if normal is not None:
        # Get normal vector and cross product to determine sign.
        cross = np.cross(v1_u, v2_u)
        if np.dot(normal, cross) < 0:
            angle = -angle
    return angle


def custom_excluded_volume_force() -> openmm.CustomNonbondedForce:
    """Define Custom Excluded Volume force."""
    energy_expression = "epsilon*((sigma)/(r))^12;"
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    custom_force = openmm.CustomNonbondedForce(energy_expression)
    custom_force.addPerParticleParameter("sigma")
    custom_force.addPerParticleParameter("epsilon")
    return custom_force


def cosine_periodic_angle_force() -> openmm.CustomAngleForce:
    """Define Custom Angle force."""
    energy_expression = "F*C*(1-A*cos(n * theta));"
    energy_expression += "A = b*(min_n);"
    energy_expression += "C = (n^2 * k)/2;"
    energy_expression += "F = (2/(n ^ 2));"
    custom_force = openmm.CustomAngleForce(energy_expression)
    custom_force.addPerAngleParameter("k")
    custom_force.addPerAngleParameter("n")
    custom_force.addPerAngleParameter("b")
    custom_force.addPerAngleParameter("min_n")
    return custom_force
