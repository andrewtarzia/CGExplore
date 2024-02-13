# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_dihedral(
    pt1: np.ndarray,
    pt2: np.ndarray,
    pt3: np.ndarray,
    pt4: np.ndarray,
) -> float:
    """Calculate the dihedral between four points.

    Uses Praxeolitic formula --> 1 sqrt, 1 cross product
    Output in range (-pi to pi).
    From: https://stackoverflow.com/questions/20305272/
    dihedral-torsion-angle-from-four-points-in-cartesian-
    coordinates-in-python
    (new_dihedral(p))
    """
    p0 = np.asarray(pt1)
    p1 = np.asarray(pt2)
    p2 = np.asarray(pt3)
    p3 = np.asarray(pt4)

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))
