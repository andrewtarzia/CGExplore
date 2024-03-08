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


def get_rotation(radians: float) -> np.ndarray:
    """Get rotation of radians."""
    c, s = np.cos(radians), np.sin(radians)
    return np.array(((c, -s), (s, c)))


def vnorm_r(v: np.ndarray, distance: float) -> np.ndarray:
    """Normalise rotation."""
    return v / np.linalg.norm(v) * distance
