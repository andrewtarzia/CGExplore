# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import json
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import stk
from openmm import openmm
from scipy.spatial.distance import euclidean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def convert_pyramid_angle(outer_angle: float) -> float:
    """Some basic trig on square-pyramids."""
    outer_angle = np.radians(outer_angle)
    # Side length, oa, does not matter.
    oa = 1
    ab = 2 * (oa * np.sin(outer_angle / 2))
    ac = ab / np.sqrt(2) * 2
    opposite_angle = 2 * np.arcsin(ac / 2 / oa)
    return round(np.degrees(opposite_angle), 2)


def check_directory(path: pathlib.Path) -> None:
    """Check if a directory exists, make if not."""
    if not path.exists():
        path.mkdir()


def get_atom_distance(
    molecule: stk.Molecule,
    atom1_id: int,
    atom2_id: int,
) -> float:
    """Return the distance between atom1 and atom2.

    Parameters:
        molecule:

        atom1_id:
            The id of atom1.

        atom2_id:
            The id of atom2.

    Returns:
        The euclidean distance between two atoms.

    """
    position_matrix = molecule.get_position_matrix()

    distance = euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id],
    )

    return float(distance)


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


def read_lib(lib_file: str) -> dict:
    """Read lib file."""
    logging.info(f"reading {lib_file}")
    with open(lib_file, "rb") as f:
        return json.load(f)


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


def draw_pie(
    colours: list[str],
    xpos: float,
    ypos: float,
    size: float,
    ax: plt.Axes,
) -> None:
    """Draw a pie chart at a specific point on ax.

    From:
    https://stackoverflow.com/questions/56337732/how-to-plot-scatter-
    pie-chart-using-matplotlib.

    """
    num_points = len(colours)
    if num_points == 1:
        ax.scatter(
            xpos,
            ypos,
            c=colours[0],
            edgecolors="k",
            s=size,
        )
    else:
        ratios = [1 / num_points for i in range(num_points)]
        if sum(ratios) > 1:
            msg = (
                f"sum of ratios needs to be < 1 (np={num_points}, "
                f"ratios={ratios})"
            )
            raise AssertionError(msg)

        markers = []
        previous = 0.0
        # calculate the points of the pie pieces
        for color, ratio in zip(colours, ratios, strict=True):
            this = 2 * np.pi * ratio + previous
            x = [0, *np.cos(np.linspace(previous, this, 100)).tolist(), 0]
            y = [0, *np.sin(np.linspace(previous, this, 100)).tolist(), 0]
            xy = np.column_stack([x, y])
            previous = this
            markers.append(
                {
                    "marker": xy,
                    "s": np.abs(xy).max() ** 2 * np.array(size),
                    "facecolor": color,
                    "edgecolors": "k",
                }
            )

        # scatter each of the pie pieces to create pies
        for marker in markers:
            ax.scatter(xpos, ypos, **marker)
