# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging
import pathlib

import atomlite
import matplotlib.pyplot as plt
import numpy as np
import stk
from rmsd import check_reflections, int_atom, kabsch_rmsd, reorder_hungarian

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


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
    path.mkdir(exist_ok=True, parents=True)


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


def extract_property(
    path: list[str],
    properties: dict[str, atomlite.Json],
) -> atomlite.Json:
    """Extract property from nested dict."""
    if len(path) == 1:
        value = properties[path[0]]
    elif len(path) == 2:  # noqa: PLR2004
        value = properties[path[0]][path[1]]  # type: ignore[index,call-overload]
    elif len(path) == 3:  # noqa: PLR2004
        value = properties[path[0]][path[1]][path[2]]  # type: ignore[index,call-overload]
    elif len(path) == 4:  # noqa: PLR2004
        value = properties[path[0]][path[1]][path[2]][path[3]]  # type: ignore[index,call-overload]
    else:
        msg = f"{path} is too deep ({len(path)})."
        raise RuntimeError(msg)
    return value


def get_energy_per_bb(
    energy_decomposition: dict[str, tuple[float, str]],
    number_building_blocks: int,
) -> float:
    """Get the energy per building blocks used in most papers."""
    energy_decomp = {}
    for component, component_tup in energy_decomposition.items():
        if component == "total energy":
            energy_decomp[f"{component}_{component_tup[1]}"] = float(
                component_tup[0]
            )
        else:
            just_name = component.split("'")[1]
            key = f"{just_name}_{component_tup[1]}"
            value = float(component_tup[0])
            if key in energy_decomp:
                energy_decomp[key] += value
            else:
                energy_decomp[key] = value

    fin_energy = energy_decomp["total energy_kJ/mol"]
    if not np.isclose(
        sum(
            energy_decomp[i] for i in energy_decomp if "total energy" not in i
        ),
        fin_energy,
    ):
        msg = "energy decompisition does not sum to total energy"
        raise RuntimeError(msg)

    return fin_energy / number_building_blocks


def rmsd_checker(
    unopt_mol: stk.ConstructedMolecule,
    unopt_name: str,
    unopt_glob: list[pathlib.Path],
) -> bool:
    """Check if an un-optimised molecule has a low RMSD to another one."""
    if len(unopt_glob) == 0:
        return False

    p_coord = unopt_mol.with_centroid(
        np.array((0, 0, 0))
    ).get_position_matrix()

    rmsd_threshold = 1

    for other_mol in unopt_glob:
        if other_mol.name.replace(".mol", "") == unopt_name:
            continue

        p_atoms = np.array(
            [int_atom(i.__class__.__name__) for i in unopt_mol.get_atoms()]
        )

        q_mol = stk.BuildingBlock.init_from_file(str(other_mol))
        q_atoms = np.array(
            [int_atom(i.__class__.__name__) for i in q_mol.get_atoms()]
        )
        q_coord = q_mol.with_centroid(
            np.array((0, 0, 0))
        ).get_position_matrix()

        # Apply reorder and reflections.
        result_rmsd, _, _, _ = check_reflections(
            p_atoms,
            q_atoms,
            p_coord,
            q_coord,
            reorder_method=reorder_hungarian,
            rmsd_method=kabsch_rmsd,
        )

        if result_rmsd < rmsd_threshold:
            return True
    return False
