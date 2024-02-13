# Distributed under the terms of the MIT License.

"""Utilities module.

Author: Andrew Tarzia

"""

import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

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
