"""Script to generate and optimise CG models."""

import itertools as it
import logging
from collections import Counter, abc
from copy import deepcopy
from typing import assert_never

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def length_2_heteroleptic_bb_dicts(tstr: str) -> dict[int, int]:
    """Define bb dictionaries available to heteroleptic systems.

    Allows for two ditopic building blocks to be added as:
        0: larger -- constant
        1: smaller
        2: smaller2

    """
    return {
        # tstr:
        "2P4": (({0: [0, 1], 1: [], 2: []}), 4),
        "3P6": (({0: [0, 1, 2], 1: [], 2: []}), 6),
        "4P8": (({0: [0, 1, 2, 3], 1: [], 2: []}), 8),
        "4P82": (({0: [0, 1, 2, 3], 1: [], 2: []}), 8),
        "6P12": (({0: [0, 1, 2, 3, 4, 5], 1: [], 2: []}), 12),
        "6P122": (({0: [0, 1, 2, 3, 4, 5], 1: [], 2: []}), 12),
        "8P162": (({0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [], 2: []}), 16),
        "8P16": (({0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [], 2: []}), 16),
        "2P3": (({0: [0, 1], 1: [], 2: []}), 3),
        "4P6": (({0: [0, 1, 2, 3], 1: [], 2: []}), 6),
        "4P62": (({0: [0, 1, 2, 3], 1: [], 2: []}), 6),
        "6P9": (({0: [0, 1, 2, 3, 4, 5], 1: [], 2: []}), 9),
        "8P12": (({0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [], 2: []}), 12),
    }[tstr]


def length_3_heteroleptic_bb_dicts(tstr: str) -> dict[int, int]:
    """Define bb dictionaries available to heteroleptic systems.

    Allows for two tritopic building blocks to be added as:
        0: larger
        1: larger2
        2: smaller -- constant

    """
    return {
        # tstr:
        "2P3": (({0: [], 1: [], 2: list(range(2, 5))}), 2),
        "4P6": (({0: [], 1: [], 2: list(range(4, 10))}), 4),
        "4P62": (({0: [], 1: [], 2: list(range(4, 10))}), 4),
        "6P9": (({0: [], 1: [], 2: list(range(6, 15))}), 6),
        "8P12": (({0: [], 1: [], 2: list(range(8, 20))}), 8),
    }[tstr]


def length_4_heteroleptic_bb_dicts(tstr: str) -> dict[int, int]:
    """Define bb dictionaries available to heteroleptic systems.

    Allows for two tetratopic building blocks to be added as:
        0: larger
        1: larger2
        2: smaller -- constant

    """
    return {
        # tstr:
        "2P4": (({0: [], 1: [], 2: list(range(2, 6))}), 2),
        "3P6": (({0: [], 1: [], 2: list(range(3, 9))}), 3),
        "4P8": (({0: [], 1: [], 2: list(range(4, 12))}), 4),
        "4P82": (({0: [], 1: [], 2: list(range(4, 12))}), 4),
        "6P12": (({0: [], 1: [], 2: list(range(6, 18))}), 6),
        "6P122": (({0: [], 1: [], 2: list(range(6, 18))}), 6),
        "8P162": (({0: [], 1: [], 2: list(range(8, 24))}), 8),
        "8P16": (({0: [], 1: [], 2: list(range(8, 24))}), 8),
    }[tstr]


def get_potential_bb_dicts(
    tstr: str,
    ratio: tuple[int, int],
    bb_type: str,
) -> abc.Sequence[dict[int, abc.Sequence[int]]]:
    """Get potential building block dictionaries."""
    match bb_type:
        case "ditopic":
            possibilities, count_to_add = length_2_heteroleptic_bb_dicts(tstr)
            current_counter = max(
                [
                    max(possibilities[i])
                    for i in possibilities
                    if len(possibilities[i]) != 0
                ]
            )

        case "tritopic":
            possibilities, count_to_add = length_3_heteroleptic_bb_dicts(tstr)
            # Use minus one because of the +1 later on needed for other states.
            current_counter = -1

        case "tetratopic":
            possibilities, count_to_add = length_4_heteroleptic_bb_dicts(tstr)
            # Use minus one because of the +1 later on needed for other states.
            current_counter = -1

        case _ as unreachable:
            assert_never(unreachable)

    modifiable = [i for i in possibilities if len(possibilities[i]) == 0]

    saved = set()
    possible_dicts = []
    for combo in it.product(modifiable, repeat=count_to_add):
        counted = Counter(combo).values()
        current_ratio = [i / min(counted) for i in counted]
        if len(current_ratio) != len(ratio):
            continue

        if tuple(i for i in current_ratio) != ratio:
            continue
        if combo in saved:
            continue
        saved.add(combo)

        new_possibility = deepcopy(possibilities)
        for idx, bb in enumerate(combo):
            new_possibility[bb].append(current_counter + idx + 1)

        possible_dicts.append((len(possible_dicts), new_possibility))

    msg = "bring rmsd checker in here"
    logging.info(msg)
    msg = "use symmetry corrected RMSD on single-bead repr of tstr"
    logging.info(msg)

    return tuple(possible_dicts)
