"""Script to generate and optimise CG models."""

import itertools as it
import logging
from collections import Counter, abc, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, assert_never

import stk

from cgexplore._internal.scram.enumeration import TopologyIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def length_2_heteroleptic_bb_dicts(
    tstr: str,
) -> tuple[dict[int, list[int]], int]:
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


def length_3_heteroleptic_bb_dicts(
    tstr: str,
) -> tuple[dict[int, list[int]], int]:
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


def length_4_heteroleptic_bb_dicts(
    tstr: str,
) -> tuple[dict[int, list[int]], int]:
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
    study_type: Literal["ditopic", "tritopic", "tetratopic"],
) -> list[tuple[int, dict[int, list[int]]]]:
    """Get potential building block dictionaries from known topology graphs.

    Parameters:
        tstr:
            A key to known topology graphs and their building dictionary.

        study_type:
            `ditopic`, `tetratopic`, `tritopic` explore 1:1:1 heteroleptic
            systems with distinct 2-,4-,3-functional group building blocks,
            respectively. If you are using this in conjuction with graph
            screening, use `get_custom_bb_configurations`.

    """
    match study_type:
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
    possible_dicts: list[tuple[int, dict[int, list[int]]]] = []
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

    msg = (
        "bring rmsd checker in here: use symmetry corrected RMSD on "
        "single-bead repr of tstr"
    )
    logger.info(msg)

    return possible_dicts


@dataclass
class BuildingBlockConfiguration:
    """Naming convention for building block configurations."""

    idx: int
    building_block_idx_map: dict[stk.BuildingBlock, int]
    building_block_idx_dict: dict[int, abc.Sequence[int]]

    def get_building_block_dictionary(
        self,
    ) -> dict[stk.BuildingBlock, abc.Sequence[int]]:
        idx_map = {idx: bb for bb, idx in self.building_block_idx_map.items()}
        return {
            idx_map[idx]: tuple(vertices)
            for idx, vertices in self.building_block_idx_dict.items()
        }

    def get_hashable_bbidx_dict(
        self,
    ) -> abc.Sequence[tuple[int, abc.Sequence[int]]]:
        """Get a hashable representation of the building block dictionary."""
        return tuple(sorted(self.building_block_idx_dict.items()))

    def __str__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return (
            f"{self.__class__.__name__}(idx={self.idx}, "
            f"building_block_idx_dict={self.building_block_idx_dict})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the OMMTrajectory."""
        return str(self)


def get_custom_bb_configurations(  # noqa: C901
    iterator: TopologyIterator,
) -> abc.Sequence[BuildingBlockConfiguration]:
    """Get potential building block dictionaries."""
    # Get building blocks with the same functional group count - these are
    # swappable.
    building_blocks_by_fg = {
        i: i.get_num_functional_groups() for i in iterator.building_blocks
    }

    count_of_fg_types: dict[int, int] = defaultdict(int)
    fg_counts_by_building_block: dict[int, int] = defaultdict(int)

    for bb, count in iterator.building_block_counts.items():
        fg_counts_by_building_block[bb.get_num_functional_groups()] += count
        count_of_fg_types[bb.get_num_functional_groups()] += 1

    modifiable_types = tuple(
        fg_count for fg_count, count in count_of_fg_types.items() if count > 1
    )
    if len(modifiable_types) != 1:
        msg = (
            f"modifiable_types is len {len(modifiable_types)}. If 0"
            ", then you have no need to screen building block configurations."
            " If greater than 2, then this code cannot handle this yet. Sorry!"
        )
        raise RuntimeError(msg)

    # Get the associated vertex ids.
    modifiable_vertices = {
        fg_count: iterator.vertex_types_by_fg[fg_count]
        for fg_count in iterator.vertex_types_by_fg
        # ASSUMES 1 modifiable FG.
        if fg_count == modifiable_types[0]
    }

    unmodifiable_vertices = {
        fg_count: iterator.vertex_types_by_fg[fg_count]
        for fg_count in iterator.vertex_types_by_fg
        # ASSUMES 1 modifiable FG.
        if fg_count != modifiable_types[0]
    }

    # Count of functional groups: number of vertices that need adding.
    count_to_add = {
        i: fg_counts_by_building_block[i] for i in modifiable_types
    }

    if len(count_to_add) != 1:
        msg = (
            f"count to add is len {len(count_to_add)}. If greater than 1, "
            "then this code cannot handle this yet. Sorry!"
        )
        raise RuntimeError(msg)

    bb_map = {bb: idx for idx, bb in enumerate(building_blocks_by_fg)}

    empty_bb_dict: dict[int, list[int]] = {}
    for bb, fg_count in building_blocks_by_fg.items():
        if fg_count in modifiable_types:
            empty_bb_dict[bb_map[bb]] = []
        else:
            empty_bb_dict[bb_map[bb]] = list(unmodifiable_vertices[fg_count])

    # ASSUMES 1 modifiable FG.
    modifiable_bb_idx = tuple(
        bb_idx
        for bb_idx, vertices in empty_bb_dict.items()
        if len(vertices) == 0
    )
    modifiable_bb_idx_counted = []
    for bb, count in iterator.building_block_counts.items():
        idx = bb_map[bb]
        if idx not in modifiable_bb_idx:
            continue
        modifiable_bb_idx_counted.extend([idx] * count)

    # Iterate over the placement of the bb indices.
    vertex_map = {
        v_idx: idx
        for idx, v_idx in enumerate(modifiable_vertices[modifiable_types[0]])
    }
    iteration = it.product(
        # ASSUMES 1 modifiable FG.
        *(modifiable_bb_idx for i in modifiable_vertices[modifiable_types[0]])
    )

    saved_bb_dicts = set()
    possible_dicts: list[BuildingBlockConfiguration] = []

    for config in iteration:
        if sorted(config) != modifiable_bb_idx_counted:
            continue

        bb_config_dict = {
            vertex_id: config[vertex_map[vertex_id]]
            for vertex_id in modifiable_vertices[modifiable_types[0]]
        }

        new_possibility = deepcopy(empty_bb_dict)
        for vertex_id, bb_idx in bb_config_dict.items():
            new_possibility[bb_idx].append(vertex_id)

        bbconfig = BuildingBlockConfiguration(
            idx=len(possible_dicts),
            building_block_idx_map=bb_map,
            building_block_idx_dict={
                i: tuple(j) for i, j in new_possibility.items()
            },
        )

        if bbconfig.get_hashable_bbidx_dict() in saved_bb_dicts:
            continue
        # Check for deduplication.
        saved_bb_dicts.add(bbconfig.get_hashable_bbidx_dict())

        possible_dicts.append(bbconfig)

    msg = (
        "bring rmsd checker in here: use symmetry corrected RMSD on "
        "single-bead repr of tstr"
    )
    logger.info(msg)

    return tuple(possible_dicts)
