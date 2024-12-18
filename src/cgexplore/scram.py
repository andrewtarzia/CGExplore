"""scram package."""

from cgexplore._internal.scram.building_block_enum import (
    BuildingBlockConfiguration,
    get_custom_bb_configurations,
    get_potential_bb_dicts,
)
from cgexplore._internal.scram.construction import (
    graph_optimise_cage,
    optimise_cage,
    try_except_construction,
)
from cgexplore._internal.scram.enumeration import (
    IHomolepticTopologyIterator,
    TopologyIterator,
)
from cgexplore._internal.scram.topology_code import Constructed, TopologyCode
from cgexplore._internal.scram.utilities import points_on_sphere, vmap_to_str

__all__ = [
    "BuildingBlockConfiguration",
    "Constructed",
    "IHomolepticTopologyIterator",
    "TopologyCode",
    "TopologyIterator",
    "get_custom_bb_configurations",
    "get_potential_bb_dicts",
    "graph_optimise_cage",
    "optimise_cage",
    "points_on_sphere",
    "try_except_construction",
    "vmap_to_str",
]
