from collections import abc
from dataclasses import dataclass

import cgexplore as cgx


@dataclass(slots=True, frozen=True)
class CaseData:
    """A test case."""

    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator
    known_map: dict[int, dict]
    known_types: dict[int, str]
    num: int
    term_ids: abc.Sequence[int]
    topo_ids: abc.Sequence[int]
    va_ids: abc.Sequence[int]
    prec_ids: abc.Sequence[int]
    merged_size: int
    pop1_fitness: list[int]
    pop1_best: tuple
    pop1_worst: tuple
    pop1_elite_fitness: int
    np: int
    name: str
