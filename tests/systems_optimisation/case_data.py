from dataclasses import dataclass

import cgexplore as cgx


@dataclass(slots=True, frozen=True)
class CaseData:
    """A test case."""

    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator
    known_map: dict[int, dict]
    known_types: dict[int, dict]
    num: int
    term_ids: tuple[int, ...]
    topo_ids: tuple[int, ...]
    va_ids: tuple[int, ...]
    prec_ids: tuple[int, ...]
    merged_size: int
    pop1_fitness: list[int]
    pop1_best: tuple
    pop1_worst: tuple
    pop1_elite_fitness: int
    np: int
    name: str
