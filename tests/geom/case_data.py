from dataclasses import dataclass

import stk

import cgexplore


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    geommeasure: cgexplore.analysis.GeomMeasure
    length_dict: dict[str : list[float]]
    angle_dict: dict[str : list[float]]
    torsion_dict: dict[str : list[float]]
    radius_gyration: float
    max_diam: float
    name: str
