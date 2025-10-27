from dataclasses import dataclass

import stk

import cgexplore as cgx


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    geommeasure: cgx.analysis.GeomMeasure
    length_dict: dict[tuple[str, str], list[float]]
    angle_dict: dict[tuple[str, str, str], list[float]]
    torsion_dict: dict[str, list[float]]
    radius_gyration: float
    max_diam: float
    name: str
