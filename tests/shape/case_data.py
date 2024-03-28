from dataclasses import dataclass

import stk


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    shape_dict: dict[str:float]
    expected_points: int
    shape_string: str
    name: str
