from dataclasses import dataclass

import cgexplore
import stk


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    forcefield: cgexplore.forcefields.ForceField
    known_decomposition: dict[str, tuple[float, str]]
    name: str
