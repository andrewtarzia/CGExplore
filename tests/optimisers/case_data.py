from dataclasses import dataclass

import stk

import cgexplore as cgx


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    forcefield: cgx.forcefields.ForceField
    known_decomposition: dict[str, tuple[float, str]]
    name: str
