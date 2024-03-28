from dataclasses import dataclass

import stk


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    name: str
