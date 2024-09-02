from dataclasses import dataclass

import stk

import cgexplore


@dataclass(slots=True, frozen=True)
class CaseData:
    molecule: stk.Molecule
    forcefield: cgexplore.forcefields.ForceField
    topology_xml_string: str
    name: str
