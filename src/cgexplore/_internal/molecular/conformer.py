# Distributed under the terms of the MIT License.

"""Module for conformer classes.

Author: Andrew Tarzia

"""

from dataclasses import dataclass

import spindry as spd
import stk

from .utilities import spd_to_stk


@dataclass(frozen=True, slots=True)
class Conformer:
    molecule: stk.Molecule
    energy_decomposition: dict
    conformer_id: int | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class SpindryConformer:
    supramolecule: spd.SupraMolecule
    energy_decomposition: dict
    conformer_id: int | None = None
    source: str | None = None

    def to_stk_molecule(self) -> stk.Molecule:
        """Get an stk molecule from spindry."""
        return spd_to_stk(self.supramolecule)
