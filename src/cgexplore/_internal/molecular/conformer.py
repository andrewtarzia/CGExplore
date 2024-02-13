# Distributed under the terms of the MIT License.

"""Module for conformer classes.

Author: Andrew Tarzia

"""

from dataclasses import dataclass

import stk


@dataclass
class Conformer:
    """Define conformer information."""

    molecule: stk.Molecule
    energy_decomposition: dict
    conformer_id: int | None = None
    source: str | None = None
