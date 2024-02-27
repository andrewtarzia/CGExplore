# Distributed under the terms of the MIT License.

"""Module for conformer classes.

Author: Andrew Tarzia

"""

from dataclasses import dataclass

import spindry as spd
import stk

from .beads import periodic_table


@dataclass
class Conformer:
    """Define conformer information."""

    molecule: stk.Molecule
    energy_decomposition: dict
    conformer_id: int | None = None
    source: str | None = None


@dataclass
class SpindryConformer:
    """Define conformer information."""

    supramolecule: spd.SupraMolecule
    energy_decomposition: dict
    conformer_id: int | None = None
    source: str | None = None

    def to_stk_molecule(self) -> stk.Molecule:
        """Get an stk molecule from spindry."""
        pt = periodic_table()

        atoms = [
            stk.Atom(id=i.get_id(), atomic_number=pt[i.get_element_string()])
            for i in self.supramolecule.get_atoms()
        ]
        bonds = [
            stk.Bond(
                atom1=atoms[i.get_atom1_id()],
                atom2=atoms[i.get_atom2_id()],
                order=1,
            )
            for i in self.supramolecule.get_bonds()
        ]

        return stk.BuildingBlock.init(
            atoms=atoms,
            bonds=bonds,
            position_matrix=self.supramolecule.get_position_matrix(),
        )
