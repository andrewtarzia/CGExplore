# Distributed under the terms of the MIT License.

"""Utilities for using Spindry module.

Author: Andrew Tarzia

"""

import logging

import openmm
import spindry as spd
import stk

from cgexplore._internal.forcefields.forcefield import ForceField

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_unforced_supramolecule(
    hgcomplex: stk.ConstructedMolecule,
) -> spd.Potential:
    return spd.SupraMolecule(
        atoms=(
            spd.Atom(id=atom.get_id(), element_string=atom.__class__.__name__)
            for atom in hgcomplex.get_atoms()
        ),
        bonds=(
            spd.Bond(
                id=i,
                atom_ids=(
                    bond.get_atom1().get_id(),
                    bond.get_atom2().get_id(),
                ),
            )
            for i, bond in enumerate(hgcomplex.get_bonds())
        ),
        position_matrix=hgcomplex.get_position_matrix(),
    )


def get_supramolecule(
    hgcomplex: stk.ConstructedMolecule,
    forcefield: ForceField,
) -> spd.Potential:
    nonbonded_targets = forcefield.get_targets()["nonbondeds"]

    epsilons = []
    sigmas = []
    for atom in hgcomplex.get_atoms():
        atom_estring = atom.__class__.__name__
        cgbead = forcefield.get_bead_library().get_cgbead_from_element(
            atom_estring
        )
        for target_term in nonbonded_targets:
            if target_term.bead_class != cgbead.bead_class:
                continue
            epsilons.append(
                target_term.epsilon.value_in_unit(
                    openmm.unit.kilojoules_per_mole
                )
            )
            sigmas.append(
                target_term.sigma.value_in_unit(openmm.unit.angstrom)
            )

    return spd.SupraMolecule(
        atoms=(
            spd.Atom(
                id=atom.get_id(),
                element_string=atom.__class__.__name__,
                epsilon=epsilons[atom.get_id()],
                sigma=sigmas[atom.get_id()],
            )
            for atom in hgcomplex.get_atoms()
        ),
        bonds=(
            spd.Bond(
                id=i,
                atom_ids=(
                    bond.get_atom1().get_id(),
                    bond.get_atom2().get_id(),
                ),
            )
            for i, bond in enumerate(hgcomplex.get_bonds())
        ),
        position_matrix=hgcomplex.get_position_matrix(),
    )
