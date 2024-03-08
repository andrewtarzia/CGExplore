"""Generator of host guest conformations using nonbonded interactions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import openmm
import spindry as spd
import stk

from cgexplore._internal.molecular.conformer import SpindryConformer

if TYPE_CHECKING:
    import pathlib
    from collections import abc

    from cgexplore._internal.forcefields.forcefield import ForceField

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class Laundrette:
    """Class to run rigid-body docking."""

    def __init__(  # noqa: PLR0913
        self,
        num_dockings: int,
        naming_prefix: str,
        output_dir: pathlib.Path,
        forcefield: ForceField,
        seed: int,
        step_size: float = 1.0,
        rotation_step_size: float = 2.0,
        num_conformers: int = 200,
        max_attempts: int = 500,
        beta: float = 1.0,
    ) -> None:
        """Initialise Laundrette."""
        self._num_dockings = num_dockings
        self._naming_prefix = naming_prefix
        self._output_dir = output_dir
        self._potential = spd.VaryingEpsilonPotential()
        self._forcefield = forcefield
        self._seed = seed
        self._rng = np.random.default_rng(seed=seed)
        self._step_size = step_size
        self._rotation_step_size = rotation_step_size
        self._num_conformers = num_conformers
        self._max_attempts = max_attempts
        self._beta = beta

    def _get_supramolecule(
        self,
        hgcomplex: stk.ConstructedMolecule,
    ) -> spd.Potential:
        nonbonded_targets = self._forcefield.get_targets()["nonbondeds"]

        epsilons = []
        sigmas = []
        for atom in hgcomplex.get_atoms():
            atom_estring = atom.__class__.__name__
            cgbead = (
                self._forcefield.get_bead_library().get_cgbead_from_element(
                    atom_estring
                )
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

    def run_dockings(
        self,
        host_bb: stk.BuildingBlock,
        guest_bb: stk.BuildingBlock,
    ) -> abc.Iterable[SpindryConformer]:
        """Run the docking algorithm."""
        for docking_id in range(self._num_dockings):
            guest = stk.host_guest.Guest(
                building_block=guest_bb,
                start_vector=guest_bb.get_direction(),
                end_vector=self._rng.random((1, 3))[0],
                # Change the displacement of the guest.
                displacement=self._rng.random((1, 3))[0],
            )

            hgcomplex = stk.ConstructedMolecule(
                topology_graph=stk.host_guest.Complex(
                    host=stk.BuildingBlock.init_from_molecule(host_bb),
                    guests=guest,
                ),
            )
            supramolecule = self._get_supramolecule(hgcomplex=hgcomplex)

            cg = spd.Spinner(
                step_size=self._step_size,
                rotation_step_size=self._rotation_step_size,
                num_conformers=self._num_conformers,
                max_attempts=self._max_attempts,
                beta=self._beta,
                potential_function=self._potential,
                random_seed=self._seed,
            )
            cid = 1
            for supraconformer in cg.get_conformers(
                supramolecule,
                verbose=False,
            ):
                yield SpindryConformer(
                    supramolecule=supraconformer,
                    conformer_id=cid,
                    source=docking_id,
                    energy_decomposition={
                        "potential": supraconformer.get_potential()
                    },
                )
                cid += 1
