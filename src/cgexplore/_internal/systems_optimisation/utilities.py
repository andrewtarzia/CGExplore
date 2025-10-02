# Distributed under the terms of the MIT License.

"""Utilities module."""

import logging
import pathlib
from collections import abc
from copy import deepcopy

import openmm
import stk

from cgexplore.forcefields import ForceField
from cgexplore.molecular import CgBead
from cgexplore.terms import (
    TargetAngle,
    TargetBond,
    TargetCosineAngle,
    TargetNonbonded,
    TargetPyramidAngle,
    TargetTorsion,
)
from cgexplore.utilities import convert_pyramid_angle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def element_from_type(
    bead_type: str,
    present_beads: abc.Sequence[CgBead],
) -> str:
    """Get element of cgbead from type of cgbead."""
    return next(
        i.element_string for i in present_beads if i.bead_type == bead_type
    )


def define_bond(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetBond:
    """Define target from a known structured list."""
    return TargetBond(
        type1=interaction_key[0],
        type2=interaction_key[1],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        bond_r=openmm.unit.Quantity(
            value=interaction_list[1], unit=openmm.unit.angstrom
        ),
        bond_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule
            / openmm.unit.mole
            / openmm.unit.nanometer**2,
        ),
    )


def define_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetAngle:
    """Define target from a known structured list."""
    return TargetAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        angle=openmm.unit.Quantity(
            value=interaction_list[1], unit=openmm.unit.degrees
        ),
        angle_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule
            / openmm.unit.mole
            / openmm.unit.radian**2,
        ),
    )


def define_pyramid_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetAngle:
    """Define target from a known structured list."""
    angle = openmm.unit.Quantity(
        value=interaction_list[1],
        unit=openmm.unit.degrees,
    )
    opposite_angle = openmm.unit.Quantity(
        value=convert_pyramid_angle(angle.value_in_unit(angle.unit)),
        unit=angle.unit,
    )
    return TargetPyramidAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        angle=angle,
        opposite_angle=opposite_angle,
        angle_k=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.kilojoule
            / openmm.unit.mole
            / openmm.unit.radian**2,
        ),
    )


def define_cosine_angle(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetCosineAngle:
    """Define target from a known structured list."""
    return TargetCosineAngle(
        type1=interaction_key[0],
        type2=interaction_key[1],
        type3=interaction_key[2],
        element1=element_from_type(interaction_key[0], present_beads),
        element2=element_from_type(interaction_key[1], present_beads),
        element3=element_from_type(interaction_key[2], present_beads),
        n=interaction_list[1],
        b=interaction_list[2],
        angle_k=openmm.unit.Quantity(
            value=interaction_list[3],
            unit=openmm.unit.kilojoule / openmm.unit.mole,
        ),
    )


def define_torsion(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetTorsion:
    """Define target from a known structured list."""
    measured_atom_ids = tuple(int(i) for i in interaction_list[1])
    if len(measured_atom_ids) != 4:  # noqa: PLR2004
        msg = (
            f"trying to define torsion with measured atoms {measured_atom_ids}"
            ", should be 4"
        )
        raise RuntimeError(msg)
    return TargetTorsion(
        search_string=tuple(i for i in interaction_key),
        search_estring=tuple(
            element_from_type(test, present_beads) for test in interaction_key
        ),
        measured_atom_ids=measured_atom_ids,
        phi0=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.degrees,
        ),
        torsion_k=openmm.unit.Quantity(
            value=interaction_list[3],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        torsion_n=interaction_list[4],
    )


def define_nonbonded(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetNonbonded:
    """Define target from a known structured list."""
    return TargetNonbonded(
        bead_class=interaction_key[0],
        bead_element=element_from_type(interaction_key[0], present_beads),
        epsilon=openmm.unit.Quantity(
            value=interaction_list[1],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        sigma=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.angstrom,
        ),
        force="custom-excl-vol",
    )


def define_lennardjones(
    interaction_key: str,
    interaction_list: list,
    present_beads: abc.Sequence[CgBead],
) -> TargetNonbonded:
    """Define target from a known structured list."""
    return TargetNonbonded(
        bead_class=interaction_key[0],
        bead_element=element_from_type(interaction_key[0], present_beads),
        epsilon=openmm.unit.Quantity(
            value=interaction_list[1],
            unit=openmm.unit.kilojoules_per_mole,
        ),
        sigma=openmm.unit.Quantity(
            value=interaction_list[2],
            unit=openmm.unit.angstrom,
        ),
        force="custom-lj",
    )


def get_neighbour_library(
    chromosome_name: abc.Sequence[int],
    modifiable_gene_ids: abc.Sequence[int],
) -> list[str]:
    new_chromosomes = []

    for gene_id in modifiable_gene_ids:
        curr_gene = chromosome_name[gene_id]

        for i in (-2, -1, +1, +2):
            new_gene = curr_gene + i
            new_chromo = list(chromosome_name)
            new_chromo[gene_id] = new_gene

            if new_gene < 0:
                continue
            new_chromosomes.append("".join(str(ng) for ng in new_chromo))

    return new_chromosomes


def yield_near_models(
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path | str,
    replace_name: str,
    neighbour_library: list,
) -> abc.Iterator[stk.Molecule]:
    """Yield structures of models with neighbouring force field parameters.

    Keywords:

        molecule:
            The molecule to modify the position matrix of.

        name:
            Name of molecule, holding force field ID.

        output_dir:
            Directory with optimisation outputs saved.

        replace_name:
            String to replace in filename.

        neighbour_library:
            IDs of force fields with nearby parameters, defined in
            `define_forcefields.py`.

    Returns:
        An stk molecule.

    """
    for new_ff_id in neighbour_library:
        new_name = name.replace(replace_name, f"{new_ff_id}")
        new_fina_mol_file = pathlib.Path(output_dir) / f"{new_name}_final.mol"
        if new_fina_mol_file.exists():
            yield molecule.with_structure_from_file(str(new_fina_mol_file))


def merge_definer_dicts(
    original_definer_dict: dict,
    new_definer_dicts: dict,
) -> dict:
    """Merge multiple definer dicts, overring the original."""
    temp_definer_dict = deepcopy(original_definer_dict)
    for new_definer_dict in new_definer_dicts:
        temp_definer_dict.update(new_definer_dict)
    return temp_definer_dict


def get_forcefield_from_dict(  # noqa: PLR0913
    identifier: str,
    prefix: str,
    vdw_bond_cutoff: int,
    present_beads: abc.Sequence[CgBead],
    definer_dict: dict,
    verbose: bool = True,
) -> ForceField:
    """Get forcefield from a definer dict."""
    bond_terms: list = []
    angle_terms: list[TargetAngle | TargetCosineAngle] = []
    torsion_terms: list = []
    nonbonded_terms: list = []
    for key_, term in definer_dict.items():
        if term[0] == "bond":
            bond_terms.append(
                define_bond(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "angle":
            angle_terms.append(
                define_angle(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "pyramid":
            angle_terms.append(
                define_pyramid_angle(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "cosine":
            angle_terms.append(
                define_cosine_angle(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "tors":
            torsion_terms.append(
                define_torsion(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "nb":
            nonbonded_terms.append(
                define_nonbonded(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        elif term[0] == "custom-lj":
            nonbonded_terms.append(
                define_lennardjones(
                    interaction_key=key_,
                    interaction_list=term,
                    present_beads=present_beads,
                )
            )

        else:
            msg = f"{term[0]} not found in known terms."
            raise RuntimeError(msg)

    return ForceField(
        identifier=identifier,
        prefix=prefix,
        present_beads=present_beads,
        bond_targets=tuple(bond_terms),
        angle_targets=tuple(angle_terms),
        torsion_targets=tuple(torsion_terms),
        nonbonded_targets=tuple(nonbonded_terms),
        vdw_bond_cutoff=vdw_bond_cutoff,
        verbose=verbose,
    )
