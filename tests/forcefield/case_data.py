from collections import abc

import stk

import cgexplore as cgx


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecule: stk.Molecule,
        forcefield_library: cgx.forcefields.ForceFieldLibrary,
        bond_ranges: abc.Sequence[cgx.terms.TargetBondRange],
        angle_ranges: abc.Sequence[
            cgx.terms.TargetAngleRange
            | cgx.terms.PyramidAngleRange
            | cgx.terms.TargetCosineAngleRange
        ],
        torsion_ranges: abc.Sequence[cgx.terms.TargetTorsionRange],
        nonbonded_ranges: abc.Sequence[cgx.terms.TargetNonbondedRange],
        present_bonds: list[list[cgx.terms.Bond]],
        present_angles: list[list[cgx.terms.Angle | cgx.terms.CosineAngle]],
        present_nonbondeds: list[list[cgx.terms.Nonbonded]],
        present_torsions: list[list[cgx.terms.Torsion]],
        num_forcefields: int,
        library_string: str,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.forcefield_library = forcefield_library
        for bond in bond_ranges:
            self.forcefield_library.add_bond_range(bond)
        for angle in angle_ranges:
            self.forcefield_library.add_angle_range(angle)
        for torsion in torsion_ranges:
            self.forcefield_library.add_torsion_range(torsion)
        for nonbonded in nonbonded_ranges:
            self.forcefield_library.add_nonbonded_range(nonbonded)
        self.num_forcefields = num_forcefields
        self.present_bonds = present_bonds
        self.present_angles = present_angles
        self.present_nonbondeds = present_nonbondeds
        self.present_torsions = present_torsions
        self.library_string = library_string
        self.name = name
