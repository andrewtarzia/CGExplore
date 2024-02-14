import cgexplore
import stk


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecule: stk.Molecule,
        forcefield_library: cgexplore.forcefields.ForceFieldLibrary,
        bond_ranges: tuple[cgexplore.terms.TargetBondRange],
        angle_ranges: tuple[cgexplore.terms.TargetAngleRange],
        torsion_ranges: tuple[cgexplore.terms.TargetTorsionRange],
        nonbonded_ranges: tuple[cgexplore.terms.TargetNonbondedRange],
        present_bonds: tuple[tuple],
        present_angles: tuple[tuple],
        present_nonbondeds: tuple[tuple],
        present_torsions: tuple[tuple],
        num_forcefields: int,
        library_string: str,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.forcefield_library = forcefield_library
        for i in bond_ranges:
            self.forcefield_library.add_bond_range(i)
        for i in angle_ranges:
            self.forcefield_library.add_angle_range(i)
        for i in torsion_ranges:
            self.forcefield_library.add_torsion_range(i)
        for i in nonbonded_ranges:
            self.forcefield_library.add_nonbonded_range(i)
        self.num_forcefields = num_forcefields
        self.present_bonds = present_bonds
        self.present_angles = present_angles
        self.present_nonbondeds = present_nonbondeds
        self.present_torsions = present_torsions
        self.library_string = library_string
        self.name = name
