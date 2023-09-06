import stk
import pathlib


class CaseData:
    """
    A test case.

    Attributes:

    """

    def __init__(
        self,
        molecule: stk.Molecule,
        force_field_library,
        bond_ranges,
        angle_ranges,
        torsion_ranges,
        nonbonded_ranges,
        present_bonds,
        present_angles,
        present_nonbondeds,
        present_torsions,
        present_custom_torsions,
        num_forcefields: int,
        xml_strings: tuple[str],
        library_string: str,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.force_field_library = force_field_library
        for i in bond_ranges:
            self.force_field_library.add_bond_range(i)
        for i in angle_ranges:
            self.force_field_library.add_angle_range(i)
        for i in torsion_ranges:
            self.force_field_library.add_torsion_range(i)
        for i in nonbonded_ranges:
            self.force_field_library.add_nonbonded_range(i)
        self.num_forcefields = num_forcefields
        self.force_fields = tuple(
            force_field_library.yield_forcefields(
                prefix="testff", output_path=pathlib.Path()
            )
        )
        self.present_bonds = present_bonds
        self.present_angles = present_angles
        self.present_nonbondeds = present_nonbondeds
        self.present_torsions = present_torsions
        self.present_custom_torsions = present_custom_torsions
        self.xml_strings = xml_strings
        self.library_string = library_string
        self.name = name
