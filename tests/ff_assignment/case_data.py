import stk


class CaseData:
    """
    A test case.

    Attributes:

    """

    def __init__(
        self,
        molecule: stk.Molecule,
        force_field,
        present_torsions,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.force_field = force_field
        self.present_torsions = present_torsions
        self.name = name
