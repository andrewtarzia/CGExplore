import stk


class CaseData:
    """
    A test case.

    Attributes:

    """

    def __init__(
        self,
        molecule: stk.Molecule,
        custom_torsion_set,
        present_torsions,
        bead_set,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.custom_torsion_set = custom_torsion_set
        self.bead_set = bead_set
        self.present_torsions = present_torsions
        self.name = name
