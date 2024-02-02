import cgexplore
import stk


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecule: stk.Molecule,
        forcefield: cgexplore.forcefield.ForceField,
        xml_string: str,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.forcefield = forcefield
        self.xml_string = xml_string
        self.name = name
