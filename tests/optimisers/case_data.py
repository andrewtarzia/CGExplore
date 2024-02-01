import stk


class CaseData:
    """A test case.

    Attributes:

    """

    def __init__(
        self,
        molecule: stk.Molecule,
        forcefield,
        xml_string,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.forcefield = forcefield
        self.xml_string = xml_string
        self.name = name
