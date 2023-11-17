import stk


class CaseData:
    """A test case.

    Attributes:

    """

    def __init__(
        self,
        molecule: stk.Molecule,
        force_field,
        xml_string,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.force_field = force_field
        self.xml_string = xml_string
        self.name = name
