import stk


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecules: list[stk.Molecule],
        property_dicts: list[dict],
        expected_count: int,
        name: str,
    ) -> None:
        self.molecules = molecules
        self.property_dicts = property_dicts
        self.expected_count = expected_count
        self.name = name
