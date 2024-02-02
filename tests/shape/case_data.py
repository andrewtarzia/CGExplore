import stk


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecule: stk.Molecule,
        shape_dict: dict[str:float],
        expected_points: int,
        shape_string: str,
        name: str,
    ) -> None:
        """Initialize CaseData."""
        self.molecule = molecule
        self.shape_dict = shape_dict
        self.expected_points = expected_points
        self.shape_string = shape_string
        self.name = name
