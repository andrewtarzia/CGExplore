import stk

import cgexplore


class CaseData:
    """
    A test case.

    Attributes:

    """

    def __init__(
        self,
        molecule: stk.Molecule,
        geommeasure: cgexplore.GeomMeasure,
        length_dict: dict[str : list[float]],
        angle_dict: dict[str : list[float]],
        name: str,
    ) -> None:
        self.molecule = molecule
        self.geommeasure = geommeasure
        self.length_dict = length_dict
        self.angle_dict = angle_dict
        self.name = name
