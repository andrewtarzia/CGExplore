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
        length_dict,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.geommeasure = geommeasure
        self.length_dict = length_dict
        self.name = name
