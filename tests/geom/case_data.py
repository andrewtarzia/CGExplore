import cgexplore
import stk


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecule: stk.Molecule,
        geommeasure: cgexplore.GeomMeasure,
        length_dict: dict[str : list[float]],
        angle_dict: dict[str : list[float]],
        torsion_dict: dict[str : list[float]],
        radius_gyration: float,
        max_diam: float,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.geommeasure = geommeasure
        self.length_dict = length_dict
        self.angle_dict = angle_dict
        self.torsion_dict = torsion_dict
        self.radius_gyration = radius_gyration
        self.max_diam = max_diam
        self.name = name
