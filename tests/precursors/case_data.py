import numpy as np
import numpy.typing as npt
from cgexplore.molecule_construction import Precursor


class CaseData:
    """A test case."""

    def __init__(
        self,
        precursor: Precursor,
        precursor_name: str,
        num_fgs: int,
        bead_set: dict,
        smiles: str,
        position_matrix: npt.NDArray[np.float64],
        name: str,
    ) -> None:
        self.precursor = precursor
        self.precursor_name = precursor_name
        self.num_fgs = num_fgs
        self.bead_set = bead_set
        self.smiles = smiles
        self.position_matrix = position_matrix
        self.name = name
