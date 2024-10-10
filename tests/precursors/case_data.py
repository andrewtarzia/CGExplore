from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cgexplore.molecular import Precursor


@dataclass(slots=True, frozen=True)
class CaseData:
    precursor: Precursor
    precursor_name: str
    num_fgs: int
    bead_set: dict
    smiles: str
    position_matrix: npt.NDArray[np.float64]
    name: str
