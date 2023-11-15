from cgexplore.molecule_construction import Precursor


class CaseData:
    """A test case.

    Attributes:

    """

    def __init__(
        self,
        precursor: Precursor,
        precursor_name: str,
        num_fgs: int,
        bead_set: dict,
        smiles: str,
        position_matrix,
        name: str,
    ) -> None:
        self.precursor = precursor
        self.precursor_name = precursor_name
        self.num_fgs = num_fgs
        self.bead_set = bead_set
        self.smiles = smiles
        self.position_matrix = position_matrix
        self.name = name
