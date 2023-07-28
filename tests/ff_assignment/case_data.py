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
        torsion_dict,
        custom_torsion_definition,
        custom_torsion_set,
        bead_set,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.custom_torsion_definition = custom_torsion_definition
        self.custom_torsion_set = custom_torsion_set
        self.bead_set = bead_set
        self.torsion_dict = torsion_dict
        self.optimizer = cgexplore.CGOptimizer(
            param_pool=bead_set,
            custom_torsion_set=custom_torsion_set,
            bonds=False,
            angles=False,
            torsions=False,
            vdw=False,
        )
        self.name = name
