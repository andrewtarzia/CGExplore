import cgexplore
import stk


class CaseData:
    """A test case."""

    def __init__(
        self,
        molecule: stk.Molecule,
        forcefield: cgexplore.forcefield.ForceField,
        topology_xml_string: str,
        name: str,
    ) -> None:
        self.molecule = molecule
        self.forcefield = forcefield
        self.topology_xml_string = topology_xml_string
        self.name = name
