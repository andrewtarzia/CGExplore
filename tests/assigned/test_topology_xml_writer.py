import pathlib

from .case_data import CaseData


def test_topology_xml_writer(molecule: CaseData) -> None:
    """Test methods toward :meth:`.AssignedSystem._get_topology_xml_string`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    assigned_system = molecule.forcefield.assign_terms(
        molecule=molecule.molecule,
        output_dir=pathlib.Path(__file__).resolve().parent,
        name=molecule.name,
    )
    xml_string = assigned_system._get_topology_xml_string(  # noqa: SLF001
        molecule.molecule
    )
    print(xml_string)
    assert xml_string == molecule.topology_xml_string
