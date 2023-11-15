import os
import pathlib


def test_topology_xml_writer(molecule):
    """Test methods toward :meth:`.AssignedSystem._get_topology_xml_string`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    assigned_system = molecule.force_field.assign_terms(
        molecule=molecule.molecule,
        output_dir=pathlib.Path(os.path.dirname(os.path.realpath(__file__))),
        name=molecule.name,
    )
    xml_string = assigned_system._get_topology_xml_string(molecule.molecule)
    print(xml_string)
    assert xml_string == molecule.topology_xml_string
