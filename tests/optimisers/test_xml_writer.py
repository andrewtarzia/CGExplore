import pathlib
from cgexplore.openmm_optimizer import CGOMMOptimizer


def test_xml_writer(molecule):
    """
    Test methods toward :meth:`.CGOMMOptimizer._get_xml_string`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    optimizer = CGOMMOptimizer(
        fileprefix="",
        output_dir=pathlib.Path(),
        force_field=molecule.force_field,
    )
    xml_string = optimizer._get_xml_string(molecule.molecule)
    print(xml_string)
    assert xml_string == molecule.xml_string
