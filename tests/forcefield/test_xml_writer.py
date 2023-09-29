import pathlib

from cgexplore.errors import ForcefieldUnitError


def test_ff_xml_writer(molecule):
    """
    Test methods toward :meth:`.ForceField.get_xml_string`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    try:
        force_fields = tuple(
            molecule.force_field_library.yield_forcefields(
                output_path=pathlib.Path()
            )
        )
        for i, ff in enumerate(force_fields):
            xml_string = ff.get_xml_string()
            print(xml_string)
            assert xml_string == molecule.xml_strings[i]
    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
