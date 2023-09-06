from cgexplore.forcefield import ForcefieldUnitError


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
        for i, ff in enumerate(molecule.force_fields):
            xml_string = ff.get_xml_string()
            print(xml_string)
            assert xml_string == molecule.xml_strings[i]
    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
