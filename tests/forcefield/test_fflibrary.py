def test_fflibrary(molecule):
    """
    Test methods toward :meth:`.ForceFieldLibrary`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """
    if molecule.num_forcefields > 0:
        print(molecule.force_field_library)
        assert molecule.num_forcefields == len(molecule.force_fields)
        assert str(molecule.force_field_library) == molecule.library_string
