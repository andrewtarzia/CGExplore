import pathlib


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
        force_fields = tuple(
            molecule.force_field_library.yield_forcefields(
                output_path=pathlib.Path()
            )
        )
        print(molecule.force_field_library)
        assert molecule.num_forcefields == len(force_fields)
        assert str(molecule.force_field_library) == molecule.library_string
