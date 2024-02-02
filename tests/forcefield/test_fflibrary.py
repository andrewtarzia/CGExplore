from .case_data import CaseData


def test_fflibrary(molecule: CaseData) -> None:
    """Test methods toward :meth:`.ForceFieldLibrary`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    if molecule.num_forcefields > 0:
        forcefields = tuple(molecule.forcefield_library.yield_forcefields())
        print(molecule.forcefield_library)
        assert molecule.num_forcefields == len(forcefields)
        assert str(molecule.forcefield_library) == molecule.library_string
