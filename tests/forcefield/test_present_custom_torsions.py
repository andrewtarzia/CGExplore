import pathlib

from cgexplore.errors import ForcefieldUnitError


def test_present_custom_torsions(molecule):
    """
    Test methods toward :meth:`.ForceField.yield_custom_torsions`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    try:
        force_fields = tuple(
            molecule.force_field_library.yield_forcefields(
                prefix="testff", output_path=pathlib.Path()
            )
        )
        for i, ff in enumerate(force_fields):
            found_torsions = list(ff.yield_custom_torsions(molecule.molecule))
            print(found_torsions)
            assert len(found_torsions) == len(
                molecule.present_custom_torsions[i]
            )
            for torsion, test in zip(
                found_torsions, molecule.present_custom_torsions[i]
            ):
                print(torsion, test)
                assert torsion == test
    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
