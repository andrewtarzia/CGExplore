import pathlib

from cgexplore.errors import ForcefieldUnitError


def test_present_custom_angles(molecule):
    """
    Test methods toward :meth:`.ForceField.yield_custom_angles`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    try:
        force_fields = tuple(
            molecule.force_field_library.yield_forcefields(
                prefix="testff",
                output_path=pathlib.Path(),
            )
        )
        for i, ff in enumerate(force_fields):
            found_angles = list(ff.yield_custom_angles(molecule.molecule))
            print(found_angles)
            print(molecule.present_custom_angles[i])
            assert len(found_angles) == len(molecule.present_custom_angles[i])
            for angle, test in zip(
                found_angles, molecule.present_custom_angles[i]
            ):
                print("a", angle)
                print("t", test)
                assert angle == test

    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
    except IndexError:
        assert molecule.num_forcefields == 0
