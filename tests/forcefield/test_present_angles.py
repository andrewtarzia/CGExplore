from cgexplore.errors import ForcefieldUnitError
import pathlib


def test_present_angles(molecule):
    """
    Test methods toward :meth:`.ForceField.get_angle_string`.

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
            string = ff.get_angle_string().split("\n")
            print(string)
            assert string[0] == " <HarmonicAngleForce>"
            assert string[-3] == " </HarmonicAngleForce>"
            if len(string) > 4:
                assert "Angle" in string[1]
                # There are some actual measures here. Test them.
                measures = string[1:-3]
                print(measures)
                for j, measure in enumerate(measures):
                    assert measure == molecule.present_angles[i][j]
            else:
                assert len(molecule.present_angles[i]) == 0
    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
