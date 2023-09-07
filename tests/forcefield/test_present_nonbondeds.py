from cgexplore.errors import ForcefieldUnitError
import pathlib


def test_present_nonbondeds(molecule):
    """
    Test methods toward :meth:`.ForceField.get_nonbonded_string`.

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
            string = ff.get_nonbonded_string().split("\n")
            print(string)
            assert string[0] == (
                ' <CustomNonbondedForce energy="sqrt(epsilon1*epsilon2)'
                '*((sigma1+sigma2)/(2*r))^12" bondCutoff="2">'
            )
            assert string[1] == '  <PerParticleParameter name="sigma"/>'
            assert string[2] == '  <PerParticleParameter name="epsilon"/>'
            assert string[-3] == " </CustomNonbondedForce>"
            if len(string) > 6:
                assert "Atom" in string[3]
                # There are some actual measures here. Test them.
                measures = string[3:-3]
                print(measures)
                for j, measure in enumerate(measures):
                    assert measure == molecule.present_nonbondeds[i][j]
            else:
                assert len(molecule.present_nonbondeds[i]) == 0
    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
