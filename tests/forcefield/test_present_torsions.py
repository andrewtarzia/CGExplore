from cgexplore.forcefield import ForcefieldUnitError


def test_present_torsions(molecule):
    """
    Test methods toward :meth:`.ForceField.get_torsion_string`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    try:
        for i, ff in enumerate(molecule.force_fields):
            string = ff.get_torsion_string().split("\n")
            print(string)
            assert string[0] == " <PeriodicTorsionForce>"
            assert string[-3] == " </PeriodicTorsionForce>"
            if len(string) > 4:
                # There are some actual torsions. Test them.
                assert "Proper" in string[1]
                # There are some actual measures here. Test them.
                measures = string[1:-3]
                print(measures)
                for j, measure in enumerate(measures):
                    assert measure == molecule.present_torsions[i][j]

            else:
                assert len(molecule.present_torsions[i]) == 0
    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
