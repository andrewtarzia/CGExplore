from cgexplore.optimizer import CGOptimizer


def test_present_torsions(molecule):
    """
    Test methods toward :meth:`.Optimizer._yield_custom_torsions`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    optimizer = CGOptimizer(
        force_field=molecule.force_field,
    )

    found_torsions = list(
        optimizer._force_field.yield_custom_torsions(molecule.molecule)
    )
    print(found_torsions)
    assert len(found_torsions) == len(molecule.present_torsions)
    for torsion, test in zip(found_torsions, molecule.present_torsions):
        print(torsion, test)
        assert torsion == test
