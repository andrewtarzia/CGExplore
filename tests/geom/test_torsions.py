def test_torsions(molecule):
    """
    Test :meth:`.GeomMeasure.calculate_torsions`.

    Parameters:

        molecule:
            The molecule with torsions.

    Returns:

        None : :class:`NoneType`

    """

    torsions = molecule.geommeasure.calculate_torsions(
        molecule=molecule.molecule,
        absolute=False,
        path_length=4,
    )

    print(torsions, molecule.torsion_dict)
    if len(molecule.torsion_dict) == 0:
        assert torsions == []
