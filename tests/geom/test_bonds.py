def test_bonds(molecule):
    """
    Test :meth:`.GeomMeasure.calculate_bonds`.

    Parameters
    ----------
        molecule:
            The molecule with bonds.

    Returns:

        None : :class:`NoneType`

    """

    lengths = molecule.geommeasure.calculate_bonds(molecule.molecule)
    print(lengths, molecule.length_dict)
    assert lengths == molecule.length_dict
