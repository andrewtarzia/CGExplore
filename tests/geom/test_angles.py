def test_angles(molecule):
    """
    Test :meth:`.GeomMeasure.calculate_angles`.

    Parameters:

        molecule:
            The molecule with angles.

    Returns:

        None : :class:`NoneType`

    """

    angles = molecule.geommeasure.calculate_angles(molecule.molecule)
    print(angles, molecule.angle_dict)
    assert angles == molecule.angle_dict
