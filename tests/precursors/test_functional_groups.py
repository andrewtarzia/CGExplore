def test_functional_groups(precursor):
    """Test :class:`.Precursor`.

    Parameters:

        precursor:
            The precursor.

    Returns:
        None : :class:`NoneType`

    """
    fgs = list(
        precursor.precursor.get_building_block().get_functional_groups()
    )
    print(fgs)
    assert len(fgs) == precursor.num_fgs
