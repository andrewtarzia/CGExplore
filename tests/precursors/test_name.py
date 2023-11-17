def test_name(precursor):
    """Test :meth:`.Precursor.get_name`.

    Parameters:

        precursor:
            The precursor.

    Returns:
        None : :class:`NoneType`

    """
    print(precursor.precursor.get_name(), precursor.precursor_name)
    assert precursor.precursor.get_name() == precursor.precursor_name
