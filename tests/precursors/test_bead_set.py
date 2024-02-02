from .case_data import CaseData


def test_bead_set(precursor: CaseData) -> None:
    """Test :meth:`.Precursor.get_bead_set`.

    Parameters:

        precursor:
            The precursor.

    Returns:
        None : :class:`NoneType`

    """
    bead_set = precursor.precursor.get_bead_set()
    print(bead_set)
    assert bead_set == precursor.bead_set
