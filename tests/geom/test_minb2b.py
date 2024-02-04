from .case_data import CaseData


def test_minb2b(molecule: CaseData) -> None:
    """Test :meth:`.GeomMeasure.calculate_minb2b`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    minb2b = molecule.geommeasure.calculate_minb2b(molecule.molecule)
    test = min(min(i) for i in molecule.length_dict.values())
    print(minb2b, test)
    assert minb2b == test
