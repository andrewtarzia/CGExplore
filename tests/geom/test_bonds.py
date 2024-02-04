import numpy as np

from .case_data import CaseData


def test_bonds(molecule: CaseData) -> None:
    """Test :meth:`.GeomMeasure.calculate_bonds`.

    Parameters:

        molecule:
            The molecule with bonds.

    Returns:
        None : :class:`NoneType`

    """
    lengths = molecule.geommeasure.calculate_bonds(molecule.molecule)
    print(lengths, molecule.length_dict)
    for key in molecule.length_dict:
        print(key)
        assert key in lengths
        for length, test in zip(
            sorted(molecule.length_dict[key]),
            sorted(lengths[key]),
            strict=True,
        ):
            assert np.isclose(length, test, atol=1e-3, rtol=0)
