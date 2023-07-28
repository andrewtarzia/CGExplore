import numpy as np


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
    for key in molecule.length_dict:
        print(key)
        assert key in lengths
        for length, test in zip(
            sorted(molecule.length_dict[key]),
            sorted(lengths[key]),
        ):
            assert np.isclose(length, test, atol=1e-3, rtol=0)
