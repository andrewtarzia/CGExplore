import numpy as np


def test_radius_gyration(molecule):
    """Test :meth:`.GeomMeasure.calculate_radius_gyration`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    rg = molecule.geommeasure.calculate_radius_gyration(molecule.molecule)
    test = molecule.radius_gyration
    print(rg, test)
    assert np.isclose(rg, test, atol=1e-3, rtol=0)
