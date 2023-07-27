import numpy as np


def test_maxdiameter(molecule):
    """
    Test :meth:`.GeomMeasure.calculate_max_diameter`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    maxdiam = molecule.geommeasure.calculate_max_diameter(molecule.molecule)
    test = molecule.max_diam
    print(maxdiam, test)
    assert np.isclose(maxdiam, test, atol=1e-3, rtol=0)
