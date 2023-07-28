import numpy as np


def test_present_torsions(molecule):
    """
    Test :meth:`XXXX`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    assert False

    angles = molecule.geommeasure.calculate_angles(molecule.molecule)
    print(angles, molecule.angle_dict)
    for key in molecule.angle_dict:
        print(key)
        assert key in angles
        for angle, test in zip(
            sorted(molecule.angle_dict[key]),
            sorted(angles[key]),
        ):
            assert np.isclose(angle, test, atol=1e-3, rtol=0)
