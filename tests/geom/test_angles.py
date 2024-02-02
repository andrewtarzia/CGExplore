import numpy as np

from .case_data import CaseData


def test_angles(molecule: CaseData) -> None:
    """Test :meth:`.GeomMeasure.calculate_angles`.

    Parameters:

        molecule:
            The molecule with angles.

    Returns:
        None : :class:`NoneType`

    """
    angles = molecule.geommeasure.calculate_angles(molecule.molecule)
    print(angles, molecule.angle_dict)
    for key in molecule.angle_dict:
        print(key)
        assert key in angles
        for angle, test in zip(
            sorted(molecule.angle_dict[key]), sorted(angles[key]), strict=True
        ):
            assert np.isclose(angle, test, atol=1e-3, rtol=0)
