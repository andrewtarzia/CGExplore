import numpy as np

from .case_data import CaseData


def test_torsions(molecule: CaseData) -> None:
    """Test :meth:`.GeomMeasure.calculate_torsions`.

    Parameters:

        molecule:
            The molecule with torsions.

    Returns:
        None : :class:`NoneType`

    """
    torsions = molecule.geommeasure.calculate_torsions(
        molecule=molecule.molecule,
        absolute=False,
    )

    print(torsions, molecule.torsion_dict)
    if len(molecule.torsion_dict) == 0:
        assert torsions == {}
        assert len(molecule.torsion_dict) == 0

    for key in molecule.torsion_dict:
        print(key)
        assert key in torsions
        for torsion, test in zip(
            sorted(molecule.torsion_dict[key]),
            sorted(torsions[key]),
            strict=True,
        ):
            assert np.isclose(torsion, test, atol=1e-3, rtol=0)
