import numpy as np
import stk

from .case_data import CaseData


def test_building_block(precursor: CaseData) -> None:
    """Test :meth:`.Precursor.get_building_block`.

    Parameters:

        precursor:
            The precursor.

    Returns:
        None : :class:`NoneType`

    """
    print(precursor.precursor.get_building_block())
    test = stk.Smiles().get_key(precursor.precursor.get_building_block())
    print(test)
    assert precursor.smiles == test

    test = precursor.precursor.get_building_block().get_position_matrix()
    print(test)
    assert np.allclose(
        a=precursor.position_matrix,
        b=test,
        atol=1e-6,
    )
