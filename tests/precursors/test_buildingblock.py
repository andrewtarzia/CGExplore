import numpy as np
import stk

from .case_data import CaseData


def test_building_block(precursor: CaseData) -> None:
    """Test :meth:`.Precursor.get_building_block`.

    Parameters:

        precursor:
            The precursor.

    """
    test = stk.Smiles().get_key(precursor.precursor.get_building_block())

    assert precursor.smiles == test

    test_posmat: np.ndarray = (
        precursor.precursor.get_building_block().get_position_matrix()
    )

    assert np.allclose(a=precursor.position_matrix, b=test_posmat, atol=1e-6)
