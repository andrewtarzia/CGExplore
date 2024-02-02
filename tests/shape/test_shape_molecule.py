import stk
from cgexplore.shape import ShapeMeasure

from .case_data import CaseData


def test_shape(molecule: CaseData) -> None:
    """Test :meth:`.ShapeMeasure.get_shape_molecule_byelement`.

    Parameters:

        molecule:
            The molecule to test.

    Returns:
        None : :class:`NoneType`

    """
    shape_calc = ShapeMeasure(
        output_dir="fake_output",
        shape_path="fake_output",
    )
    shape_mol = shape_calc.get_shape_molecule_byelement(
        molecule=molecule.molecule,
        element="C",
        expected_points=molecule.expected_points,
    )

    print(stk.Smiles().get_key(shape_mol))
    assert stk.Smiles().get_key(shape_mol) == molecule.shape_string
