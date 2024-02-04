import pathlib
import shutil

import pytest
from cgexplore.shape import ShapeMeasure

from .case_data import CaseData


def test_shape(molecule: CaseData) -> None:
    """Test :meth:`.ShapeMeasure.calculate`.

    Parameters:

        molecule:
            The molecule to test.

    Returns:
        None : :class:`NoneType`

    """
    # Search for shape executable, if it exists, run the test.
    expected_shape_path = pathlib.Path.home() / (
        "software/shape_2.1_linux_64/SHAPE_2.1_linux_64/shape_2.1_linux64"
    )

    if expected_shape_path.exists():
        # Do test.
        output_dir = pathlib.Path(__file__).resolve().parent / (
            f"{molecule.name}_test_output"
        )
        shape_calc = ShapeMeasure(
            output_dir=output_dir,
            shape_path=expected_shape_path,
        )
        shape_mol = shape_calc.get_shape_molecule_byelement(
            molecule=molecule.molecule,
            element="C",
            expected_points=molecule.expected_points,
        )

        shape_output = shape_calc.calculate(shape_mol)
        print(shape_output, molecule.shape_dict)
        assert shape_output == molecule.shape_dict
        shutil.rmtree(output_dir)

    else:
        pytest.skip(f"shape software not found at {expected_shape_path}")
