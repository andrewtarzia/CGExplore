import pathlib

import cgexplore
import numpy as np
import stk

from .case_data import CaseData


def test_openmmoptimizer(molecule: CaseData) -> None:
    """Test :class:`.CGOMMOptimizer`.

    Parameters:

        molecule:
            The molecule to optimise.

    """
    output_dir = pathlib.Path(__file__).resolve().parent / "output"

    # Then running optimisation.
    assigned_system = molecule.forcefield.assign_terms(
        molecule=molecule.molecule,
        name=f"test_{molecule.name}",
        output_dir=output_dir,
    )
    conformer = cgexplore.utilities.run_optimisation(
        assigned_system=assigned_system,
        name=f"test_{molecule.name}",
        file_suffix=f"test_{molecule.name}",
        output_dir=output_dir,
        platform=None,
    )

    # Compare to known output.
    energy_decomposition = conformer.energy_decomposition
    print(energy_decomposition)
    assert tuple(sorted(energy_decomposition.keys())) == tuple(
        sorted(molecule.known_decomposition.keys())
    )
    for test in energy_decomposition:
        assert (
            energy_decomposition[test][1]
            == molecule.known_decomposition[test][1]
        )
        assert np.isclose(
            energy_decomposition[test][0],
            molecule.known_decomposition[test][0],
            rtol=0,
            atol=1e-4,
        )

    known_molecule = stk.BuildingBlock.init_from_file(
        output_dir / f"{molecule.name}.mol"
    )
    known_txt = stk.MolWriter().to_string(known_molecule)
    test_txt = stk.MolWriter().to_string(conformer.molecule)
    print(known_txt, test_txt)
    assert known_txt == test_txt

    (
        output_dir / f"test_{molecule.name}_test_{molecule.name}_omm.out"
    ).unlink()
    (output_dir / f"test_{molecule.name}_opt_tests_test_syst.xml").unlink()
