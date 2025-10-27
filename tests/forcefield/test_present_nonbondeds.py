import pathlib

from cgexplore.utilities import ForceFieldUnitError

from .case_data import CaseData


def test_present_nonbondeds(molecule: CaseData) -> None:
    """Test methods toward :meth:`.ForceField._assign_nonbonded_terms`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    try:
        forcefields = tuple(molecule.forcefield_library.yield_forcefields())
        for i, ff in enumerate(forcefields):
            assigned_system = ff.assign_terms(
                molecule=molecule.molecule,
                output_dir=pathlib.Path(__file__).resolve().parent,
                name=molecule.name,
            )

            present_terms = assigned_system.forcefield_terms["nonbonded"]
            # Ignore a bunch of typing because of different term types.
            assert len(present_terms) == len(molecule.present_nonbondeds[i])
            for test, present in zip(
                present_terms, molecule.present_nonbondeds[i], strict=True
            ):
                assert test.atom_id == present.atom_id  # type:ignore[union-attr]
                assert test.bead_class == present.bead_class  # type:ignore[union-attr]
                assert test.bead_element == present.bead_element  # type:ignore[union-attr]
                assert test.sigma == present.sigma  # type:ignore[union-attr]
                assert test.epsilon == present.epsilon  # type:ignore[union-attr]
                assert test.force == present.force  # type:ignore[union-attr]

    except ForceFieldUnitError:
        assert molecule.num_forcefields == 0
