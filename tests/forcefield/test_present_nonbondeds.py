import os
import pathlib

from cgexplore.errors import ForceFieldUnitError


def test_present_nonbondeds(molecule):
    """Test methods toward :meth:`.ForceField._assign_nonbonded_terms`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    try:
        force_fields = tuple(molecule.force_field_library.yield_forcefields())
        for i, ff in enumerate(force_fields):
            assigned_system = ff.assign_terms(
                molecule=molecule.molecule,
                output_dir=pathlib.Path(
                    os.path.dirname(os.path.realpath(__file__))
                ),
                name=molecule.name,
            )

            present_terms = assigned_system.force_field_terms["nonbonded"]
            print(present_terms)
            assert len(present_terms) == len(molecule.present_nonbondeds[i])
            for test, present in zip(
                present_terms, molecule.present_nonbondeds[i], strict=True
            ):
                assert test.atom_id == present.atom_id
                assert test.bead_class == present.bead_class
                assert test.bead_element == present.bead_element
                assert test.sigma == present.sigma
                assert test.epsilon == present.epsilon
                assert test.force == present.force

    except ForceFieldUnitError:
        assert molecule.num_forcefields == 0
