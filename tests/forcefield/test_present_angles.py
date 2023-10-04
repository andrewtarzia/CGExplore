import os
import pathlib

from cgexplore.errors import ForcefieldUnitError

from .utilities import is_equivalent_atom


def test_present_angles(molecule):
    """
    Test methods toward :meth:`.ForceField._assign_angle_terms`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    try:
        force_fields = tuple(
            molecule.force_field_library.yield_forcefields(
                output_path=pathlib.Path()
            )
        )
        for i, ff in enumerate(force_fields):
            assigned_system = ff.assign_terms(
                molecule=molecule.molecule,
                output_dir=pathlib.Path(
                    os.path.dirname(os.path.realpath(__file__))
                ),
                name=molecule.name,
            )

            present_terms = assigned_system.force_field_terms["angle"]
            print(present_terms)
            assert len(present_terms) == len(molecule.present_angles[i])
            for test, present in zip(
                present_terms, molecule.present_angles[i]
            ):
                assert test.atom_names == present.atom_names
                if present.atoms is None:
                    assert test.atoms is None
                else:
                    for a1, a2 in zip(test.atoms, present.atoms):
                        is_equivalent_atom(a1, a2)
                assert test.atom_ids == present.atom_ids
                assert test.angle == present.angle
                assert test.angle_k == present.angle_k
                assert test.force == present.force

    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
