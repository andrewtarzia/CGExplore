import os
import pathlib

from cgexplore.errors import ForcefieldUnitError


def test_present_torsions(molecule):
    """
    Test methods toward :meth:`.ForceField._assign_torsion_terms`.

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

            present_terms = assigned_system.force_field_terms["torsion"]
            print(present_terms)
            assert len(present_terms) == len(molecule.present_torsions[i])
            for test, present in zip(
                present_terms, molecule.present_torsions[i]
            ):
                assert test.atom_names == present.atom_names
                assert test.atom_ids == present.atom_ids
                assert test.phi0 == present.phi0
                assert test.torsion_k == present.torsion_k
                assert test.torsion_n == present.torsion_n
                assert test.force == present.force

    except ForcefieldUnitError:
        assert molecule.num_forcefields == 0
