import pathlib

from cgexplore.errors import ForceFieldUnitError

from .case_data import CaseData


def test_present_torsions(molecule: CaseData) -> None:
    """Test methods toward :meth:`.ForceField._assign_torsion_terms`.

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

            present_terms = assigned_system.forcefield_terms["torsion"]
            print(present_terms)
            assert len(present_terms) == len(molecule.present_torsions[i])
            for test, present in zip(
                present_terms, molecule.present_torsions[i], strict=True
            ):
                assert test.atom_names == present.atom_names
                assert test.atom_ids == present.atom_ids
                assert test.phi0 == present.phi0
                assert test.torsion_k == present.torsion_k
                assert test.torsion_n == present.torsion_n
                assert test.force == present.force

    except ForceFieldUnitError:
        assert molecule.num_forcefields == 0
