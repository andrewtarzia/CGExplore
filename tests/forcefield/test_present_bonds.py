import pathlib

from cgexplore.utilities import ForceFieldUnitError

from .case_data import CaseData
from .utilities import is_equivalent_atom


def test_present_bonds(molecule: CaseData) -> None:
    """Test methods toward :meth:`.ForceField._assign_bond_terms`.

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

            present_terms = assigned_system.forcefield_terms["bond"]
            print(present_terms)
            assert len(present_terms) == len(molecule.present_bonds[i])
            for test, present in zip(
                present_terms, molecule.present_bonds[i], strict=True
            ):
                assert test.atom_names == present.atom_names
                if present.atoms is None:
                    assert test.atoms is None
                else:
                    for a1, a2 in zip(test.atoms, present.atoms, strict=True):
                        is_equivalent_atom(a1, a2)
                assert test.atom_ids == present.atom_ids
                assert test.bond_r == present.bond_r
                assert test.bond_k == present.bond_k
                assert test.force == present.force

    except ForceFieldUnitError:
        assert molecule.num_forcefields == 0
