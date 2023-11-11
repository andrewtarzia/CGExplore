import os

import atomlite
import pytest
import stk
from cgexplore.databases import AtomliteDatabase


def test_databasing(molecule):
    """
    Test :class:`.AtomliteDatabase`.
    """

    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "test.db",
    )
    print(path)
    database = AtomliteDatabase(path)
    print(database)

    assert database.get_num_entries() == 0
    for mol, prop in zip(molecule.molecules, molecule.property_dicts):
        print(mol, prop)
        key = stk.Smiles().get_key(mol)
        database.add_molecule(molecule=mol, key=key)

        database.add_properties(key, prop)

        entry = database.get_entry(key)
        print(entry.molecule)
        assert entry.molecule == atomlite.json_from_rdkit(mol.to_rdkit_mol())
        print(stk.Smiles().get_key(database.get_molecule(key)))
        assert stk.Smiles().get_key(database.get_molecule(key)) == key

        print(database.get_property(key, "1", ensure_type=int))
        assert database.get_property(key, "1", ensure_type=int) == prop["1"]
        if "2" in prop:
            assert (
                database.get_property(key, "2", ensure_type=int) == prop["2"]
            )

        with pytest.raises(RuntimeError):
            print(database.get_property(key, "1", ensure_type=str))

    print(database.get_num_entries())
    assert database.get_num_entries() == molecule.expected_count

    os.remove(path)
