import pathlib
import shutil

import atomlite
import pytest
import stk
from cgexplore.databases import AtomliteDatabase

from .case_data import CaseData


def test_databasing(molecule: CaseData) -> None:
    """Test :class:`.AtomliteDatabase`."""
    path = pathlib.Path(__file__).resolve().parent / "test.db"

    if path.exists():
        shutil.rmtree(path)

    database = AtomliteDatabase(path)

    assert database.get_num_entries() == 0
    for mol, prop in zip(
        molecule.molecules,
        molecule.property_dicts,
        strict=True,
    ):
        print(mol, prop)
        key = stk.Smiles().get_key(mol)
        database.add_molecule(molecule=mol, key=key)

        database.add_properties(key, prop)

        entry = database.get_entry(key)
        print(entry.molecule)
        assert entry.molecule == atomlite.json_from_rdkit(mol.to_rdkit_mol())
        print(stk.Smiles().get_key(database.get_molecule(key)))
        assert stk.Smiles().get_key(database.get_molecule(key)) == key

        print(entry.properties)
        assert (
            database.get_property(key, property_key="1", property_type=int)
            == prop["1"]
        )
        if "2" in prop:
            assert (
                database.get_property(
                    key=key, property_key="2", property_type=dict
                )
                == prop["2"]
            )

        with pytest.raises(TypeError):
            print(database.get_property(key, "1", property_type=str))

    print(database.get_num_entries())
    assert database.get_num_entries() == molecule.expected_count

    path.unlink()
