import logging
import pathlib
import sqlite3
from collections import abc

import atomlite
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class AtomliteDatabase:
    """Holds an atomlite database with some useful methods."""

    def __init__(self, db_file: pathlib.Path) -> None:
        self._db_file = db_file
        self._db = atomlite.Database(db_file)

    def get_num_entries(self) -> int:
        """
        Get the number of molecular entries in the database.
        """
        return self._db.num_entries()

    def add_molecule(self, molecule: stk.Molecule, key: str) -> None:
        entry = atomlite.Entry.from_rdkit(
            key=key,
            molecule=molecule.to_rdkit_mol(),
        )
        try:
            self._db.add_entries(entry)
        except sqlite3.IntegrityError:
            self._db.update_entries(entry)

    def get_entries(self) -> abc.Iterator[atomlite.Entry]:
        return self._db.get_entries()

    def get_entry(self, key: str) -> atomlite.Entry:
        if not self._db.has_entry(key):
            raise RuntimeError(f"{key} not in database")
        return self._db.get_entry(key)  # type: ignore[return-value]

    def get_molecule(self, key: str) -> stk.Molecule:
        rdkit_molecule = atomlite.json_to_rdkit(self.get_entry(key).molecule)
        return stk.BuildingBlock.init_from_rdkit_mol(rdkit_molecule)

    def add_properties(
        self,
        key: str,
        property_dict: dict[str, atomlite.Json],
    ) -> None:
        self._db.update_properties(
            atomlite.PropertyEntry(key=key, properties=property_dict)
        )

    def get_property(
        self,
        key: str,
        property_key: str,
        property_type: type,
    ) -> atomlite.Json:
        if property_type is bool:
            value = self._db.get_bool_property(  # type: ignore[assignment]
                key=key,
                path=f"$.{property_key}",
            )
        elif property_type is float:
            value = self._db.get_float_property(  # type: ignore[assignment]
                key=key,
                path=f"$.{property_key}",
            )
        elif property_type is str:
            value = self._db.get_str_property(  # type: ignore[assignment]
                key=key,
                path=f"$.{property_key}",
            )
        elif property_type is int:
            value = self._db.get_int_property(  # type: ignore[assignment]
                key=key,
                path=f"$.{property_key}",
            )
        elif property_type is dict:
            value = self.get_entry(key).properties[  # type: ignore[assignment]
                property_key
            ]
        else:
            msg = f"{property_key} has unexpected type"
            raise RuntimeError(msg)

        if value is None:
            msg = f"{property_key} has no value"
            raise RuntimeError(msg)

        return value  # type: ignore[return-value]

    def has_molecule(self, key: str) -> bool:
        if self._db.has_entry(key):
            return True
        else:
            return False

    def keep_if(
        self,
        column: str,
        value: str | int | float,
    ) -> abc.Iterator[atomlite.Entry]:
        for entry in self.get_entries():
            if entry.properties[column] == value:
                yield entry