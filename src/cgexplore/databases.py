"""Module for databasing, with AtomLite.

Author: Andrew Tarzia
"""

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
        """Initialize database."""
        self._db_file = db_file
        self._db = atomlite.Database(db_file)

    def get_num_entries(self) -> int:
        """Get the number of molecular entries in the database."""
        return self._db.num_entries()

    def add_molecule(self, molecule: stk.Molecule, key: str) -> None:
        """Add molecule to database as entry."""
        entry = atomlite.Entry.from_rdkit(
            key=key,
            molecule=molecule.to_rdkit_mol(),
        )
        try:
            self._db.add_entries(entry)
        except sqlite3.IntegrityError:
            self._db.update_entries(entry)

    def get_entries(self) -> abc.Iterator[atomlite.Entry]:
        """Get all entries."""
        return self._db.get_entries()

    def get_entry(self, key: str) -> atomlite.Entry:
        """Get specific entry."""
        if not self._db.has_entry(key):
            msg = f"{key} not in database"
            raise RuntimeError(msg)
        return self._db.get_entry(key)  # type: ignore[return-value]

    def get_molecule(self, key: str) -> stk.Molecule:
        """Get a molecule."""
        rdkit_molecule = atomlite.json_to_rdkit(self.get_entry(key).molecule)
        return stk.BuildingBlock.init_from_rdkit_mol(rdkit_molecule)

    def add_properties(
        self,
        key: str,
        property_dict: dict[str, atomlite.Json],
    ) -> None:
        """Add properties to an entry by key."""
        self._db.update_properties(
            atomlite.PropertyEntry(key=key, properties=property_dict)
        )

    def get_property(
        self,
        key: str,
        property_key: str,
        property_type: type,
    ) -> atomlite.Json:
        """Get the properties of an entry."""
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
            value = self.get_entry(key).properties[property_key]  # type: ignore[assignment]
        else:
            msg = f"{property_key} has unexpected type"
            raise RuntimeError(msg)

        if value is None:
            msg = f"{property_key} has no value"
            raise RuntimeError(msg)

        return value  # type: ignore[return-value]

    def has_molecule(self, key: str) -> bool:
        """Check if database has a molecule by key."""
        return bool(self._db.has_entry(key))

    def remove_entry(self, key: str) -> None:
        """Remove an entry by key."""
        self._db.remove_entries(keys=key)

    def keep_if(
        self,
        column: str,
        value: str | float,
    ) -> abc.Iterator[atomlite.Entry]:
        """Filter database entries by properties."""
        for entry in self.get_entries():
            if entry.properties[column] == value:
                yield entry
