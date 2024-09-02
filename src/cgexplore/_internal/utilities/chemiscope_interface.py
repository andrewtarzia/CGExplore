"""Module for chemiscope usage."""

import json
import logging
import pathlib
from dataclasses import dataclass

import stk

from cgexplore._internal.utilities.utilities import extract_property

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass(kw_only=True)
class ChemiscopeInterface:
    """Interface between stk, cgexplore and chemiscope visualisation."""

    json_file: pathlib.Path
    x_axis_name: str
    y_axis_name: str
    z_axis_name: str
    color_dict: dict[str, str | float]
    meta_dict: dict[str, str | list]
    properties_to_get: dict[str, dict]

    def __post_init__(self) -> None:
        self.json_data = {
            "meta": self.meta_dict,
            "structures": [],
            "properties": {},
            "shapes": {},
            "settings": {
                "target": "structure",
                "map": {
                    "x": {"property": self.x_axis_name},
                    "y": {"property": self.y_axis_name},
                    "z": {"property": self.z_axis_name},
                    "color": self.color_dict,
                    "palette": "plasma",
                },
                "structure": [
                    {
                        "atoms": True,
                        "bonds": False,
                        "playbackDelay": 200,
                        "spaceFilling": False,
                        "atomLabels": False,
                        "unitCell": False,
                        "rotation": False,
                        "supercell": {"0": 2, "1": 2, "2": 2},
                        "keepOrientation": False,
                    }
                ],
            },
        }

    def append_property(self, property_dict: dict) -> None:
        """Append properties in property_dict to json data."""
        for prop in self.properties_to_get:
            value = extract_property(
                path=self.properties_to_get[prop]["path"],
                properties=property_dict,
            )

            if self.properties_to_get[prop]["function"] is not None:
                value = self.properties_to_get[prop]["function"](value)

            if prop not in self.json_data["properties"]:
                self.json_data["properties"][prop] = {  # type: ignore[index]
                    "target": "structure",
                    "values": [],
                    "units": self.properties_to_get[prop]["unit"],
                    "description": self.properties_to_get[prop]["description"],
                }
            self.json_data["properties"][prop]["values"].append(value)  # type: ignore[index]

    def append_molecule(self, molecule: stk.Molecule) -> None:
        """Append molecule to json data."""
        pos_mat = molecule.get_position_matrix()
        names = []
        xs = []
        ys = []
        zs = []
        for atom in molecule.get_atoms():
            names.append(atom.__class__.__name__)
            id_ = atom.get_id()
            xs.append(pos_mat[id_][0])
            ys.append(pos_mat[id_][1])
            zs.append(pos_mat[id_][2])

        struct_dict = {
            "size": molecule.get_num_atoms(),
            "names": names,
            "x": xs,
            "y": ys,
            "z": zs,
        }
        self.json_data["structures"].append(struct_dict)  # type: ignore[attr-defined]

    def write_json(self) -> None:
        """Write the chemiscope json."""
        with self.json_file.open("w") as f:
            json.dump(self.json_data, f, indent=4)
