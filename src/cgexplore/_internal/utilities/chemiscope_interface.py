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
        self.all_shapes: dict[str, dict] = {}

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

    def append_bonds_as_shapes(self, molecule: stk.Molecule) -> None:
        """Append bonds as shapes to json data."""
        pos_mat = molecule.get_position_matrix()

        # Get bonds and their directions first.
        bond_vectors = [
            {
                "vector": (
                    pos_mat[bond.get_atom2().get_id()]
                    - pos_mat[bond.get_atom1().get_id()]
                ).tolist(),
                "position": (pos_mat[bond.get_atom1().get_id()]).tolist(),
            }
            for bond in molecule.get_bonds()
        ]

        # Now add to shape list. Each time, adding a new shape per
        # structure if needed.
        for i, bond_vector in enumerate(bond_vectors):
            bname = f"bond_{i}"
            if bname not in self.all_shapes:
                if bname == "bond_0":
                    self.all_shapes[bname] = {
                        "kind": "cylinder",
                        "parameters": {
                            "global": {"radius": 0.12, "color": "#d9d9d9"},
                            "structure": [],
                        },
                    }
                else:
                    self.all_shapes[bname] = {
                        "kind": "cylinder",
                        "parameters": {
                            "global": {"radius": 0.12, "color": "#d9d9d9"},
                            # Add zero placements for previously non-existant
                            # bond shapes up to the length of bond_0 -1,
                            # because that should already be at the current
                            # length that the new one should be.
                            "structure": [
                                {"vector": [0, 0, 0], "position": [0, 0, 0]}
                                for i in range(
                                    len(
                                        self.all_shapes["bond_0"][
                                            "parameters"
                                        ]["structure"]
                                    )
                                    - 1
                                )
                            ],
                        },
                    }

            self.all_shapes[bname]["parameters"]["structure"].append(
                bond_vector
            )

            self.json_data["shapes"] = self.all_shapes

            shape_string = ",".join(self.all_shapes.keys())

            self.json_data["settings"]["structure"][0]["shape"] = shape_string  # type: ignore[index]

    def write_json(self) -> None:
        """Write the chemiscope json."""
        with self.json_file.open("w") as f:
            json.dump(self.json_data, f, indent=4)
