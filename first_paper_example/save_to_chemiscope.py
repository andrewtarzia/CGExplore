# Distributed under the terms of the MIT License.

"""Script to test chemiscope writing."""

import itertools as it
import json
import logging

import atomlite
import cgexplore
from analysis import stoich_map, topology_labels
from env_set import outputdata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def extract_property(path: list[str], properties: dict) -> atomlite.Json:
    """Extract property from nested dict."""
    if len(path) == 1:
        value = properties[path[0]]
    elif len(path) == 2:  # noqa: PLR2004
        value = properties[path[0]][path[1]]
    elif len(path) == 3:  # noqa: PLR2004
        value = properties[path[0]][path[1]][path[2]]
    elif len(path) == 4:  # noqa: PLR2004
        value = properties[path[0]][path[1]][path[2]][path[3]]
    else:
        msg = f"{path} is too deep ({len(path)})."
        raise RuntimeError(msg)
    return value


def axes_show() -> tuple[dict, dict, dict]:
    """Set 2D plots as defult of angle maps."""
    x_setter = {"property": "s_angle"}
    y_setter = {"property": "l_angle"}
    z_setter = {"property": ""}

    return x_setter, y_setter, z_setter


def main() -> None:
    """Run script."""
    data_output = outputdata()

    study_map = {
        "2p3": ("2P3", "4P6", "4P62", "6P9", "8P12"),
        "2p4": ("2P4", "3P6", "4P8", "4P82", "6P12", "8P16", "12P24"),
        "3p4": ("6P8",),
    }

    for tstr, torsion in it.product(
        topology_labels(short="P"), ("ton", "toff")
    ):
        study_pre = next(i for i in study_map if tstr in study_map[i])

        if torsion == "ton" and study_pre == "3p4":
            continue
        study = f"{study_pre}_{torsion}"

        chemiscope_json = data_output / f"cs_{study}_{tstr}.json"

        database = cgexplore.utilities.AtomliteDatabase(
            db_file=data_output / f"first_{study_pre}.db"
        )

        x_setter, y_setter, z_setter = axes_show()

        json_data = {
            "meta": {
                "name": f"CGGeom: {study_pre}|{tstr}|{torsion}",
                "description": (
                    f"Minimal models in {tstr} topology with {torsion} "
                    "torsion state"
                ),
                "authors": ["Andrew Tarzia"],
                "references": [
                    "'Systematic exploration of accessible topologies of "
                    "cage molecules via minimalistic models, Chem. Sci, "
                    "DOI: 10.1039/D3SC03991A'",
                ],
            },
            "structures": [],
            "properties": {},
            "shapes": {},
            "settings": {
                "target": "structure",
                "map": {
                    "x": x_setter,
                    "y": y_setter,
                    "z": z_setter,
                    "color": {"property": "E_b", "min": 0, "max": 1.0},
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

        properties_to_get = {
            "E_b": {
                "path": ["fin_energy_kjmol"],
                "description": "energy per building block",
                "unit": "kjmol-1",
            },
            "l_angle": {
                "path": ["forcefield_dict", "clangle"],
                "description": "large input angle of the two options",
                "unit": "degrees",
            },
            "s_angle": {
                "path": ["forcefield_dict", "c3angle"]
                if tstr in ("6P8",)
                else ["forcefield_dict", "c2angle"],
                "description": "small input angle of the two options",
                "unit": "degrees",
            },
            "pore_radius": {
                "path": ["opt_pore_data", "min_distance"],
                "description": "pore radius of the cage",
                "unit": "AA",
            },
        }

        all_shapes = {}
        count_added = 0
        for entry in database.get_entries():
            properties = entry.properties
            entry_tstr = entry.key.split("_")[0]
            entry_torsion = properties["forcefield_dict"]["torsions"]

            if entry_torsion != torsion:
                continue

            if entry_tstr != tstr:
                continue

            for prop in properties_to_get:
                value = extract_property(
                    path=properties_to_get[prop]["path"],
                    properties=properties,
                )

                if prop == "E_b/kjmol-1":
                    value = value / stoich_map(tstr)

                if prop not in json_data["properties"]:
                    json_data["properties"][prop] = {
                        "target": "structure",
                        "values": [],
                        "units": properties_to_get[prop]["unit"],
                        "description": properties_to_get[prop]["description"],
                    }
                json_data["properties"][prop]["values"].append(value)

            molecule = database.get_molecule(entry.key)

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
            json_data["structures"].append(struct_dict)

            # Topology to shape.
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
                if bname not in all_shapes:
                    all_shapes[bname] = {
                        "kind": "cylinder",
                        "parameters": {
                            "global": {"radius": 1},
                            "structure": [
                                {"vector": [0, 0, 0], "position": [0, 0, 0]}
                                for i in range(count_added)
                            ],
                        },
                    }

                all_shapes[bname]["parameters"]["structure"].append(
                    bond_vector
                )

            count_added += 1

            json_data["shapes"] = all_shapes

            shape_string = ",".join(all_shapes.keys())

            # json_data["settings"]["structure"][0]["shape"] = shape_string
            break
        with chemiscope_json.open("w") as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
