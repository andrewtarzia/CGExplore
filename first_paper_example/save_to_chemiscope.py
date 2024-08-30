# Distributed under the terms of the MIT License.

"""Script to test chemiscope writing."""

import json
import logging

import atomlite
import cgexplore
from analysis import stoich_map
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


def main() -> None:
    """Run script."""
    data_output = outputdata()

    for study in ("2p3", "2p4", "3p4"):
        database = cgexplore.utilities.AtomliteDatabase(
            db_file=data_output / f"first_{study}.db"
        )
        chemiscope_json = data_output / f"{study}.json"

        json_data = {
            "meta": {
                "name": f"CGGeom: {study}",
                "description": f"Minimal models from {study}",
                "authors": ["Andrew Tarzia"],
                "references": [
                    "'Systematic exploration of accessible topologies of '",
                    "'cage molecules via minimalistic models, Chem. Sci, '",
                    "'DOI: 10.1039/D3SC03991A'",
                ],
            },
            "structures": [],
            "properties": {},
            "settings": {
                "structure": [
                    {
                        "bonds": True,
                        "spaceFilling": False,
                        "atomLabels": False,
                        "unitCell": False,
                        "rotation": False,
                        "supercell": {"0": 2, "1": 2, "2": 2},
                        "keepOrientation": False,
                    }
                ]
            },
        }

        properties_to_get = {
            "tstr": {
                "path": None,
                "description": "name of topology graph",
                "unit": "none",
            },
            "torsion": {
                "path": ["forcefield_dict", "torsions"],
                "description": "ton if torsions are restricted, else toff",
                "unit": "none",
            },
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
                "path": ["forcefield_dict", "c2angle"]
                if study in ("2p3", "2p4")
                else ["forcefield_dict", "c3angle"],
                "description": "small input angle of the two options",
                "unit": "degrees",
            },
            "pore_radius": {
                "path": ["opt_pore_data", "min_distance"],
                "description": "pore radius of the cage",
                "unit": "AA",
            },
        }

        for entry in database.get_entries():
            properties = entry.properties
            tstr = entry.key.split("_")[0]
            for prop in properties_to_get:
                if prop == "tstr":
                    value = tstr

                else:
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

            break

        with chemiscope_json.open("w") as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
