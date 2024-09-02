# Distributed under the terms of the MIT License.

"""Script to test chemiscope writing."""

import itertools as it
import logging
from collections import abc

from analysis import stoich_map, topology_labels
from env_set import outputdata

import cgexplore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def divider(tstr: str) -> abc.Callable:
    """Encapsulate function with new variable."""

    def divide_by_stoich(value: float) -> float:
        """Divide energy by stoichiometry."""
        return value / stoich_map(tstr)

    return divide_by_stoich


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

        database = cgexplore.utilities.AtomliteDatabase(
            db_file=data_output / f"first_{study_pre}.db"
        )

        chemiscope_writer = cgexplore.utilities.ChemiscopeInterface(
            json_file=data_output / f"cs_{study}_{tstr}.json",
            x_axis_name="s_angle",
            y_axis_name="l_angle",
            z_axis_name="",
            meta_dict={
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
            properties_to_get={
                "E_b": {
                    "path": ["fin_energy_kjmol"],
                    "description": "energy per building block",
                    "unit": "kjmol-1",
                    "function": divider(tstr),
                },
                "l_angle": {
                    "path": ["forcefield_dict", "clangle"],
                    "description": "large input angle of the two options",
                    "unit": "degrees",
                    "function": None,
                },
                "s_angle": {
                    "path": ["forcefield_dict", "c3angle"]
                    if tstr in ("6P8",)
                    else ["forcefield_dict", "c2angle"],
                    "description": "small input angle of the two options",
                    "unit": "degrees",
                    "function": None,
                },
                "pore_radius": {
                    "path": ["opt_pore_data", "min_distance"],
                    "description": "pore radius of the cage",
                    "unit": "AA",
                    "function": None,
                },
            },
        )

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

            chemiscope_writer.append_property(property_dict=properties)
            chemiscope_writer.append_molecule(
                molecule=database.get_molecule(entry.key)
            )
            continue
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
        chemiscope_writer.write_json()


if __name__ == "__main__":
    main()
