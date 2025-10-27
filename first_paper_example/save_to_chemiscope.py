# Distributed under the terms of the MIT License.

"""Script to test chemiscope writing."""

import itertools as it
import logging
from collections import abc

from analysis import topology_labels
from env_set import outputdata

import cgexplore as cgx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def divider(tstr: str) -> abc.Callable:
    """Encapsulate function with new variable."""

    def divide_by_stoich(value: float) -> float:
        """Divide energy by stoichiometry."""
        return value / cgx.topologies.stoich_map(tstr)

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

        database = cgx.utilities.AtomliteDatabase(
            db_file=data_output / f"first_{study_pre}.db"
        )

        properties_to_get = {
            "E_b / kjmol-1": {
                "path": ["fin_energy_kjmol"],
                "function": divider(tstr),
            },
            "l_angle / deg": {
                "path": ["forcefield_dict", "clangle"],
                "function": None,
            },
            "s_angle / deg": {
                "path": ["forcefield_dict", "c3angle"]
                if tstr in ("6P8",)
                else ["forcefield_dict", "c2angle"],
                "function": None,
            },
            "pore_radius / AA": {
                "path": ["opt_pore_data", "min_distance"],
                "function": None,
            },
        }

        structures = []
        properties = {}
        for entry in database.get_entries():
            entry_tstr = entry.key.split("_")[0]
            entry_torsion = entry.properties["forcefield_dict"]["torsions"]

            if entry_torsion != torsion:
                continue

            if entry_tstr != tstr:
                continue

            structures.append(database.get_molecule(entry.key))

            for prop in properties_to_get:
                value = cgx.utilities.extract_property(
                    path=properties_to_get[prop]["path"],
                    properties=entry.properties,
                )

                if properties_to_get[prop]["function"] is not None:
                    value = properties_to_get[prop]["function"](value)

                if prop not in properties:
                    properties[prop] = []
                properties[prop].append(value)

        logger.info(
            "for %s|%s|%s, structures: %s, properties: %s",
            study,
            tstr,
            torsion,
            len(structures),
            len(properties),
        )
        cgx.utilities.write_chemiscope_json(
            json_file=data_output / f"cs_{study}_{tstr}.json.gz",
            structures=structures,
            properties=properties,  # type:ignore[arg-type]
            bonds_as_shapes=True,
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
            x_axis_dict={"property": "s_angle / deg"},
            y_axis_dict={"property": "l_angle / deg"},
            z_axis_dict={"property": ""},
            color_dict={"property": "E_b / kjmol-1", "min": 0, "max": 1.0},
            bond_hex_colour="#919294",
        )


if __name__ == "__main__":
    main()
