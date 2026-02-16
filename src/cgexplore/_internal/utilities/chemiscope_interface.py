"""Module for chemiscope usage."""

import logging
import pathlib
from collections import abc

import chemiscope
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def convert_stk_bonds_as_shapes(
    structures: abc.Sequence[stk.Molecule],
    bond_color: str,
    bond_radius: float,
) -> dict[str, dict]:
    """Convert connections between atom ids in each structure to shapes.

    :param structures: List of stk Molecule objects.
    :param bond_color: How to color the bonds added.
    :param bond_radius: Radius of bonds to add.
    """
    shape_dict: dict[str, dict] = {}
    max_length = 0
    for molecule in structures:
        bonds_to_add = tuple(
            (bond.get_atom1().get_id(), bond.get_atom2().get_id())
            for bond in molecule.get_bonds()
        )

        for bid, bond_info in enumerate(bonds_to_add):
            bname = f"bond_{bid}"

            # Compute the bond vector.
            position_matrix = molecule.get_position_matrix()
            bond_geometry = {
                "vector": (
                    position_matrix[bond_info[1]]
                    - position_matrix[bond_info[0]]
                ).tolist(),
                "position": (position_matrix[bond_info[0]]).tolist(),
            }

            # Add the bond name to the dictionary to be iterated through.
            if bname not in shape_dict:
                if bname == "bond_0":
                    shape_dict[bname] = {
                        "kind": "cylinder",
                        "parameters": {
                            "global": {
                                "radius": bond_radius,
                                "color": bond_color,
                            },
                            "structure": [],
                        },
                    }

                else:
                    num_to_add = len(
                        shape_dict["bond_0"]["parameters"]["structure"]
                    )
                    shape_dict[bname] = {
                        "kind": "cylinder",
                        "parameters": {
                            "global": {
                                "radius": bond_radius,
                                "color": bond_color,
                            },
                            # Add zero placements for previously non-existant
                            # bond shapes up to the length of bond_0 -1,
                            # because that should already be at the current
                            # length that the new one should be.
                            "structure": [
                                {"vector": [0, 0, 0], "position": [0, 0, 0]}
                                for i in range(num_to_add - 1)
                            ],
                        },
                    }

            # Add vector to the shape dictionary.
            shape_dict[bname]["parameters"]["structure"].append(bond_geometry)
            max_length = max(
                (max_length, len(shape_dict[bname]["parameters"]["structure"]))
            )

        # Fill in bond shapes that are not the same length as the max length.
        for bname in shape_dict:
            missing = max_length - len(
                shape_dict[bname]["parameters"]["structure"]
            )
            if missing == 0:
                continue
            for _ in range(missing):
                fake_bond = {"vector": [0, 0, 0], "position": [0, 0, 0]}
                shape_dict[bname]["parameters"]["structure"].append(fake_bond)

    return shape_dict


def write_chemiscope_json(  # noqa: PLR0913
    json_file: pathlib.Path | str,
    structures: abc.Sequence[stk.Molecule],
    properties: dict[str, abc.Sequence[float | str | int]],
    bonds_as_shapes: bool,
    meta_dict: dict[str, str | list],
    x_axis_dict: dict[str, str],
    y_axis_dict: dict[str, str],
    z_axis_dict: dict[str, str],
    color_dict: dict[str, str | int | float],
    bond_hex_colour: str = "#fc5500",
) -> None:
    """Write the chemiscope json.

    Parameters:
        json_file:
            File to save to, this can be loaded into the `chemiscope`
            interface.

        structures:
            Sequence of structures to include. If they do not have the same
            number of properties, they may not all be added.

        properties:
            Properties with name as key and list of property values ordered by
            structure as values.

        bonds_as_shapes:
            `True` if you want to show the bonds as shapes, rather than bonds.

        meta_dict:
            Dictionary of meta information. See chemiscope for details, can
            include: `name`, `description`, `authors`, `references`.

        x_axis_dict:
            Dictionary of `{"property": "name-of-property"}`. Value can be `""`
            to be unused.

        y_axis_dict:
            Dictionary of `{"property": "name-of-property"}`. Value can be `""`
            to be unused.

        z_axis_dict:
            Dictionary of `{"property": "name-of-property"}`. Value can be `""`
            to be unused.

        color_dict:
            Dictionary of
            `{"property": "name-of-property", "min": float, "max": float}` to
            set the colour of data plot.

        bond_hex_colour:
            Colour for bonds made from shapes. Does not work if
            `bonds_as_shapes=False`.

    """
    if bonds_as_shapes:
        shape_dict = convert_stk_bonds_as_shapes(
            structures=structures,
            bond_color=bond_hex_colour,
            bond_radius=0.10,
        )
        shape_string = ",".join(shape_dict.keys())
        bonds = False

    else:
        shape_dict = None
        shape_string = ""
        bonds = True

    chemiscope.write_input(
        path=str(json_file),
        frames=structures,
        properties=properties,
        meta=meta_dict,
        settings=chemiscope.quick_settings(
            map_settings={
                "x": x_axis_dict,
                "y": y_axis_dict,
                "z": z_axis_dict,
                "color": color_dict,
                "palette": "plasma",
            },
            structure_settings={
                "shape": shape_string,
                "atoms": True,
                "bonds": bonds,
                "spaceFilling": False,
            },
        ),
        shapes=shape_dict,
    )
    logger.info("saved to %s", json_file)
