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
        shape_dict = chemiscope.convert_stk_bonds_as_shapes(
            frames=structures,
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
    logging.info("saved to %s", json_file)
