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
    """Write the chemiscope json."""
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
        shape_string = False
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
