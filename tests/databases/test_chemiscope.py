import json
import pathlib

import numpy as np
import pytest

from cgexplore.utilities import write_chemiscope_json

from .case_data import CaseData


def test_chemiscope(molecule: CaseData) -> None:
    """Test :function:`.write_chemiscope_json`."""
    path = pathlib.Path(__file__).resolve().parent / "test.json"

    if path.exists():
        path.unlink()

    properties = {}
    for propm in molecule.property_dicts:
        for prop in propm:
            if prop not in properties:
                properties[prop] = []
            properties[prop].append(propm[prop])

    # Run a test with the provided dicts, which do not work because one
    # property is too long.
    with pytest.raises(
        ValueError,
        match=(
            "The length of property values is different from the number of "  # noqa: RUF043, RUF100
            "structures and the number of atoms, we can not guess the target. "
            "Got n_atoms = 40, n_structures = 4, the length of property values"
            " is 1, for the '2' property"
        ),
    ):
        write_chemiscope_json(
            json_file=path,
            structures=molecule.molecules,
            properties=properties,
            bonds_as_shapes=True,
            meta_dict={
                "name": "a test!",
                "description": "testing this",
                "authors": ["Andrew Tarzia"],
                "references": ["Testing code"],
            },
            x_axis_dict={"property": "1"},
            y_axis_dict={"property": "3"},
            z_axis_dict={"property": ""},
            color_dict={"property": ""},
            bond_hex_colour="#ffffff",
        )

    # Then test with the filter.
    to_delete = [
        prop
        for prop in properties
        if len(properties[prop]) != len(molecule.molecules)
    ]
    for prop in to_delete:
        properties.pop(prop, None)

    write_chemiscope_json(
        json_file=path,
        structures=molecule.molecules,
        properties=properties,
        bonds_as_shapes=True,
        meta_dict={
            "name": "a test!",
            "description": "testing this",
            "authors": ["Andrew Tarzia"],
            "references": ["Testing code"],
        },
        x_axis_dict={"property": "1"},
        y_axis_dict={"property": "3"},
        z_axis_dict={"property": ""},
        color_dict={"property": ""},
        bond_hex_colour="#ffffff",
    )

    with path.open("r") as f:
        test_data = json.load(f)

    assert test_data["properties"] == {
        "1": {"target": "structure", "values": [2.0, 2.0, 2.0, 3.0]},
        "3": {
            "target": "structure",
            "values": ["astr", "bstr", "cstr", "cstr"],
        },
    }
    assert test_data["settings"]["structure"][0]["shape"] == (
        "bond_0,bond_1,bond_2,bond_3,bond_4,bond_5,bond_6,bond_7,bond_8,"
        "bond_9,bond_10"
    )
    assert len(test_data["structures"]) == len(molecule.molecules)

    for test_molecule, json_molecule in zip(
        molecule.molecules, test_data["structures"], strict=True
    ):
        pos_mat = test_molecule.get_position_matrix()
        assert test_molecule.get_num_atoms() == json_molecule["size"]
        for i, atom in enumerate(test_molecule.get_atoms()):
            assert json_molecule["names"][i] == atom.__class__.__name__

            posx, posy, posz = pos_mat[i]

            assert np.isclose(posx, json_molecule["x"][i], rtol=0, atol=1e-6)
            assert np.isclose(posy, json_molecule["y"][i], rtol=0, atol=1e-6)
            assert np.isclose(posz, json_molecule["z"][i], rtol=0, atol=1e-6)

    # Just test one.
    assert test_data["shapes"]["bond_4"] == {
        "kind": "cylinder",
        "parameters": {
            "global": {"radius": 0.1, "color": "#ffffff"},
            "structure": [
                {
                    "vector": [
                        0.22304293537021347,
                        0.054461393723354276,
                        -1.0952980247208781,
                    ],
                    "position": [
                        0.7563930326851623,
                        -0.01926391540105227,
                        -0.036668302736346435,
                    ],
                },
                {
                    "vector": [
                        -0.5223374946083974,
                        0.8112914209329797,
                        0.5661687958436197,
                    ],
                    "position": [
                        -1.1975128536868043,
                        -0.07986658455873856,
                        -0.0564288743416208,
                    ],
                },
                {
                    "vector": [
                        -0.01425198400365102,
                        0.9614121228603414,
                        -0.5657493874400913,
                    ],
                    "position": [
                        -1.7244190203314924,
                        -0.31510088950566123,
                        -0.07183607003687872,
                    ],
                },
                {
                    "vector": [
                        -0.5223374946083974,
                        0.8112914209329797,
                        0.5661687958436197,
                    ],
                    "position": [
                        -1.1975128536868043,
                        -0.07986658455873856,
                        -0.0564288743416208,
                    ],
                },
            ],
        },
    }

    path.unlink()
