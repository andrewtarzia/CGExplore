Atomistic structure prediction
==============================

Given two building blocks:

.. moldoc::

    import moldoc.molecule as molecule
    import stk

    tritopic_building_block = stk.BuildingBlock(
        smiles="C1=C(C=C(C=C1C=O)C=O)C=O",
        functional_groups=[stk.AldehydeFactory()],
    )

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=atom.get_atomic_number(),
                position=position,
            ) for atom, position in zip(
                tritopic_building_block.get_atoms(),
                tritopic_building_block.get_position_matrix(),
            )
        ),
        bonds=(
            molecule.Bond(
                atom1_id=bond.get_atom1().get_id(),
                atom2_id=bond.get_atom2().get_id(),
                order=bond.get_order(),
            ) for bond in tritopic_building_block.get_bonds()
        ),
    )


.. moldoc::

    import moldoc.molecule as molecule
    import stk

    ditopic_building_block = stk.BuildingBlock(
        smiles="NC1CCCCC1N",
        functional_groups=[stk.PrimaryAminoFactory()],
    )

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=atom.get_atomic_number(),
                position=position,
            ) for atom, position in zip(
                ditopic_building_block.get_atoms(),
                ditopic_building_block.get_position_matrix(),
            )
        ),
        bonds=(
            molecule.Bond(
                atom1_id=bond.get_atom1().get_id(),
                atom2_id=bond.get_atom2().get_id(),
                order=bond.get_order(),
            ) for bond in ditopic_building_block.get_bonds()
        ),
    )

Let's predict structures with ``m=2`` and ``s=2:3``.

First (but optional!), use `bbprepared
<https://bbprepared.readthedocs.io/en/latest/recipes/recipe_2.html>`_
to get the lowest energy conformer.

Then we define the system.

.. testcode:: recipe4-test
    :hide:

    import stk

    tritopic_building_block = stk.BuildingBlock(
        smiles="C1=C(C=C(C=C1C=O)C=O)C=O",
        functional_groups=[stk.AldehydeFactory()],
    )
    ditopic_building_block = stk.BuildingBlock(
        smiles="NC1CCCCC1N",
        functional_groups=[stk.PrimaryAminoFactory()],
    )


.. testcode:: recipe4-test

    import stk
    import stko
    import cgexplore as cgx
    import logging
    import pathlib

    logger = logging.getLogger(__name__)

    # Define a working directory.
    wd = pathlib.Path.cwd() / "source"/ "recipes" / "recipe_4_output"

    # Currently, this definition is up to the user, but we will make this
    # uniform soon.
    building_block_library = {
        "ditopic": ditopic_building_block,
        "tritopic": tritopic_building_block,
    }

    # Currently, this definition is up to the user, but we will make this
    # uniform soon.
    systems = {
        "s1": {
            # Always order with the most functional groups first.
            "stoichiometry_map": {"tritopic": 2, "ditopic": 3},
            "multipliers": (1, 2),
        },
    }


Collate iterators as a function of mulipliers, graphs and building block
configurations.

.. testcode:: recipe4-test

    syst_d = systems["s1"]
    iterators = {}
    for multiplier in syst_d["multipliers"]:
        # Define the iterator.
        iterator = cgx.scram.TopologyIterator(
            building_block_counts={
                building_block_library[name]: stoich * multiplier
                for name, stoich in syst_d["stoichiometry_map"].items()
            },
        )
        logger.info(
            "graph iteration has %s graphs", iterator.count_graphs()
        )
        iterators[multiplier] = iterator

.. testcode:: recipe4-test
    :hide:

    assert iterator.graph_type == "4-3FG_6-2FG"
    assert len(iterators) == 2
    assert iterator.count_graphs() == 5

For each iterator, we can build a test molecule and compile them for further
analysis. Note that the process from here on is much simplified than one would
use for production structure prediction. For example,
`atomistic structure prediction <https://github.com/andrewtarzia/topology_scrambler/src/model_enumeration/mgen_cs6.py>`_
in my recent work was more complicated.

.. testcode:: recipe4-test

    for multiplier in syst_d["multipliers"]:
        iterator = iterators[multiplier]
        for topology_code in iterator.yield_graphs():
            # Filter graphs for 1-loops.
            if topology_code.contains_parallels():
                continue

            name = f"s1_{multiplier}_{topology_code.idx}"

            # Use vertex set regraphing.
            constructed_molecule = cgx.scram.get_vertexset_molecule(
                layout_type="kamada",
                scale=5,
                topology_code=topology_code,
                iterator=iterator,
                configuration=None,
            )
            # Output to file.
            # constructed_molecule.write(wd / f"{name}_unopt.mol")

            # Implement optimisation workflows!

            # And then do some analysis!


Here are the three generated structures, including (finally) the well known
porous organic cage, CC3.

.. moldoc::

    import moldoc.molecule as molecule
    import stk
    import pathlib

    try:
        wd = pathlib.Path.cwd() / "source" / "recipes" / "recipe_4_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_1_0_unopt.mol"))
    except OSError:
        wd = pathlib.Path.cwd() / "recipes" / "recipe_4_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_1_0_unopt.mol"))

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=atom.get_atomic_number(),
                position=position,
            ) for atom, position in zip(
                structure.get_atoms(),
                structure.get_position_matrix(),
            )
        ),
        bonds=(
            molecule.Bond(
                atom1_id=bond.get_atom1().get_id(),
                atom2_id=bond.get_atom2().get_id(),
                order=bond.get_order(),
            ) for bond in structure.get_bonds()
        ),
    )


.. moldoc::

    import moldoc.molecule as molecule
    import stk
    import pathlib

    try:
        wd = pathlib.Path.cwd() / "source" / "recipes" / "recipe_4_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_2_1_unopt.mol"))
    except OSError:
        wd = pathlib.Path.cwd() / "recipes" / "recipe_4_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_2_1_unopt.mol"))

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=atom.get_atomic_number(),
                position=position,
            ) for atom, position in zip(
                structure.get_atoms(),
                structure.get_position_matrix(),
            )
        ),
        bonds=(
            molecule.Bond(
                atom1_id=bond.get_atom1().get_id(),
                atom2_id=bond.get_atom2().get_id(),
                order=bond.get_order(),
            ) for bond in structure.get_bonds()
        ),
    )

.. moldoc::

    import moldoc.molecule as molecule
    import stk
    import pathlib

    try:
        wd = pathlib.Path.cwd() / "source" / "recipes" / "recipe_4_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_2_4_unopt.mol"))
    except OSError:
        wd = pathlib.Path.cwd() / "recipes" / "recipe_4_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_2_4_unopt.mol"))

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=atom.get_atomic_number(),
                position=position,
            ) for atom, position in zip(
                structure.get_atoms(),
                structure.get_position_matrix(),
            )
        ),
        bonds=(
            molecule.Bond(
                atom1_id=bond.get_atom1().get_id(),
                atom2_id=bond.get_atom2().get_id(),
                order=bond.get_order(),
            ) for bond in structure.get_bonds()
        ),
    )

.. raw:: html

    <a class="btn-download" href="../_static/recipes/recipe_4.py" download>⬇️ Download Python Script</a>
