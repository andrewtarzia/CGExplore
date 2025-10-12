Minimal model genetic algorithm
===============================

We first define a chromosome of one pair of building blocks, with a range of
forcefield parameters and multiple topology graph choices.

.. testcode:: recipe1-test
    :hide:

    import stk
    import stko
    import cgexplore as cgx
    import logging
    import pathlib

    logger = logging.getLogger(__name__)

    # Define a working directory.
    wd = pathlib.Path.cwd() / "source"/ "recipes" / "recipe_1_output"
    struct_output = wd / "structures"
    calc_dir = wd / "calculations"
    data_dir = wd / "data"
    figure_dir = wd / "figures"


.. testcode:: recipe1-test

    # Define a database, and a prefix for naming structure, forcefield and
    # output files.
    prefix = "opt"
    database_path = data_dir / "test.db"
    database = cgx.utilities.AtomliteDatabase(database_path)

    # Define beads.
    bead_library = cgx.molecular.BeadLibrary.from_bead_types(
        # Type and coordination.
        {"a": 3, "b": 2, "c": 2, "o": 2}
    )

    syst_d = systems["s1"]
    iterators = {}
    for multiplier in syst_d["multipliers"]:
        # Automate the graph type naming.
        graph_type = ""
        building_block_counts = {}
        for name, stoich in syst_d["stoichiometry_map"].items():
            fgnum = building_block_library[name].get_num_functional_groups()
            graph_type += f"{stoich * multiplier}-{fgnum}FG_"
            building_block_counts[building_block_library[name]] = (
                stoich * multiplier
            )

        graph_type = graph_type.rstrip("_")

        # Define the iterator.
        iterator = cgx.scram.TopologyIterator(
            building_block_counts=building_block_counts,
            graph_type=graph_type,
            # Use a known graph set.
            graph_set="rxx",
        )
        logger.info(
            "graph iteration has %s graphs", iterator.count_graphs()
        )
        iterators[multiplier] = iterator

.. testcode:: recipe1-test
    :hide:

    assert graph_type == "4-3FG_6-2FG"
    assert len(iterators) == 2
    assert iterator.count_graphs() == 5

For each iterator, we can build a test molecule and compile them for further
analysis. Note that the process from here on is much simplified than one would
use for production structure prediction. For example,
`atomistic structure prediction <https://github.com/andrewtarzia/topology_scrambler/src/model_enumeration/mgen_cs6.py>`_
in my recent work was more complicated.

.. testcode:: recipe1-test

    for multiplier in syst_d["multipliers"]:
        iterator = iterators[multiplier]
        for idx, topology_code in enumerate(iterator.yield_graphs()):
            # Filter graphs for 1-loops.
            if topology_code.contains_parallels():
                continue

            name = f"s1_{multiplier}_{idx}"

            # Use vertex set regraphing.
            constructed_molecule = cgx.scram.get_vertexset_molecule(
                graph_type="kamada",
                scale=5,
                topology_code=topology_code,
                iterator=iterator,
                bb_config=None,
            )
            constructed_molecule.write(wd / f"{name}_unopt.mol")

            # Implement optimisation workflows!

            # And then do some analysis!


Here are the three generated structures, including (finally) the well known
porous organic cage, CC3.

.. moldoc::

    import moldoc.molecule as molecule
    import stk
    import pathlib

    try:
        wd = pathlib.Path.cwd() / "source" / "recipes" / "recipe_1_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_1_0_unopt.mol"))
    except OSError:
        wd = pathlib.Path.cwd() / "recipes" / "recipe_1_output"
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
        wd = pathlib.Path.cwd() / "source" / "recipes" / "recipe_1_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_2_1_unopt.mol"))
    except OSError:
        wd = pathlib.Path.cwd() / "recipes" / "recipe_1_output"
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
        wd = pathlib.Path.cwd() / "source" / "recipes" / "recipe_1_output"
        structure = stk.BuildingBlock.init_from_file(str(wd / "s1_2_4_unopt.mol"))
    except OSError:
        wd = pathlib.Path.cwd() / "recipes" / "recipe_1_output"
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

    <a class="btn-download" href="_static/recipes/recipe_1.py" download>⬇️ Download Python Script</a>
