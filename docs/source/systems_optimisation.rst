
Systems Optimisation
====================

A package of the classes for optimising CG models in :mod:`.CGExplore`.

.. note::

  Examples of these functions and the use of this package can be found in the
  `optimisation_example <https://github.com/andrewtarzia/CGExplore/tree/main/optimisation_example>`_
  directory.

.. toctree::
  :maxdepth: 1

  Systems optimisation module <_autosummary/cgexplore.systems_optimisation>

Inputs
------

A :class:`cgexplore.systems_optimisation.ChromosomeGenerator` is used to add
`genes` to an optimisation problem and automatically provide a list of
:class:`cgexplore.systems_optimisation.Chromosome` for modelling (plus helpful
methods for exploring this library through optimisation algorithms, such as a
genetic algorithm).

.. toctree::
  :maxdepth: 1

  ChromosomeGenerator <_autosummary/cgexplore.systems_optimisation.ChromosomeGenerator>
  Chromosome <_autosummary/cgexplore.systems_optimisation.Chromosome>

Generation Handling
-------------------

Once you have a library of chromosomes, you can iterate over them to produce
generations, which are handled here:

.. toctree::
  :maxdepth: 1

  Generation <_autosummary/cgexplore.systems_optimisation.Generation>


Fitness and Structure Calculation
---------------------------------

A :class:`cgexplore.systems_optimisation.Generation` requires the definition of
how a :class:`cgexplore.systems_optimisation.Chromosome` is translated into a
model and how to calculate that models fitness. These are provided as functions
by the user of the form:

.. note::
  The `options` argument allows the user to provide custom information or
  functions in a dictionary.

.. code-block:: python

  def fitness_calculator(
    chromosome,
    chromosome_generator,
    database,
    calculation_output,
    structure_output,
    options={},
  ):
    """Calculate the fitness of a chromosome.

    All arguments are needed, even if not used.

    Returns:
      float

    """
    target_pore = 2
    name = f"{chromosome.prefix}_{chromosome.get_string()}"

    # Extract needed properties from database (done in `structure_calculator`).
    entry = database.get_entry(name)
    tstr = entry.properties["topology"]
    pore = entry.properties["opt_pore_data"]["min_distance"]
    energy = entry.properties["energy_per_bb"]
    pore_diff = abs(target_pore - pore) / target_pore

    fitness = 1 / (pore_diff + energy)

    # Add fitness to database.
    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


.. code-block:: python

  def structure_calculator(
    chromosome,
    database,
    calculation_output,
    structure_output,
    options={},
  ):
    """Define model and calculate its properties.

    All arguments are needed, even if not used.

    """
    # Build structure.
    topology_str, topology_fun = chromosome.get_topology_information()
    building_blocks = chromosome.get_building_blocks()
    cage = stk.ConstructedMolecule(topology_fun(building_blocks))
    name = f"{chromosome.prefix}_{chromosome.get_string()}"

    # Select forcefield by chromosome.
    forcefield = chromosome.get_forcefield()

    # Optimise with some procedure.
    conformer = optimise_cage(
        molecule=cage,
        name=name,
        output_dir=calculation_output,
        forcefield=forcefield,
        platform=None,
        database=database,
        chromosome=chromosome,
    )

    # Analyse cage.
    analyse_cage(
        name=name,
        output_dir=calculation_output,
        forcefield=forcefield,
        node_element="Ag",
        database=database,
        chromosome=chromosome,
    )


Anatomies of a script
---------------------

.. important::
  WIP: I need to add further examples (like figures) from the test script.

