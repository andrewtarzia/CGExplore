
Systems Optimisation
====================

A package of the classes for optimising CG models in :mod:`cgexplore`.

.. note::

  Examples of these functions and the use of this package can be found in the
  `optimisation_example <https://github.com/andrewtarzia/CGExplore/tree/main/optimisation_example>`_
  directory.

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
by the user (described below) to:

.. toctree::
  :maxdepth: 1

  FitnessCalculator <_autosummary/cgexplore.systems_optimisation.FitnessCalculator>
  StructureCalculator <_autosummary/cgexplore.systems_optimisation.StructureCalculator>

.. note::
  The `options` argument allows the user to provide custom information or
  functions in a dictionary.

.. code-block:: python

  import cgexplore as cgx

  def fitness_function(
    chromosome,
    chromosome_generator,
    database_path,
    calculation_output,
    structure_output,
    options={},
  ):
    """Calculate the fitness of a chromosome.

    All arguments are needed, even if not used.

    Returns:
      float

    """
    database = cgx.utilities.AtomliteDatabase(database_path)
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

  def structure_function(
    chromosome,
    database_path,
    calculation_output,
    structure_output,
    options={},
  ):
    """Define model and calculate its properties.

    All arguments are needed, even if not used.

    """
    database = cgx.utilities.AtomliteDatabase(database_path)
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
  All of the below can be found in the script `optimisation_test.py`.

First you must define the :class:`.ChromosomeGenerator`, which holds all the
changeable gene information and allows iteration and selection of
:class:`.Chromosome` from it. You define the object, and then use the
:meth:`add_gene` or :meth:`add_forcefield_dict` methods to add the genes,
which the generator should analyse automatically to find the changeable
features.

.. code-block:: python

    # Define the chromosome generator, holding all the changeable genes.
    chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
        prefix=prefix,
        present_beads=(abead, bbead, cbead, dbead),
        vdw_bond_cutoff=2,
    )
    chromo_it.add_gene(
        iteration=(
            ("2P3", stk.cage.TwoPlusThree),
            ("4P6", stk.cage.FourPlusSix),
            ("4P62", stk.cage.FourPlusSix2),
            ("6P9", stk.cage.SixPlusNine),
            ("8P12", stk.cage.EightPlusTwelve),
        ),
        gene_type="topology",
    )
    # Set some basic building blocks up. This should be run by an algorithm
    # later.
    chromo_it.add_gene(
        iteration=(cgx.molecular.TwoC1Arm(bead=bbead, abead1=cbead),),
        gene_type="precursor",
    )
    chromo_it.add_gene(
        iteration=(cgx.molecular.ThreeC1Arm(bead=abead, abead1=dbead),),
        gene_type="precursor",
    )

    # Define the forcefield terms.
    definer_dict = {
        # Bonds.
        "ao": ("bond", 1.5, 1e5),
        "bc": ("bond", 1.5, 1e5),
        "co": ("bond", 1.0, 1e5),
        "cc": ("bond", 1.0, 1e5),
        "oo": ("bond", 1.0, 1e5),
        # Angles.
        "ccb": ("angle", 180.0, 1e2),
        "ooc": ("angle", 180.0, 1e2),
        "occ": ("angle", 180.0, 1e2),
        "ccc": ("angle", 180.0, 1e2),
        "oco": ("angle", 180.0, 1e2),
        "aoc": ("angle", 180.0, 1e2),
        "aoo": ("angle", 180.0, 1e2),
        "bco": ("angle", tuple(i for i in range(90, 181, 5)), 1e2),
        "cbc": ("angle", 180.0, 1e2),
        "oao": ("angle", tuple(i for i in range(50, 121, 5)), 1e2),
        # Torsions.
        "ocbco": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "o": ("nb", 10.0, 1.0),
    }
    chromo_it.add_forcefield_dict(definer_dict=definer_dict)

Then, you can run the genetic algorithm (I will not show all the information
for brevity):

.. code-block:: python

  # Define the structure and fitness calculators.
  fitness_calculator = cgx.systems_optimisation.FitnessCalculator(...)
  structure_calculator = cgx.systems_optimisation.StructureCalculator(...)

  # Set a random number generator for consistency. Normally, run over multiple
  # seeds.
  generator = np.random.default_rng(seed)

  initial_population = chromo_it.select_random_population(
      generator,
      size=selection_size,
  )

  # This holds the generational information.
  generations = []
  # Define a generation!
  generation = cgx.systems_optimisation.Generation(
      chromosomes=initial_population,
      fitness_calculator=fitness_calculator,
      structure_calculator=structure_calculator,
      num_processes=num_processes,
  )
  # Run the structures, and get its fitness.
  generation.run_structures()
  _ = generation.calculate_fitness_values()
  generations.append(generation)

  # Now we can iterate over mutations and generations.
  for generation_id in range(1, num_generations + 1):
      # Extend the list of new chromosomes with mutations.
      merged_chromosomes = []
      merged_chromosomes.extend(
          chromo_it.mutate_population(
              list_of_chromosomes=generation.chromosomes,
              generator=generator,
              gene_range=chromo_it.get_term_ids(),
              selection="random",
              num_to_select=5,
              database=database,
          )
      )
      merged_chromosomes.extend(chromo_it.mutate_population(...))

      # Extend the list of new chromosomes with crossovers.
      merged_chromosomes.extend(
          chromo_it.crossover_population(
              list_of_chromosomes=generation.chromosomes,
              generator=generator,
              selection="random",
              num_to_select=5,
              database=database,
          )
      )
      merged_chromosomes.extend(chromo_it.crossover_population(...))

      # Extend the list of new chromosomes with the best from the last
      # generation.
      merged_chromosomes.extend(generation.select_best(selection_size=5))

      # Define the new generation and run its structures and fitness.
      generation = cgx.systems_optimisation.Generation(
          chromosomes=chromo_it.dedupe_population(merged_chromosomes),
          ...
      )

      # Build, optimise and analyse each structure.
      generation.run_structures()
      _ = generation.calculate_fitness_values()

      # Add final state to generations.
      generations.append(generation)

      # Select the best of the generation for the next generation.
      best = generation.select_best(selection_size=selection_size)
      generation = cgx.systems_optimisation.Generation(
          chromosomes=chromo_it.dedupe_population(best),
          ...
      )
