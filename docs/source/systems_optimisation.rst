
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

.. code-block:: python

  def fitness_calculator():
    pass


.. code-block:: python

  def structure_calculator():
    pass

.. important::
  Add further anatomies (like figures) from the test script.
