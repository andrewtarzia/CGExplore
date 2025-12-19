Scrambler: blind structure prediction
=====================================

A package of the classes for scrambling and blind structure prediction in
:mod:`cgexplore`.

The graphing aspect and the graphs available in :mod:`cgexplore.scram` for
performing blind structure prediction is now part of :mod:`agx`, which can
be found in the
`agx documentation <https://agx.readthedocs.io/en/latest/index.html>`_.

.. toctree::
  :maxdepth: 1

  Available graphs <avail_graphs>

The iterators
-------------

This class now handles graph and building block configuration exploration.

- :doc:`TopologyIterator <_autosummary/cgexplore.scram.TopologyIterator>`


Containers
----------

These are likely to be updated, but when performing structure prediction,
you will use these classes to access the new graphs/configurations.

- :doc:`TopologyCode <_autosummary/cgexplore.scram.TopologyCode>`
- :doc:`Configuration <_autosummary/cgexplore.scram.Configuration>`

Performing target optimisation
------------------------------

Using :func:`cgexplore.scram.target_optimisation`,
it is now possible to optimise a forcefield or a produced structure to minimise
the structure energy (energy per building block).

.. note::

  Examples of this function can be found in `recipe 6 <recipes/recipe_6.html>`_.

- :doc:`target_optimisation <_autosummary/cgexplore.scram.target_optimisation>`


Construction with arbitrary graphs
----------------------------------

- :doc:`get_vertexset_molecule <_autosummary/cgexplore.scram.get_vertexset_molecule>`
- :doc:`get_regraphed_molecule <_autosummary/cgexplore.scram.get_regraphed_molecule>`
- :doc:`get_stk_topology_code <_autosummary/cgexplore.scram.get_stk_topology_code>`
- :doc:`optimise_cage <_autosummary/cgexplore.scram.optimise_cage>`
- :doc:`optimise_from_files <_autosummary/cgexplore.scram.optimise_from_files>`
- :doc:`graph_optimise_cage <_autosummary/cgexplore.scram.graph_optimise_cage>`
- :doc:`points_on_sphere <_autosummary/cgexplore.scram.points_on_sphere>`
- :doc:`try_except_construction <_autosummary/cgexplore.scram.try_except_construction>`
