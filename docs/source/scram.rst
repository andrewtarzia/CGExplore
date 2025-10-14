Scrambler: blind structure prediction
=====================================

A package of the classes for scrambling and blind structure prediction in
:mod:`cgexplore`.

.. toctree::
  :maxdepth: 1

  Available graphs <avail_graphs>

The iterators
-------------

.. note::

  These are under a lot of development!


- :doc:`TopologyIterator <_autosummary/cgexplore.scram.TopologyIterator>`
- :doc:`get_custom_bb_configurations <_autosummary/cgexplore.scram.get_custom_bb_configurations>`


Containers
----------

These are likely to be updated, but when performing structure prediction,
you will use these classes.

- :doc:`TopologyCode <_autosummary/cgexplore.scram.TopologyCode>`
- :doc:`BuildingBlockConfiguration <_autosummary/cgexplore.scram.BuildingBlockConfiguration>`
- :doc:`Constructed <_autosummary/cgexplore.scram.Constructed>`

Performing target optimisation
------------------------------

Using :func:`cgexplore.scram.target_optimisation`,
it is now possible to optimise a forcefield or a produced structure to minimise
the structure energy (energy per building block).

.. note::

  Examples of this function can be found in `recipe 6 <recipes/recipe_6.html>`_.

- :doc:`target_optimisation <_autosummary/cgexplore.scram.target_optimisation>`


Graph manipulation and mapping to building block configurations
---------------------------------------------------------------

- :doc:`get_bb_topology_code_graph <_autosummary/cgexplore.scram.get_bb_topology_code_graph>`
- :doc:`get_potential_bb_dicts <_autosummary/cgexplore.scram.get_potential_bb_dicts>`
- :doc:`passes_graph_bb_iso <_autosummary/cgexplore.scram.passes_graph_bb_iso>`


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
