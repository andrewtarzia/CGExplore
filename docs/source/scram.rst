Scrambler: blind structure prediction
=====================================

A package of the classes for scrambling and blind structure prediction in
:mod:`cgexplore`.

.. toctree::
  :maxdepth: 1

  TopologyIterator <_autosummary/cgexplore.scram.TopologyIterator>
  Available graphs <avail_graphs>

Containers
----------

These are likely to be updated, but when performing structure prediction,
you will use these classes.

.. toctree::
  :maxdepth: 1


  TopologyCode <_autosummary/cgexplore.scram.TopologyCode>
  BuildingBlockConfiguration <_autosummary/cgexplore.scram.BuildingBlockConfiguration>
  Constructed <_autosummary/cgexplore.scram.Constructed>

Performing target optimisation
------------------------------

Using :func:`cgexplore.scram.target_optimisation`,
it is now possible to optimise a forcefield or a produced structure to minimise
the structure energy (energy per building block).

.. toctree::
  :maxdepth: 1

  target_optimisation <_autosummary/cgexplore.scram.target_optimisation>


Graph manipulation and mapping to building block configurations
---------------------------------------------------------------

.. toctree::
  :maxdepth: 1

  get_bb_topology_code_graph <_autosummary/cgexplore.scram.get_bb_topology_code_graph>
  get_custom_bb_configurations <_autosummary/cgexplore.scram.get_custom_bb_configurations>
  get_potential_bb_dicts <_autosummary/cgexplore.scram.get_potential_bb_dicts>
  passes_graph_bb_iso <_autosummary/cgexplore.scram.passes_graph_bb_iso>


Construction with arbitrary graphs
----------------------------------

.. toctree::
  :maxdepth: 1

  get_vertexset_molecule <_autosummary/cgexplore.scram.get_vertexset_molecule>
  get_regraphed_molecule <_autosummary/cgexplore.scram.get_regraphed_molecule>
  get_stk_topology_code <_autosummary/cgexplore.scram.get_stk_topology_code>
  optimise_cage <_autosummary/cgexplore.scram.optimise_cage>
  optimise_from_files <_autosummary/cgexplore.scram.optimise_from_files>
  graph_optimise_cage <_autosummary/cgexplore.scram.graph_optimise_cage>
  points_on_sphere <_autosummary/cgexplore.scram.points_on_sphere>
  try_except_construction <_autosummary/cgexplore.scram.try_except_construction>
