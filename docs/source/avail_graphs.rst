Available graphs
================

A list of the graphs available in :mod:`cgexplore.scram` for performing blind
structure prediction. Files can be found in ``src/cgexplore/_internal/scram/known_graphs``.

.. important::

  All previous graphs will be kept in ``src/cgexplore/_internal/scram/known_graphs``
  and are available for use, however, I recommend using the new naming convention.

New naming convention ``rxx``:
------------------------------

For the new ``graph_set``: ``rxx``, ``graph_type`` names are defined by
separating the building blocks of different number of FGs by underscores and
using hyphens to distinguish the multiplier by the number of FGs.

For example: ``rxx_4-4FG_6-2FG_4-1FG`` has 4 4FG building blocks,
6 2FG building blocks and 4 1FG building blocks.

Graphs:
-------

.. important::

  All new graphs are run with a ``max_samples`` of 1e7 (compared to 1e5 of
  ``cgexplore`` version ``2025.2.5.1``).

Filtering graphs
^^^^^^^^^^^^^^^^

The code no longer hard codes filtering within generation, but
:class:`cgexplore.scram.TopologyCode` offers methods for filtering for
double wells or parallel edges.
Additionally, one might want to build graphs that are not only
``one connected graph``. To do so, you must change ``allowed_num_components``.


.. important::

  While we have one, two and three type graphs below, within each, any
  stoichiometry or configuration of same-numbered functional groups can be used.
  For example, a heteroleptic cage of form Pd_n L_n L'_n is a two type graph
  with ``n`` 4-FG Pd, ``n`` 2-FG L and ``n`` 2-FG L' building blocks.


One-type graphs
^^^^^^^^^^^^^^^

Produced graphs for ``m`` in (1 - 10) with FGs in (1 - 4).
Generated with code:

.. code-block:: python

    bbs = {
        1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
        2: stk.BuildingBlock("BrCCCCBr", (stk.BromoFactory(),)),
        3: stk.BuildingBlock("BrCC(Br)CCBr", (stk.BromoFactory(),)),
        4: stk.BuildingBlock("BrCC(Br)CCC(Br)CCBr", (stk.BromoFactory(),)),
    }
    multipliers = range(1, 11)

    for midx, fgnum in it.product(multipliers, bbs):
        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum]: midx,
                },
                graph_type=f"{midx}-{fgnum}FG",
                graph_set="rxx",
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            pass


Two-type graphs
^^^^^^^^^^^^^^^

Produced graphs for ``m`` in (1 - 12) with FGs in (1 - 4) and
stoichiometries of ``bigger``:``smaller`` (in terms of FGs): 1:2, 2:3, 3:4.
Generated with code:

.. code-block:: python

    bbs = {
        1: stk.BuildingBlock("BrC", (stk.BromoFactory(),)),
        2: stk.BuildingBlock("BrCCCCBr", (stk.BromoFactory(),)),
        3: stk.BuildingBlock("BrCC(Br)CCBr", (stk.BromoFactory(),)),
        4: stk.BuildingBlock("BrCC(Br)CCC(Br)CCBr", (stk.BromoFactory(),)),
    }
    multipliers = range(1, 13)

    two_type_stoichiometries = ((1, 2), (2, 3), (3, 4))
    for midx, fgnum1, fgnum2, stoich in it.product(
        multipliers, bbs, bbs, two_type_stoichiometries
    ):
        if fgnum1 == fgnum2:
            continue
        fgnum1_, fgnum2_ = sorted((fgnum1, fgnum2), reverse=True)

        try:
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bbs[fgnum1_]: midx * stoich[0],
                    bbs[fgnum2_]: midx * stoich[1],
                },
                graph_type=f"{midx * stoich[0]}-{fgnum1_}FG_"
                f"{midx * stoich[1]}-{fgnum2_}FG",
                graph_set="rxx",
            )
            logging.info(
                "graph iteration has %s graphs", iterator.count_graphs()
            )
        except (ZeroDivisionError, ValueError):
            pass


Three-type graphs
^^^^^^^^^^^^^^^^^
