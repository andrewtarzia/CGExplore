Skewed graphs
=============

Since our recent publication it became clear that exhaustive searching of
graph space for large number of nodes was intractable.

.. warning::

  The skewed iterator below only works for systems that do not contain
  parallel edges. It may not catch this mistake, so caution!

So, under the assumption that cages are structurally concave with an
empty pore, we can apply graph theory to optimise the graph set.
In particular, ``planar`` graphs, or graphs with a ``skewness`` of 0, match
this assumption. And the :class:`cgx.scram.SkewedTopologyIterator` provides the
same graph generation interface, but now there are less graphs and they are
focussed on planar graphs in the ``skw`` set.

The graphing aspect and the graphs available in :mod:`cgexplore.scram` for
performing unbiased structure prediction is still part of :mod:`agx`, which can
be found in the
`documentation <https://agx.readthedocs.io/en/latest/skewed_graphs.html>`_.

.. important::

  All skewed graph sets (``skw``) are run with a ``max_samples`` of 2e5
  (compared to 1e6 for the standard iterator).

If you use the skewed iterator in `recipe 7 <recipes/recipe_7.html>`_, you
can find all known ``stk`` graphs except one that has parallel edges.
