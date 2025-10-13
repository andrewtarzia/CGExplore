Molecular
=========

CG-Beads
--------

.. note::

    It is recommended to use the method
    :meth:`from_bead_types` in :class:`cgexplore.molecular.BeadLibrary` to
    create a :class:`cgexplore.molecular.CgBead` instance.
    This will automatically assign element strings based on coordination
    numbers, avoiding double ups and simplifying the selection process.


- :doc:`BeadLibrary <_autosummary/cgexplore.molecular.BeadLibrary>`
- :doc:`CgBead <_autosummary/cgexplore.molecular.CgBead>`
- :doc:`string_to_atom_number <_autosummary/cgexplore.molecular.string_to_atom_number>`
- :doc:`periodic_table <_autosummary/cgexplore.molecular.periodic_table>`

Molecule containers
-------------------

- :doc:`Conformer <_autosummary/cgexplore.molecular.Conformer>`
- :doc:`SpindryConformer <_autosummary/cgexplore.molecular.SpindryConformer>`
- :doc:`Ensemble <_autosummary/cgexplore.molecular.Ensemble>`
- :doc:`Timestep <_autosummary/cgexplore.molecular.Timestep>`

Precursors
----------

These classes provide interfaces between :mod:`stk` and :mod:`cgexplore`. A
:class:`cgexplore.molecular.Precursor` contains an :class:`stk.BuildingBlock`
for molecule construction.

- :doc:`Precursor <_autosummary/cgexplore.molecular.Precursor>`
- :doc:`FourC0Arm <_autosummary/cgexplore.molecular.FourC0Arm>`
- :doc:`FourC1Arm <_autosummary/cgexplore.molecular.FourC1Arm>`
- :doc:`ThreeC0Arm <_autosummary/cgexplore.molecular.ThreeC0Arm>`
- :doc:`ThreeC1Arm <_autosummary/cgexplore.molecular.ThreeC1Arm>`
- :doc:`ThreeC2Arm <_autosummary/cgexplore.molecular.ThreeC2Arm>`
- :doc:`TwoC0Arm <_autosummary/cgexplore.molecular.TwoC0Arm>`
- :doc:`TwoC1Arm <_autosummary/cgexplore.molecular.TwoC1Arm>`
- :doc:`TwoC2Arm <_autosummary/cgexplore.molecular.TwoC2Arm>`
- :doc:`TwoC3Arm <_autosummary/cgexplore.molecular.TwoC3Arm>`

For rotatable precursors:


- :doc:`SixBead <_autosummary/cgexplore.molecular.SixBead>`

For precursors with steric bulk:

- :doc:`StericTwoC1Arm <_autosummary/cgexplore.molecular.StericTwoC1Arm>`
- :doc:`StericSixBead <_autosummary/cgexplore.molecular.StericSixBead>`
- :doc:`StericSevenBead <_autosummary/cgexplore.molecular.StericSevenBead>`

For precursors useful in random/algorithmic structure generation:

- :doc:`GeneratedPrecursor <_autosummary/cgexplore.molecular.GeneratedPrecursor>`
- :doc:`PrecursorGenerator <_autosummary/cgexplore.molecular.PrecursorGenerator>`
- :doc:`VaBene <_autosummary/cgexplore.molecular.VaBene>`
- :doc:`VaBeneGenerator <_autosummary/cgexplore.molecular.VaBeneGenerator>`
- :doc:`check_fit <_autosummary/cgexplore.molecular.check_fit>`


.. For rigid, shape-based precursors:

..   - :doc:`ShapePrecursor <_autosummary/cgexplore.molecular.ShapePrecursor>`
..   - :doc:`DiatomShape <_autosummary/cgexplore.molecular.DiatomShape>`
..   - :doc:`TriangleShape <_autosummary/cgexplore.molecular.TriangleShape>`
..   - :doc:`SquareShape <_autosummary/cgexplore.molecular.SquareShape>`
..   - :doc:`StarShape <_autosummary/cgexplore.molecular.StarShape>`
..   - :doc:`HexagonShape <_autosummary/cgexplore.molecular.HexagonShape>`
..   - :doc:`TdShape <_autosummary/cgexplore.molecular.TdShape>`
..   - :doc:`CuShape <_autosummary/cgexplore.molecular.CuShape>`
..   - :doc:`OcShape <_autosummary/cgexplore.molecular.OcShape>`
..   - :doc:`V2P3Shape <_autosummary/cgexplore.molecular.V2P3Shape>`
..   - :doc:`V2P4Shape <_autosummary/cgexplore.molecular.V2P4Shape>`
..   - :doc:`V4P62Shape <_autosummary/cgexplore.molecular.V4P62Shape>`
..   - :doc:`V6P9Shape <_autosummary/cgexplore.molecular.V6P9Shape>`
..   - :doc:`V8P16Shape <_autosummary/cgexplore.molecular.V8P16Shape>`
..   - :doc:`V10P20Shape <_autosummary/cgexplore.molecular.V10P20Shape>`
..   - :doc:`V12P24Shape <_autosummary/cgexplore.molecular.V12P24Shape>`
..   - :doc:`V12P30Shape <_autosummary/cgexplore.molecular.V12P30Shape>`
..   - :doc:`V20P30Shape <_autosummary/cgexplore.molecular.V20P30Shape>`
..   - :doc:`V24P48Shape <_autosummary/cgexplore.molecular.V24P48Shape>`
