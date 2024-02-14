
Analysis
========

A package of the classes for analysing molecules in :mod:`.CGExplore`.

.. toctree::
  :maxdepth: 1

  Analysis module <_autosummary/cgexplore.analysis>

Geometry
--------

To analyse the geometry of CG molecules (including size and pore metrics),
see:

.. toctree::
  :maxdepth: 1

  GeomMeasure <_autosummary/cgexplore.analysis.GeomMeasure>

To calculate torsions:

.. toctree::
  :maxdepth: 1

  get_dihedral <_autosummary/cgexplore.analysis.get_dihedral>

Shape
-----

We have a series of functions (moving into one class
:class:`cgexplore.analysing.ShapeMeasure`) for interfacing with
`Shape <https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/>`_.
I recommend using what is in this class:

.. toctree::
  :maxdepth: 1

  ShapeMeasure <_autosummary/cgexplore.analysis.ShapeMeasure>

Old utility functions (will be removed one day):

.. toctree::
  :maxdepth: 1

  fill_position_matrix_molecule <_autosummary/cgexplore.analysis.fill_position_matrix_molecule>
  fill_position_matrix <_autosummary/cgexplore.analysis.fill_position_matrix>
  get_shape_molecule_byelement <_autosummary/cgexplore.analysis.get_shape_molecule_byelement>
  get_shape_molecule_ligands <_autosummary/cgexplore.analysis.get_shape_molecule_ligands>
  get_shape_molecule_nodes <_autosummary/cgexplore.analysis.get_shape_molecule_nodes>
  known_shape_vectors <_autosummary/cgexplore.analysis.known_shape_vectors>
  test_shape_mol <_autosummary/cgexplore.analysis.test_shape_mol>


