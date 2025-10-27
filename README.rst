:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_
:documentation: https://cgexplore.readthedocs.io/en/latest/

.. figure:: docs/source/_static/logo.png


Overview
========

``cgexplore`` or ``cgx`` is a general toolkit built on
`stk <https://stk.readthedocs.io/en/stable/>`_ for constructing,
optimising and exploring molecular coarse-grained models.

.. important::

  **Warning**: This package is still very much underdevelopment and many changes
  are expected.

  In particular, if you are trying to reproduce exactly
  `our recent work on structure prediction <https://chemrxiv.org/engage/chemrxiv/article-details/68f0ef40bc2ac3a0e051be52>`_,
  then you should install an old version of the code
  (``cgexplore==2025.2.5.1``) alongside the `topology_scrambler <https://github.com/andrewtarzia/topology_scrambler/tree/main>`_
  code and use `these docs <https://cgexplore.readthedocs.io/en/v2025.02.05.1/>`_.
  Note, however, that the `recipes <recipes.html>`_ actually reproduce that
  work with the updated interface.


Installation
============

``cgexplore`` can be installed with pip:

.. code-block:: bash

  pip install cgexplore

With dependancies `openmm <https://openmm.org/>`_:

.. code-block:: bash

  mamba install openmm


Then, update directory structure in ``env_set.py`` if using example code.


The library implements some analysis that uses ``Shape 2.1``. Follow the
instructions to download and installed at
`Shape <https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/>`_


Developer Setup
---------------

To develop with ``cgexplore``, you can clone the repo and use
`just <https://github.com/casey/just>`_ and `uv <https://docs.astral.sh>`_
to setup the dev environment:

.. code-block:: bash

  just setup


Usage
=====

We are moving toward implementing a recipe list, which can be found in the
`recipe page <recipes.html>`_.


**To reproduce data in DOI:
`10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_**:
Download the source code from ``first_paper_example - presubmission``
release from ``Releases``.I do not guarantee that running the example code
on the current version will work. However, with each pull request a test is run
as a GitHub Action connected to this
`repository <https://github.com/andrewtarzia/cg_model_test>`_.
This ensures that the results obtained for a subset of the original data set
do not change with changes to this library. Additionally, the naming
convention has changed and force field xml files should provide the
appropriate information for mapping angles to models.


* The directory ``cgexplore`` contains the actual source code for the package.
* The directory ``first_paper_example`` contains the code for `10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_.

  * ``generate_XX.py`` generates cage structures for different topology sets
  * ``env_set.py`` sets a specific environment for file outputs
  * ``plot_XX.py`` produces images and figures, and performs analysis


Important:

  **Warning**: If you have a CUDA-capable GPU and attempt to use CUDA in the
  first example, you may get ``NaN`` errors due to the torsion restriction for
  angles at 180 degrees, which cause problematic forces. This will be handled
  in future versions of the code. And logically, I would suggest removing the
  torsion restriction for those angles. The ``platform`` can be handled through
  this argument in ``build_building_blocks`` and ``build_populations``, which I
  currently set to ``None``, meaning ``OpenMM`` will decide for itself.


How To Cite
===========

If you use ``cgexplore``, please cite this paper

  `https://doi.org/10.26434/chemrxiv-2025-f034c <https://chemrxiv.org/engage/chemrxiv/article-details/68f0ef40bc2ac3a0e051be52>`_

and reference this URL

  https://github.com/andrewtarzia/CGExplore

If you use our minimial model, please cite this paper

  `Systematic exploration of accessible topologies of cage molecules via minimalistic models <https://doi.org/10.1039/D3SC03991A>`_


Publications using CGExplore
============================

* Using stk for constructing larger numbers of coarse-grained models: `Systematic exploration of accessible topologies of cage molecules via minimalistic models <https://doi.org/10.1039/D3SC03991A>`_.
* Starship structure prediction: (`Adjacent backbone interactions control self-sorting of chiral heteroleptic Pd3A2B4 isosceles triangles and Pd4A4C4 pseudo-tetrahedra <https://doi.org/10.1016/j.chempr.2025.102780>`_)
* Structure prediction: (`Predicting stable cage structures by enumerating stoichiometry and topology <https://chemrxiv.org/engage/chemrxiv/article-details/68f0ef40bc2ac3a0e051be52>`_)

Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006 and the ERC under projects DYNAPOL.

This work is now developed as part of the `Tarzia Research Group at the
University of Birmingham <https://tarziaresearchgroup.github.io>`_.
