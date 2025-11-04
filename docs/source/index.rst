.. toctree::
   :hidden:
   :caption: CGExplore
   :maxdepth: 1

   Recipes <recipes>
   First paper example <first_paper_example>
   Structure prediction <scram>
   Systems optimisation <systems_optimisation>
   Sharing with collaborators <databasing>
   Molecular <molecular>
   OpenMM Interface <optimisation>
   Forcefields <forcefields>
   Terms <terms>
   Analysis <analysis>
   Executables <executables>
   Topologies <topologies>
   Atomistic tools <atomistic>
   Utilities <utilities>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Modules:

  Modules <modules>


============
Introduction
============

| GitHub: https://www.github.com/andrewtarzia/CGExplore


:mod:`cgexplore` or ``cgx`` is a general toolkit built on
`stk <https://stk.readthedocs.io/en/stable/>`_ for constructing,
optimising and exploring molecular coarse-grained models.

.. tip::

  ⭐ Star us on `GitHub <https://www.github.com/andrewtarzia/CGExplore>`_! ⭐

.. figure:: _static/logo.png


.. important::

  **Warning**: This package is still very much underdevelopment and many changes
  are expected.

  In particular, if you are trying to reproduce exactly
  `DOI: 10.26434/chemrxiv-2025-f034c <https://chemrxiv.org/engage/chemrxiv/article-details/68f0ef40bc2ac3a0e051be52>`_,
  then you should install an old version of the code
  (``cgexplore==2025.2.5.1``) alongside the `topology_scrambler <https://github.com/andrewtarzia/topology_scrambler/tree/main>`_
  code and use `these docs <https://cgexplore.readthedocs.io/en/v2025.02.05.1/>`_.
  Note, however, that the `recipes <recipes.html>`_ actually reproduce that
  work with the updated interface.

Installation
============

:mod:`cgexplore` can be installed with pip:

.. code-block:: bash

  pip install cgexplore

With dependancies `openmm <https://openmm.org/>`_:

.. code-block:: bash

  mamba install openmm


Then, update directory structure in `env_set.py` if using example code.


The library implements some analysis that uses `Shape 2.1`. Follow the
instructions to download and installed at
`Shape <https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/>`_


Developer Setup
---------------

To develop with :mod:`cgexplore`, you can clone the repo and use
`just <https://github.com/casey/just>`_ and `uv <https://docs.astral.sh>`_
to setup the dev environment:

.. code-block:: bash

  just setup


Examples
========

We are moving toward implementing a recipe list, which can be found in the
`recipe page <recipes.html>`_.

The main series of examples are in `First Paper Example`_. In that page you
will find all the information necessary to reproduce the work in
`10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_

With each pull request a test is run as a GitHub Action connected to this
`repository <https://github.com/andrewtarzia/cg_model_test>`_.
This ensures that the results obtained for a subset of the original data set do
not change with changes to this library.

.. note::

  `cg_model_test <https://github.com/andrewtarzia/cg_model_test>`_ is a good
  example of usage too!

How To Cite
===========

If you use ``cgexplore``, please cite this paper

  `Predicting stable cage structures by enumerating stoichiometry and topology <https://chemrxiv.org/engage/chemrxiv/article-details/68f0ef40bc2ac3a0e051be52>`_

and reference this URL

  https://github.com/andrewtarzia/CGExplore

If you use our minimial model, please cite this paper

  `Systematic exploration of accessible topologies of cage molecules via minimalistic models <https://doi.org/10.1039/D3SC03991A>`_


Publications using CGExplore
============================

* Using stk for constructing larger numbers of coarse-grained models: `Systematic exploration of accessible topologies of cage molecules via minimalistic models <https://doi.org/10.1039/D3SC03991A>`_.
* Starship structure prediction: (`Adjacent backbone interactions control self-sorting of chiral heteroleptic Pd3A2B4 isosceles triangles and Pd4A4C4 pseudo-tetrahedra <https://doi.org/10.1016/j.chempr.2025.102780>`_)
* Structure prediction: (`Predicting stable cage structures... <https://chemrxiv.org/engage/chemrxiv/article-details/68f0ef40bc2ac3a0e051be52>`_)

Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006 and the ERC under projects DYNAPOL.

This work is now developed as part of the `Tarzia Research Group at the
University of Birmingham <https://tarziaresearchgroup.github.io>`_.

.. _`First Paper Example`: first_paper_example.html
