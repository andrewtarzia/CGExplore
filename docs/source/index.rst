.. toctree::
   :hidden:
   :caption: CGExplore
   :maxdepth: 2

   Analysis <analysis>
   Sharing with collaborators <databasing>
   Forcefields <forcefields>
   Molecular <molecular>
   Optimisation <optimisation>
   Systems optimisation <systems_optimisation>
   Terms <terms>
   Utilities <utilities>
   Executables <executables>
   Topologies <topologies>
   Scrambler <scram>
   Atomistic tools <atomistic>
   First paper example <first_paper_example>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Modules:

  Modules <modules>


.. tip::

  ⭐ Star us on `GitHub <https://www.github.com/andrewtarzia/CGExplore>`_! ⭐

============
Introduction
============

| GitHub: https://www.github.com/andrewtarzia/CGExplore


:mod:`cgexplore` or ``cgx`` is a general toolkit built on
`stk <https://stk.readthedocs.io/en/stable/>`_ for constructing,
optimising and exploring molecular coarse-grained models.

.. figure:: _static/logo.png


.. important::

  **Warning**: This package is still very much underdevelopment and many changes
  are expected.

Installation
============

:mod:`cgexplore` can be installed with pip:

.. code-block:: bash

  pip install cgexplore

With dependancies `openmm <https://openmm.org/>`_ and `openmmtools <https://openmmtools.readthedocs.io/en/stable/gettingstarted.html>`_:

.. code-block:: bash

  mamba install openmm openmmtools


Then, update directory structure in `env_set.py` if using example code.


The library implements some analysis that uses `Shape 2.1`. Follow the
instructions to download and installed at
`Shape <https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/>`_


Developer Setup
---------------

To develop with :mod:`cgexplore`, you can clone the repo and use
`just <https://github.com/casey/just>`_ to setup the dev environment:

.. code-block:: bash

  just dev


Examples
========


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


New works done with :mod:`cgexplore`:

* TBC.


Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006 and the ERC under projects DYNAPOL.

.. _`First Paper Example`: first_paper_example.html
