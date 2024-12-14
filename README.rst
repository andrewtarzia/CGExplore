:maintainers:
  `andrewtarzia <https://github.com/andrewtarzia/>`_
:documentation: https://cgexplore.readthedocs.io/en/latest/

.. figure:: docs/source/_static/logo.png


Overview
========

:mod:`cgexplore` or ``cgx`` is a general toolkit built on
`stk <https://stk.readthedocs.io/en/stable/>`_ for constructing,
optimising and exploring molecular coarse-grained models.

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


Usage
=====

**To reproduce data in DOI:
`10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_**:
Download the source code from `first_paper_example - presubmission`
release from ``Releases``.I do not guarantee that running the example code
on the current version will work. However, with each pull request a test is run
as a GitHub Action connected to this
`repository <https://github.com/andrewtarzia/cg_model_test>`_.
This ensures that the results obtained for a subset of the original data set
do not change with changes to this library. Additionally, the naming
convention has changed and force field xml files should provide the
appropriate information for mapping angles to models.


* The directory `cgexplore` contains the actual source code for the package.
* The directory `first_paper_example` contains the code for `10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_.
  * `generate_XX.py` generates cage structures for different topology sets
  * `env_set.py` sets a specific environment for file outputs
  * `plot_XX.py` produces images and figures, and performs analysis

.. important::
  **Warning**: If you have a CUDA-capable GPU and attempt to use CUDA in the
  first example, you may get `NaN` errors due to the torsion restriction for
  angles at 180 degrees, which cause problematic forces. This will be handled
  in future versions of the code. And logically, I would suggest removing the
  torsion restriction for those angles. The `platform` can be handled through
  this argument in `build_building_blocks` and `build_populations`, which I
  currently set to `None`, meaning `OpenMM` will decide for itself.


How To Cite
===========

If you use ``stk`` please cite

  https://github.com/andrewtarzia/CGExplore

and

  https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc03991a

Publications using CGExplore
============================

* Using stk for constructing larger numbers of coarse-grained models: `Systematic exploration of accessible topologies of cage molecules via minimalistic models`__


Acknowledgements
================

Funded by the European Union - Next Generation EU, Mission 4 Component 1
CUP E13C22002930006 and the ERC under projects DYNAPOL.
