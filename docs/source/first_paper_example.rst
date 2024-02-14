First Paper Example
===================

.. attention::

    This page is very much underdevelopment.

.. important::

  **Warning**: If you have a CUDA-capable GPU and attempt to use CUDA in the
  first example, you may get `NaN` errors due to the torsion restriction for
  angles at 180 degrees, which cause problematic forces. This will be handled
  in future versions of the code. And logically, I would suggest removing the
  torsion restriction for those angles. The `platform` can be handled through
  this argument in `build_building_blocks` and `build_populations`, which I
  currently set to `None`, meaning `OpenMM` will decide for itself.


:mod:`.CGExplore` is a library that can be used very broadly on many systems.
However, my first usage on the systematic modelling of cage molecules is
described here and the code is available in the directory `first_paper_example`.
This code should reproduce the data in `10.1039/D3SC03991A <https://doi.org/10.1039/D3SC03991A>`_.
However, as :mod:`.CGExplore` is updated, some deviations may occur.
With each pull request a test is run as a GitHub Action connected to this
`repository <https://github.com/andrewtarzia/cg_model_test>`_. This ensures that
the results obtained for a subset of the original data set do not change with
changes to this library. Additionally, the naming convention has changed and
forcefield `.xml` files should provide the appropriate information for mapping
angles to models.

.. note::
    The main change is the use of `Atomlite` databasing!


I recommend installing the :mod:`.CGExplore` with the following instructions
and then using the example directory in a separate repository or code base of
your own.

Download the source code from `first_paper_example - presubmission` release
from `Releases <https://github.com/andrewtarzia/CGExplore/releases>`_ instead
of the `main` repository. Then follow the instructions on the main page.


The workflow:

* `generate_XX.py` generates cage structures for different topology sets
* `env_set.py` sets a specific environment for file outputs
* `plot_XX.py` produces images and figures, and performs analysis


