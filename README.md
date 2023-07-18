# CGExplore
A general toolkit for working with coarse-grained stk models.

# Installation

The code can be installed by cloning this repository and running `pip install -e .` within the top directory. We recommend using a `conda` or `mamba` environment, which can be created from the `environment.yml` and then running `pip install -e .` once that environment is activated. 

The package requires: 

`stk`: Install using pip following [stk](https://stk.readthedocs.io/en/stable/)

`OpenMM`: Install following [OpenMM](https://openmm.org/)

`Shape 2.1`: Follow the instructions to download and installed at [Shape](https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/)

# Usage

* The directory `cgexplore` contains the actual source code for the package. **Warning**: This package is still very much underdevelopment and many changes are expected.
* The directory `first_paper_example` contains the code to generate the cages and data for DOI: XX. This series of examples uses the classes and tools in `CGExplore`.
  * `generate_XX.py` generates cage structures for different topology sets
  * `env_set.py` sets a specific environment for file outputs
  * `plot_XX.py` produces images and figures, and performs analysis

# Acknowledgements

This work was completed during my time as a postdoc, and then research fellow in the Pavan group at PoliTO (https://www.gmpavanlab.com/).
