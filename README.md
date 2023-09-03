# CGExplore
A general toolkit for working with coarse-grained models.

The library is built off of [`stk`](https://stk.readthedocs.io/en/stable/), which comes with the pip install

# Installation

`CGExplore` is a library but contains an example library usage in the `first_paper_example` directory. I recommend installing the library with the following instructions and then using the example directory in a separate repository or code base of your own.

The code can be installed following these steps:

1. clone `CGExplore` from [here](https://github.com/andrewtarzia/CGExplore)

2. Create a `conda` or `mamba` environment:
 ```
 mamba create -n NAME python=3.11
 ```

3. Activate the environment:
 ```
 conda activate NAME
 ```

5. From `CGExplore` directory, install pip environment:
```
pip install .
```
or for development,
```
pip install -e .
```

6. Install `OpenMM` [docs](https://openmm.org/):
 ```
mamba install openmm
```
or
```
conda install -c conda-forge openmm
```

5. Install `openmmtools` [docs](https://openmmtools.readthedocs.io/en/stable/gettingstarted.html):
```
mamba install openmmtools
```
or
```
conda config --add channels omnia --add channels conda-forge
conda install openmmtools
```

The library implements some analysis that uses:

`Shape 2.1`: Follow the instructions to download and installed at [Shape](https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/)

# Usage

* The directory `cgexplore` contains the actual source code for the package. **Warning**: This package is still very much underdevelopment and many changes are expected.
* The directory `first_paper_example` contains the code to generate the cages and data for DOI: XX. This series of examples uses the classes and tools in `CGExplore`.
  * `generate_XX.py` generates cage structures for different topology sets
  * `env_set.py` sets a specific environment for file outputs
  * `plot_XX.py` produces images and figures, and performs analysis

**Warning**: If you have a CUDA-capable GPU and attempt to use CUDA in the first example, you may get `NaN` errors due to the torsion restriction for angles at 180 degrees, which cause problematic forces. This will be handled in future versions of the code. And logically, I would suggest removing the torsion restriction for those angles. The `platform` can be handled through this argument in `build_building_blocks` and `build_populations`, which I currently set to `None`, meaning `OpenMM` will decide for itself.

# Acknowledgements

This work was completed during my time as a postdoc, and then research fellow in the Pavan group at PoliTO (https://www.gmpavanlab.com/).
