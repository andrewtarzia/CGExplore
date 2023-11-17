# CGExplore
A general toolkit for working with coarse-grained models.

The library is built off of [`stk`](https://stk.readthedocs.io/en/stable/), which comes with the pip install

# Installation

`CGExplore` is a library but contains an example library usage in the `first_paper_example` directory. I recommend installing the library with the following instructions and then using the example directory in a separate repository or code base of your own.

**To reproduce data in DOI: [10.1039/D3SC03991A](https://doi.org/10.1039/D3SC03991A)**: Download the source code from `first_paper_example - presubmission` release from [Releases](https://github.com/andrewtarzia/CGExplore/releases) instead of the repo in step 1. I do not guarantee that running the example code on the current version will work. However, with each pull request a test is run as a GitHub Action connected to this [repository](https://github.com/andrewtarzia/cg_model_test). This ensures that the results obtained for a subset of the original data set do not change with changes to this library. Additionally, the naming convention has changed and force field xml files should provide the appropriate information for mapping angles to models.

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

4. From `CGExplore` directory, install pip environment:
```
pip install .
```
or for development, use [just](https://github.com/casey/just) to install a dev environment with:
```
just dev
```

5. Install `OpenMM` [docs](https://openmm.org/):
 ```
mamba install openmm
```
or
```
conda install -c conda-forge openmm
```

6. Install `openmmtools` [docs](https://openmmtools.readthedocs.io/en/stable/gettingstarted.html):
```
mamba install openmmtools
```
or
```
conda config --add channels omnia --add channels conda-forge
conda install openmmtools
```

7. Update directory structure in `env_set.py` if using example code.

The library implements some analysis that uses:

`Shape 2.1`: Follow the instructions to download and installed at [Shape](https://www.iqtc.ub.edu/uncategorised/program-for-the-stereochemical-analysis-of-molecular-fragments-by-means-of-continous-shape-measures-and-associated-tools/)

# Usage

* The directory `cgexplore` contains the actual source code for the package. **Warning**: This package is still very much underdevelopment and many changes are expected.
* The directory `first_paper_example` contains the code to generate the cages and data for DOI: [10.1039/D3SC03991A](https://doi.org/10.1039/D3SC03991A). This series of examples uses the classes and tools in `CGExplore`. I would no longer recommend using it as a perfect example as many changes, including using databasing have been introduced without optimising these scripts. [This testing repository is a good example](https://github.com/andrewtarzia/cg_model_test/blob/main/cg_model_test.py).
  * `generate_XX.py` generates cage structures for different topology sets
  * `env_set.py` sets a specific environment for file outputs
  * `plot_XX.py` produces images and figures, and performs analysis

**Warning**: If you have a CUDA-capable GPU and attempt to use CUDA in the first example, you may get `NaN` errors due to the torsion restriction for angles at 180 degrees, which cause problematic forces. This will be handled in future versions of the code. And logically, I would suggest removing the torsion restriction for those angles. The `platform` can be handled through this argument in `build_building_blocks` and `build_populations`, which I currently set to `None`, meaning `OpenMM` will decide for itself.

# Acknowledgements

This work was completed during my time as a postdoc, and then research fellow in the Pavan group at PoliTO (https://www.gmpavanlab.com/).
