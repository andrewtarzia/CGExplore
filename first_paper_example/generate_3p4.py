#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging
import sys

from bead_libraries import binder_bead, tetragonal_bead, trigonal_bead
from cgexplore.beads import bead_library_check
from cgexplore.databases import AtomliteDatabase
from cgexplore.molecule_construction import FourC1Arm, ThreeC1Arm
from define_forcefields import define_3p4_forcefield_library
from env_set import calculations, ligands, outputdata, structures
from generation import build_populations
from rdkit import RDLogger
from topologies import cage_topology_options

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def main():
    first_line = f"Usage: {__file__}.py"
    if len(sys.argv) != 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = structures()
    calculation_output = calculations()
    ligand_output = ligands()
    data_output = outputdata()

    # Define bead libraries.
    full_bead_library = (
        tetragonal_bead(),
        binder_bead(),
        trigonal_bead(),
    )
    bead_library_check(full_bead_library)

    logging.info("defining force field")
    forcefieldlibrary = define_3p4_forcefield_library(
        full_bead_library=full_bead_library,
        prefix="3p4",
    )

    logging.info("defining building blocks")
    tetratopic = FourC1Arm(bead=tetragonal_bead(), abead1=binder_bead())
    tritopic = ThreeC1Arm(bead=trigonal_bead(), abead1=binder_bead())

    # Define list of topology functions.
    cage_3p4_topologies = cage_topology_options("3p4")

    # Non-ditopic populations.
    populations = {
        "3p4": {
            "topologies": cage_3p4_topologies,
            "c2": tritopic,
            "cl": tetratopic,
            "fflibrary": forcefieldlibrary,
        },
    }

    database = AtomliteDatabase(db_file=data_output / "first.db")

    build_populations(
        populations=populations,
        struct_output=struct_output,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
        node_element="Pd",
        ligand_element="C",
        platform=None,
        database=database,
    )


if __name__ == "__main__":
    main()
