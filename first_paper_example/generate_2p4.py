#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging
import sys

from bead_libraries import arm_bead, binder_bead, core_bead, tetragonal_bead
from cgexplore.beads import bead_library_check
from cgexplore.databases import AtomliteDatabase
from cgexplore.molecule_construction import FourC1Arm, TwoC1Arm
from define_forcefields import define_2p4_forcefield_library
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
        core_bead(),
        arm_bead(),
        binder_bead(),
        tetragonal_bead(),
    )
    bead_library_check(full_bead_library)

    logging.info("defining force field")
    forcefieldlibrary = define_2p4_forcefield_library(
        full_bead_library=full_bead_library,
        prefix="2p4",
    )

    logging.info("defining building blocks")
    ditopic = TwoC1Arm(bead=core_bead(), abead1=arm_bead())
    tetratopic = FourC1Arm(bead=tetragonal_bead(), abead1=binder_bead())

    # Define list of topology functions.
    cage_2p4_topologies = cage_topology_options("2p4")

    populations = {
        "2p4": {
            "topologies": cage_2p4_topologies,
            "c2": ditopic,
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
        ligand_element="Ag",
        platform=None,
        database=database,
    )


if __name__ == "__main__":
    main()
