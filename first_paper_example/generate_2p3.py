# Distributed under the terms of the MIT License.

"""Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging

from bead_libraries import arm_bead, binder_bead, core_bead, trigonal_bead
from cgexplore.molecular import ThreeC1Arm, TwoC1Arm
from cgexplore.utilities import AtomliteDatabase
from define_forcefields import define_forcefield_library
from env_set import calculations, ligands, outputdata, structures
from generation import build_populations
from rdkit import RDLogger
from topologies import cage_topology_options

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def main() -> None:
    """Run script."""
    struct_output = structures()
    calculation_output = calculations()
    ligand_output = ligands()
    data_output = outputdata()

    # Define bead libraries.
    present_beads = (
        core_bead(),
        arm_bead(),
        binder_bead(),
        trigonal_bead(),
    )

    logging.info("defining force field")
    forcefieldlibrary = define_forcefield_library(
        present_beads=present_beads,
        prefix="2p3",
    )

    logging.info("defining building blocks")
    ditopic = TwoC1Arm(bead=core_bead(), abead1=arm_bead())
    tritopic = ThreeC1Arm(bead=trigonal_bead(), abead1=binder_bead())

    # Define list of topology functions.
    cage_2p3_topologies = cage_topology_options("2p3")

    populations = {
        "2p3": {
            "topologies": cage_2p3_topologies,
            "c2": ditopic,
            "cl": tritopic,
            "fflibrary": forcefieldlibrary,
        },
    }

    database = AtomliteDatabase(db_file=data_output / "first.db")

    build_populations(
        populations=populations,
        struct_output=struct_output,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
        node_element="C",
        ligand_element="Ag",
        platform=None,
        database=database,
    )


if __name__ == "__main__":
    main()
