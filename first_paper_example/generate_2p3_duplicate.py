# Distributed under the terms of the MIT License.

"""Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging

import stk
from bead_libraries import arm_bead, binder_bead, core_bead, trigonal_bead
from define_forcefields import define_forcefield_library
from env_set import cages, ligands
from generation import build_populations
from rdkit import RDLogger

from cgexplore.molecular import ThreeC1Arm, TwoC1Arm
from cgexplore.utilities import AtomliteDatabase, check_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run script."""
    struct_output = cages() / "duplicate_structures"
    check_directory(struct_output)
    calculation_output = cages() / "duplicate_calculations"
    check_directory(calculation_output)
    ligand_output = ligands()
    data_output = cages() / "duplicate_outputdata"
    check_directory(data_output)

    # Define bead libraries.
    present_beads = (
        core_bead(),
        arm_bead(),
        binder_bead(),
        trigonal_bead(),
    )

    logger.info("defining force field")
    forcefieldlibrary = define_forcefield_library(
        present_beads=present_beads,
        prefix="2p3",
    )

    logger.info("defining building blocks")
    ditopic = TwoC1Arm(bead=core_bead(), abead1=arm_bead())
    tritopic = ThreeC1Arm(bead=trigonal_bead(), abead1=binder_bead())

    # Define list of topology functions.
    cage_2p3_topologies = {
        "4P6": stk.cage.FourPlusSix,
        "4P62": stk.cage.FourPlusSix2,
        "8P12": stk.cage.EightPlusTwelve,
    }

    populations = {
        "2p3": {
            "topologies": cage_2p3_topologies,
            "c2": ditopic,
            "cl": tritopic,
            "fflibrary": forcefieldlibrary,
        },
    }

    database = AtomliteDatabase(db_file=data_output / "duplicate.db")

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
