#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import sys
import logging
from rdkit import RDLogger

from env_set import cages
from utilities import check_directory
from cage_construction.topologies import cage_topology_options
from beads import bead_library_check

from precursor_db.topologies import TwoC1Arm, ThreeC1Arm
from generation_utilities import (
    custom_torsion_definitions,
    custom_vdw_definitions,
    build_building_block,
    build_populations,
    beads_3c,
    arm_2c_beads,
    core_2c_beads,
    binder_beads,
)


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = cages() / "ommstructures"
    check_directory(struct_output)
    figure_output = cages() / "ommfigures"
    check_directory(figure_output)
    calculation_output = cages() / "ommcalculations"
    check_directory(calculation_output)
    ligand_output = cages() / "ommligands"
    check_directory(ligand_output)

    # Define bead libraries.
    beads_core_2c_lib = core_2c_beads()
    beads_3c_lib = beads_3c()
    beads_arm_2c_lib = arm_2c_beads()
    beads_binder_lib = binder_beads()
    full_bead_library = (
        list(beads_3c_lib.values())
        + list(beads_arm_2c_lib.values())
        + list(beads_core_2c_lib.values())
        + list(beads_binder_lib.values())
    )
    bead_library_check(full_bead_library)

    logging.info("building building blocks")
    c2_blocks = build_building_block(
        topology=TwoC1Arm,
        option1_lib=beads_core_2c_lib,
        option2_lib=beads_arm_2c_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )
    c3_blocks = build_building_block(
        topology=ThreeC1Arm,
        option1_lib=beads_3c_lib,
        option2_lib=beads_binder_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )

    logging.info(
        f"there are {len(c2_blocks)} 2-C and "
        f"{len(c3_blocks)} 3-C and building blocks."
    )

    # Define list of topology functions.
    cage_3p2_topologies = cage_topology_options("2p3")

    populations = {
        "2p3": {
            "t": cage_3p2_topologies,
            "c2": c2_blocks,
            "cl": c3_blocks,
        },
    }
    custom_torsion_options = custom_torsion_definitions("2p3")
    custom_vdw_options = custom_vdw_definitions("2p3")
    build_populations(
        populations=populations,
        custom_torsion_options=custom_torsion_options,
        custom_vdw_options=custom_vdw_options,
        struct_output=struct_output,
        calculation_output=calculation_output,
    )


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
