#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import sys

from cgexplore.beads import bead_library_check
from cgexplore.molecule_construction.topologies import (
    TwoC1Arm,
    ThreeC1Arm,
)
from cgexplore.generation_utilities import build_building_block

from topologies import cage_topology_options
from generation import (
    custom_torsion_definitions,
    custom_vdw_definitions,
    build_populations,
)
from bead_libraries import (
    beads_3c,
    arm_2c_beads,
    core_2c_beads,
    binder_beads,
)
from env_set import structures, calculations, ligands

import logging
from rdkit import RDLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = structures()
    calculation_output = calculations()
    ligand_output = ligands()

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
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )
    c3_blocks = build_building_block(
        topology=ThreeC1Arm,
        option1_lib=beads_3c_lib,
        option2_lib=beads_binder_lib,
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
        node_element="C",
        ligand_element="Ag",
    )


if __name__ == "__main__":
    main()
