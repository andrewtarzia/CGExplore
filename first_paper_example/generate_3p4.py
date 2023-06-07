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

from molecule_construction.topologies import ThreeC1Arm, FourC1Arm
from generation_utilities import (
    custom_torsion_definitions,
    custom_vdw_definitions,
    build_building_block,
    build_populations,
    beads_3c,
    beads_4c,
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
    beads_4c_lib = beads_4c()
    beads_3c_lib = beads_3c()
    beads_binder_lib = binder_beads()
    full_bead_library = (
        list(beads_3c_lib.values())
        + list(beads_4c_lib.values())
        + list(beads_binder_lib.values())
    )
    bead_library_check(full_bead_library)

    logging.info("building building blocks")
    c3_blocks = build_building_block(
        topology=ThreeC1Arm,
        option1_lib=beads_3c_lib,
        option2_lib=beads_binder_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )
    c4_blocks = build_building_block(
        topology=FourC1Arm,
        option1_lib=beads_4c_lib,
        option2_lib=beads_binder_lib,
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )

    logging.info(
        f"there are {len(c3_blocks)} 3-C and "
        f"{len(c4_blocks)} 4-C building blocks."
    )

    # Define list of topology functions.
    cage_3p4_topologies = cage_topology_options("3p4")

    # Non-ditopic populations.
    populations = {
        "3p4": {
            "t": cage_3p4_topologies,
            "c2": c3_blocks,
            "cl": c4_blocks,
        },
    }
    custom_torsion_options = custom_torsion_definitions("3p4")
    custom_vdw_options = custom_vdw_definitions("3p4")
    build_populations(
        populations=populations,
        custom_torsion_options=custom_torsion_options,
        custom_vdw_options=custom_vdw_options,
        struct_output=struct_output,
        calculation_output=calculation_output,
        node_element="Pd",
        ligand_element="C",
    )


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
