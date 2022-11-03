#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to optimise CG models of fourplussix host-guest systems.

Author: Andrew Tarzia

"""

import sys
import stk
import stko
import os
import json
from collections import Counter
import logging
import numpy as np
from scipy.spatial.distance import cdist
import spindry as spd
from itertools import combinations

from shape import ShapeMeasure

from env_set import (
    fourplussix_figures,
    fourplussix_optimisation,
    fourplussix_calculations,
    fourplussix_backmap,
    guest_structures,
    gulp_path,
)

from gulp_optimizer import CGGulpOptimizer

from fourplusix_construction.topologies import cage_topology_options

from precursor_db.precursors import (
    three_precursor_topology_options,
    two_precursor_topology_options,
)

from beads import (
    core_beads,
    beads_2c,
    beads_3c,
    guest_beads,
)

from ea_module import (
    CgEvolutionaryAlgorithm,
    RecordFitnessFunction,
)
from ea_plotters import (
    CgProgressPlotter,
    plot_existing_guest_data_distributions,
)
from fourplussix_model_optimisation import get_molecule_name_from_record


class FixedGulpUFFOptimizer(stko.GulpUFFOptimizer):
    def _position_section(self, mol, type_translator):
        position_section = "\ncartesian\n"
        count = 0
        for atom in mol.get_atoms():
            atom_type = type_translator[
                self.atom_labels[atom.get_id()][0]
            ]
            position = mol.get_centroid(atom_ids=atom.get_id())

            if atom.get_id() in self._fixed_ids:
                count += 1
                position_section += (
                    f"{atom_type} core {round(position[0], 5)} "
                    f"{round(position[1], 5)} {round(position[2], 5)} "
                    "0 0 0 \n"
                )
            else:

                position_section += (
                    f"{atom_type} core {round(position[0], 5)} "
                    f"{round(position[1], 5)} {round(position[2], 5)} "
                    "1 1 1\n"
                )

        logging.info(f"OPT: fixing {count} atoms.")
        return position_section

    def _write_gulp_file(
        self,
        mol,
        metal_atoms,
        in_file,
        output_xyz,
        unit_cell=None,
    ):

        type_translator = self._type_translator()

        top_line = "opti "

        if self._conjugate_gradient:
            top_line += "conj unit "

        if unit_cell is not None:
            # Constant pressure.
            top_line += "conp "
            cell_section = self._cell_section(unit_cell)
            # Output CIF.
            output_cif = output_xyz.replace("xyz", "cif")
            periodic_output = f"output cif {output_cif}\n"
        else:
            # Constant volume.
            top_line += " "
            cell_section = ""
            periodic_output = ""

        top_line += "noautobond fix molmec cartesian\n"

        position_section = self._position_section(mol, type_translator)
        bond_section = self._bond_section(mol, metal_atoms)
        species_section = self._species_section(type_translator)

        library = "\nlibrary uff4mof.lib\n"

        output_section = (
            "\n"
            f"maxcyc {self._maxcyc}\n"
            "terse inout potentials\n"
            "terse in cell\n"
            "terse in structure\n"
            "terse inout derivatives\n"
            f"output xyz {output_xyz}\n"
            f"{periodic_output}"
            # 'output movie xyz steps_.xyz\n'
        )

        with open(in_file, "w") as f:
            f.write(top_line)
            f.write(cell_section)
            f.write(position_section)
            f.write(bond_section)
            f.write(species_section)
            f.write(library)
            f.write(output_section)

    def fix_atoms(self, mol, fixed_ids):
        self._fixed_ids = fixed_ids


def get_a_cage():

    # # Define list of topology functions.
    cage_topologies = cage_topology_options()

    bb_2c = stk.BuildingBlock.init_from_file(
        path="temp_bbc2_2.mol",
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts=("[Bi][Mn]"),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        ),
    )

    bb_3c = stk.BuildingBlock.init_from_file(
        path="temp_bbc3_2.mol",
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts=("[Bi][Ho]"),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        ),
    )

    const_mol = stk.ConstructedMolecule(
        topology_graph=cage_topologies["FourPlusSix"](
            building_blocks=(bb_2c, bb_3c),
        ),
    )
    return const_mol

    # generator = np.random.RandomState(23)
    # # Define precursor topologies.
    # three_precursor_topologies = three_precursor_topology_options()
    # two_precursor_topologies = two_precursor_topology_options()

    # for i in range(10000):

    #     selected_cage_topology = "FourPlusSix"

    #     s_2c_topology = "2c-1"

    #     bb_2c_template = two_precursor_topologies[s_2c_topology]
    #     bb_2c = bb_2c_template.get_building_block(
    #         core_bead_lib=core_beads(),
    #         bead_2c_lib=beads_2c(),
    #         generator=generator,
    #     )

    #     s_3c_topology = "3c-1"

    #     bb_3c_template = three_precursor_topologies[s_3c_topology]
    #     bb_3c = bb_3c_template.get_building_block(
    #         bead_2c_lib=beads_2c(),
    #         bead_3c_lib=beads_3c(),
    #         generator=generator,
    #     )
    #     mol_rec = stk.MoleculeRecord(
    #         topology_graph=cage_topologies[selected_cage_topology](
    #             building_blocks=(bb_2c, bb_3c),
    #         ),
    #     )
    #     mol_name = get_molecule_name_from_record(mol_rec)
    #     print(mol_name)
    #     if mol_name == "FourPlusSix_Bi24Ho4Mn6":
    #         print(bb_2c)
    #         print(bb_3c)
    #         bb_3c.write("temp_bbc3_2.mol")
    #         bb_2c.write("temp_bbc2_2.mol")
    #         return mol_rec

    # print("failed")
    # raise SystemExit()


def main():
    first_line = f"Usage: {__file__}.py "
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = fourplussix_backmap()
    cage_struct_output = fourplussix_optimisation()
    figure_output = fourplussix_figures()
    calculation_output = fourplussix_calculations()

    # test_bb2 = stk.BuildingBlock(
    #     smiles="C1C(N([H])[H])=CC=C(C2C=CC(N([H])[H])=CC=2)C=1",
    #     functional_groups=(stk.PrimaryAminoFactory(),),
    # )
    # test_bb3 = stk.BuildingBlock(
    #     smiles="C1C(C=O)=CC(C=O)=CC=1C=O",
    #     functional_groups=(stk.AldehydeFactory(),),
    # )
    test_bb2 = stk.BuildingBlock(
        smiles="NCCN",
        functional_groups=(stk.PrimaryAminoFactory(),),
    )
    test_bb3 = stk.BuildingBlock(
        smiles="C1C(C=O)=CC(C=O)=CC=1C=O",
        functional_groups=(stk.AldehydeFactory(),),
    )
    test_bb2.write("test_bb2_2.mol")
    test_bb3.write("test_bb3_2.mol")
    topology_graph = stk.cage.FourPlusSix((test_bb2, test_bb3))
    construction_result = topology_graph.construct()
    test_cage = stk.ConstructedMolecule.init_from_construction_result(
        construction_result=construction_result,
    ).with_centroid((0, 0, 0))
    test_cage.write("test_cage_2.mol")

    temp_const_mol = get_a_cage()
    temp_const_mol.write("temp_constmol_2.mol")

    temp_host_molecule = stk.BuildingBlock.init_from_file(
        str(
            cage_struct_output
            # / "g_99_m_0_FourPlusSix_Ce12Eu6Ir4Sb12.mol"
            / "g_15_m_0_FourPlusSix_Bi24Ho4Mn6.mol"
        )
    ).with_centroid((0, 0, 0))
    temp_host_molecule.write("temp_host_2.mol")

    cg_bb_ids = {}
    for atom_info in temp_const_mol.get_atom_infos():
        bb_id = atom_info.get_building_block_id()
        if bb_id not in cg_bb_ids:
            cg_bb_ids[bb_id] = []
        cg_bb_ids[bb_id].append(atom_info)

    print(cg_bb_ids.keys())

    cg_bonded_atoms = []
    for bond_info in temp_const_mol.get_bond_infos():
        if bond_info.get_building_block() is not None:
            continue
        cg_bonded_atoms.append(bond_info.get_bond().get_atom1())
        cg_bonded_atoms.append(bond_info.get_bond().get_atom2())

    print(cg_bonded_atoms)

    bb_ids = {}
    for atom_info in test_cage.get_atom_infos():
        bb_id = atom_info.get_building_block_id()
        if bb_id not in bb_ids:
            bb_ids[bb_id] = []
        bb_ids[bb_id].append(atom_info)

    print(bb_ids.keys())

    bonded_atoms = []
    for bond_info in test_cage.get_bond_infos():
        if bond_info.get_building_block() is not None:
            continue
        bonded_atoms.append(bond_info.get_bond().get_atom1())
        bonded_atoms.append(bond_info.get_bond().get_atom2())

    print(bonded_atoms)
    bonded_atom_ids = tuple(i.get_id() for i in bonded_atoms)
    other_atom_ids = tuple(
        i.get_id()
        for i in test_cage.get_atoms()
        if i.get_id() not in bonded_atom_ids
    )

    cg_pos_mat = temp_host_molecule.get_position_matrix()
    for bb_id in bb_ids:
        print(bb_id)
        bb_ais = bb_ids[bb_id]
        bb_a_ids = tuple(i.get_atom().get_id() for i in bb_ais)
        bb_ = bb_ais[0].get_building_block()
        print(bb_)
        cg_ais = cg_bb_ids[bb_id]
        cg_a_ids = tuple(i.get_atom().get_id() for i in cg_ais)
        cg_bb_ = cg_ais[0].get_building_block()
        print(cg_bb_)

        bb_connectors = tuple(
            i for i in bonded_atoms if i.get_id() in bb_a_ids
        )
        bb_connectors_ids = tuple(i.get_id() for i in bb_connectors)
        cg_connectors = tuple(
            i for i in cg_bonded_atoms if i.get_id() in cg_a_ids
        )
        print(bb_connectors)
        print(cg_connectors)

        bb_nonconnectors = tuple(
            i
            for i in bb_ais
            if i.get_atom().get_id() not in bonded_atoms
        )
        cg_nonconnectors = tuple(
            i
            for i in cg_ais
            if i.get_atom().get_id() not in cg_bonded_atoms
        )
        print(bb_nonconnectors, cg_nonconnectors)

        # Set position of bb_connectors to cg_connectors.
        new_posmat = []
        for i, pos in enumerate(test_cage.get_position_matrix()):
            if i not in bb_connectors_ids:
                new_posmat.append(pos)
            else:
                bb_conn = bb_connectors[bb_connectors_ids.index(i)]
                cg_conn = cg_connectors[bb_connectors_ids.index(i)]
                cg_conn_pos = cg_pos_mat[cg_conn.get_id()]
                new_posmat.append(cg_conn_pos)
        new_posmat = np.array(new_posmat)
        test_cage = test_cage.with_position_matrix(new_posmat)

    test_cage.write("updated_2.mol")

    opt = FixedGulpUFFOptimizer(
        gulp_path=gulp_path(),
        maxcyc=500,
        metal_FF={},
        output_dir=os.path.join(
            fourplussix_calculations(),
            "test_uffopt1_2",
        ),
        conjugate_gradient=True,
    )
    opt.assign_FF(test_cage)
    opt.fix_atoms(
        mol=test_cage,
        fixed_ids=bonded_atom_ids,
    )
    test_cage = opt.optimize(test_cage)
    test_cage.write("updated_optimised1_2.mol")

    opt = FixedGulpUFFOptimizer(
        gulp_path=gulp_path(),
        maxcyc=500,
        metal_FF={},
        output_dir=os.path.join(
            fourplussix_calculations(),
            "test_uffopt2_2",
        ),
        conjugate_gradient=True,
    )
    opt.assign_FF(test_cage)
    opt.fix_atoms(
        mol=test_cage,
        fixed_ids=other_atom_ids,
    )
    test_cage = opt.optimize(test_cage)
    test_cage.write("updated_optimised2_2.mol")

    opt = FixedGulpUFFOptimizer(
        gulp_path=gulp_path(),
        maxcyc=500,
        metal_FF={},
        output_dir=os.path.join(
            fourplussix_calculations(),
            "test_uffopt3_2",
        ),
        conjugate_gradient=True,
    )
    opt.assign_FF(test_cage)
    opt.fix_atoms(
        mol=test_cage,
        fixed_ids=bonded_atom_ids,
    )
    test_cage = opt.optimize(test_cage)
    test_cage.write("updated_optimised3_2.mol")

    print(
        # "you have AA as ConstructedMolecule..."
        # "you have CG as ConstructedMolecule...\n\n"
        # "This gives you all BB ids, plus AtomInfos- \n\n"
        # "you could technically do MolRecord as ConstructedMolecule\n\n"
        # "What needs doing is - match BB_id=0 in AA with BB_id=0 in CG\n\n"
        # "Set positions of conneciton atoms of AA BB_ID to CG postions\n\n"
        "Set Centroid of AA BB to match CG BB\n\n"
        # "Optimise -- this requires a UFF Gulp optimiser with "
        # "opt.fix_atoms(atom_ids=()) as a new method, that fixes the "
        # "placed atoms.\n\n"
        "Calculate strain relative to input?\n\n"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
