#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate, optimise all CG models of two-c + three-c systems.

Author: Andrew Tarzia

"""

import sys
import stk
import os
import json
from openmm import openmm
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
from rdkit import RDLogger

from shape import (
    ShapeMeasure,
    get_shape_molecule_ligands,
    get_shape_molecule_nodes,
)
from geom import GeomMeasure
from pore import PoreMeasure
from env_set import cages
from utilities import check_directory
from openmm_optimizer import CGOMMOptimizer
from beads import produce_bead_library, bead_library_check


def c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        sigmas=(2,),
        angles=(180,),
        bond_ks=(10,),
        angle_ks=(20,),
        coordination=2,
    )


def optimise_cage(
    molecule,
    name,
    output_dir,
    bead_set,
    custom_torsion_set,
):

    opt1_mol_file = os.path.join(output_dir, f"{name}_omm1.mol")

    if os.path.exists(opt1_mol_file):
        molecule = molecule.with_structure_from_file(opt1_mol_file)
    else:
        logging.info(f"optimising {name}")

        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=output_dir,
            param_pool=bead_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        molecule = opt.optimize(molecule)
        # molecule = molecule.with_structure_from_file(opt_xyz_file)
        # molecule = molecule.with_centroid((0, 0, 0))
        # molecule.write(opt1_mol_file)

    return molecule


def analyse_cage(
    molecule,
    name,
    output_dir,
    bead_set,
):
    raise NotImplementedError()

    output_file = os.path.join(output_dir, f"{name}_res.json")
    # gulp_output_file2 = os.path.join(output_dir, f"{name}_o2.ginout")
    gulp_output_file3 = os.path.join(output_dir, f"{name}_o3.ginout")
    shape_molfile1 = os.path.join(output_dir, f"{name}_shape1.mol")
    shape_molfile2 = os.path.join(output_dir, f"{name}_shape2.mol")
    # xyz_template = os.path.join(output_dir, f"{name}_temp.xyz")
    # pm_output_file = os.path.join(output_dir, f"{name}_pm.json")

    if molecule is None:
        res_dict = {"optimised": False}
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f"analysing {name}")
        if os.path.exists(gulp_output_file3):
            run_data = None
            # run_data = extract_gulp_optimisation(gulp_output_file3)
        else:
            run_data = None
            # run_data = extract_gulp_optimisation(gulp_output_file2)

        try:
            fin_energy = run_data["final_energy"]
            fin_gnorm = run_data["final_gnorm"]
        except KeyError:
            raise KeyError(
                "final energy not found in run_data, suggests failure "
                f"of {name}"
            )

        n_shape_mol = get_shape_molecule_nodes(molecule, name)
        l_shape_mol = get_shape_molecule_ligands(molecule, name)
        if n_shape_mol is None:
            node_shape_measures = None
        else:
            n_shape_mol.write(shape_molfile1)
            node_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_nshape"),
                target_atmnums=None,
                shape_string=None,
            ).calculate(n_shape_mol)

        if l_shape_mol is None:
            lig_shape_measures = None
        else:
            lig_shape_measures = ShapeMeasure(
                output_dir=(output_dir / f"{name}_lshape"),
                target_atmnums=None,
                shape_string=None,
            ).calculate(l_shape_mol)
            l_shape_mol.write(shape_molfile2)

        opt_pore_data = PoreMeasure().calculate_min_distance(molecule)

        # Always want to extract target torions.
        temp_custom_torsion_set = target_torsions(
            bead_set=bead_set,
            custom_torsion_option=None,
        )

        custom_torsion_atoms = [
            bead_set[j].element_string
            for i in temp_custom_torsion_set
            for j in i
        ]
        g_measure = GeomMeasure(custom_torsion_atoms)
        bond_data = g_measure.calculate_bonds(molecule)
        angle_data = g_measure.calculate_angles(molecule)
        dihedral_data = g_measure.calculate_dihedrals(molecule)
        min_b2b_distance = g_measure.calculate_minb2b(molecule)

        res_dict = {
            "optimised": True,
            "fin_energy": fin_energy,
            "fin_gnorm": fin_gnorm,
            "opt_pore_data": opt_pore_data,
            "lig_shape_measures": lig_shape_measures,
            "node_shape_measures": node_shape_measures,
            "bond_data": bond_data,
            "angle_data": angle_data,
            "dihedral_data": dihedral_data,
            "min_b2b_distance": min_b2b_distance,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f, indent=4)

    return res_dict


def target_torsions(bead_set, custom_torsion_option):
    (c_key,) = (i for i in bead_set if i[0] == "c")
    (t_key_1,) = (i for i in bead_set if i[0] == "a")
    (t_key_2,) = (i for i in bead_set if i[0] == "b")
    custom_torsion_set = {
        (
            t_key_2,
            t_key_1,
            c_key,
            t_key_1,
            t_key_2,
        ): custom_torsion_option,
    }
    return custom_torsion_set


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = cages() / "ommtest"
    check_directory(struct_output)
    figure_output = cages() / "ommfigures"
    check_directory(figure_output)
    calculation_output = cages() / "ommtestcalculations"
    check_directory(calculation_output)

    # Define bead libraries.
    beads = c_beads()
    full_bead_library = list(beads.values())
    bead_library_check(full_bead_library)

    raise SystemExit(
        "you need to think about these tests properly - currently the distances and x axis are wrong"
    )

    bead = beads["c0000"]
    linear_bb = stk.BuildingBlock(
        smiles=f"[{bead.element_string}][{bead.element_string}]",
        position_matrix=[[0, 0, 0], [1, 0, 0]],
    )

    coords = np.linspace(0, 5, 20)
    print(coords)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l1_{i}"
        new_posmat = linear_bb.get_position_matrix() * coord
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        energy = opt.calculate_energy(new_bb)
        print(name, coord, energy)
        xys.append(
            (
                coord,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="k",
        alpha=1.0,
    )
    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance [A]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l1.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{bead.element_string}][{bead.element_string}]"
            f"[{bead.element_string}]"
        ),
        position_matrix=[[0, 0, 0], [2, 0, 0], [3, 0, 0]],
    )

    coords = np.linspace(0, 5, 20)
    print(coords)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l2_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[2] = new_posmat[2] * coord
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            param_pool=beads,
            custom_torsion_set=None,
            bonds=True,
            angles=True,
            torsions=False,
            vdw=False,
        )
        energy = opt.calculate_energy(new_bb)
        print(name, coord, energy)
        xys.append(
            (
                coord,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="k",
        alpha=1.0,
    )
    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance [A]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "l2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    raise SystemExit()

    custom_torsion_options = {
        "ton": (0, 5),
        "toff": None,
    }

    for popn in populations:
        logging.info(f"building {popn} population")
        popn_iterator = itertools.product(
            populations[popn]["t"],
            populations[popn]["c2"],
            populations[popn]["cl"],
        )
        count = 0
        for iteration in popn_iterator:
            for custom_torsion in custom_torsion_options:
                tname = custom_torsion
                cage_topo_str, bb2_str, bbl_str = iteration
                name = f"{cage_topo_str}_{bbl_str}_{bb2_str}_{tname}"
                bb2, bb2_bead_set = populations[popn]["c2"][bb2_str]
                bbl, bbl_bead_set = populations[popn]["cl"][bbl_str]

                bead_set = bb2_bead_set.copy()
                bead_set.update(bbl_bead_set)
                (ba_bead,) = (
                    bb2_bead_set[i] for i in bb2_bead_set if "a" in i
                )
                bite_angle = (ba_bead.angle_centered - 90) * 2
                if custom_torsion_options[custom_torsion] is None:
                    custom_torsion_set = None
                elif bite_angle == 180:
                    # There are no torsions for bite angle == 180.
                    custom_torsion_set = None
                else:
                    tors_option = custom_torsion_options[custom_torsion]
                    custom_torsion_set = target_torsions(
                        bead_set=bead_set,
                        custom_torsion_option=tors_option,
                    )

                cage = stk.ConstructedMolecule(
                    topology_graph=populations[popn]["t"][
                        cage_topo_str
                    ](
                        building_blocks=(bb2, bbl),
                    ),
                )
                save_idealised_topology(
                    cage=cage,
                    cage_topo_str=cage_topo_str,
                    struct_output=struct_output,
                )

                cage = optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_output,
                    bead_set=bead_set,
                    custom_torsion_set=custom_torsion_set,
                )

                if cage is not None:
                    cage.write(str(struct_output / f"{name}_optc.mol"))

                analyse_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_output,
                    bead_set=bead_set,
                )
                count += 1

        logging.info(f"{count} {popn} cages built.")


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
