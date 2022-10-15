#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate CG models of MnL2n systems.

Author: Andrew Tarzia

"""

import sys
import stk
import numpy as np
import os
import json
import pore_mapper as pm
import logging

from env_set import (
    mnl2n_figures,
    mnl2n_structures,
)
from utilities import (
    get_distances,
    get_angles,
)
from gulp_optimizer import (
    CGGulpOptimizer,
    CGGulpMD,
    HarmBond,
    ThreeAngle,
    IntSet,
    CheckedThreeAngle,
)

from mnl2n_construction.topologies import cage_topologies

from mnl2n_construction.plotting import (
    convergence,
    md_output_plot,
    scatter,
    geom_distributions,
    heatmap,
    ey_vs_property,
)

from precursor_db.precursors import twoc_bb, fourc_bb


def defined_bonds():
    return (
        HarmBond("N", "Pd", bond_r=2, bond_k=10),
        HarmBond("C", "N", bond_r=2, bond_k=10),
        HarmBond("B", "C", bond_r=3, bond_k=10),
    )


def defined_angles():
    return (
        ThreeAngle("B", "C", "C", theta=180, angle_k=20),
        CheckedThreeAngle(
            "Pd",
            "N",
            "N",
            cut_angle=130,
            min_angle=90,
            max_angle=180,
            angle_k=20,
        ),
        ThreeAngle("C", "B", "N", theta=180, angle_k=20),
        ThreeAngle("N", "C", "Pd", theta=180, angle_k=20),
    )


def defined_pairs():
    return (
        # LennardJones("N", "N", epsilon=0.1, sigma=1),
        # LennardJones("C", "N", epsilon=0.1, sigma=1),
        # LennardJones("B", "N", epsilon=0.1, sigma=1),
        # LennardJones("Pd", "N", epsilon=0.1, sigma=1),
        # LennardJones("C", "C", epsilon=0.1, sigma=1),
        # LennardJones("B", "C", epsilon=0.1, sigma=1),
        # LennardJones("Pd", "C", epsilon=0.1, sigma=1),
        # LennardJones("B", "B", epsilon=0.1, sigma=1),
        # LennardJones("Pd", "B", epsilon=0.1, sigma=1),
        # LennardJones("Pd", "Pd", epsilon=0.1, sigma=1),
    )


class MNL2NOptimizer(CGGulpOptimizer):
    def define_bond_potentials(self):
        bonds = defined_bonds()
        new_bonds = self._update_bonds(bonds)
        return IntSet(new_bonds)

    def define_angle_potentials(self):
        angles = defined_angles()
        new_angles = self._update_angles(angles)
        return IntSet(new_angles)

    def define_vdw_potentials(self):
        pairs = defined_pairs()
        new_pairs = self._update_pairs(pairs)
        return IntSet(new_pairs)


class MNL2NMD(CGGulpMD):
    def define_bond_potentials(self):
        bonds = defined_bonds()
        new_bonds = self._update_bonds(bonds)
        return IntSet(new_bonds)

    def define_angle_potentials(self):
        angles = defined_angles()
        new_angles = self._update_angles(angles)
        return IntSet(new_angles)

    def define_vdw_potentials(self):
        pairs = defined_pairs()
        new_pairs = self._update_pairs(pairs)
        return IntSet(new_pairs)


def calculate_rmsd(init_coords, coords):
    init_coords = np.array(init_coords, dtype=float)
    coords = np.array(coords, dtype=float)
    deviations = init_coords - coords
    N = len(init_coords)
    return np.sqrt(np.sum(deviations * deviations) / N)


def add_traj_rmsd(md_traj_data):
    for step in md_traj_data:
        sdict = md_traj_data[step]
        if step == 0:
            init_coords = sdict["coords"]
            rmsd = 0
        else:
            coords = sdict["coords"]
            rmsd = calculate_rmsd(init_coords, coords)
        md_traj_data[step]["rmsd"] = rmsd
    return md_traj_data


def calculate_pore(xyz_file):

    output_file = xyz_file.replace(".xyz", ".json")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            pore_data = json.load(f)
        return pore_data

    host = pm.Host.init_from_xyz_file(xyz_file)
    host = host.with_centroid([0.0, 0.0, 0.0])

    # Define calculator object.
    logging.warning(
        "currently using very small pore, would want to use normal "
        "size in future."
    )
    calculator = pm.Inflater(bead_sigma=0.5)
    # Run calculator on host object, analysing output.
    logging.info(f"calculating pore of {xyz_file}...")
    final_result = calculator.get_inflated_blob(host=host)
    pore = final_result.pore
    blob = final_result.pore.get_blob()
    windows = pore.get_windows()
    pore_data = {
        "step": final_result.step,
        "num_movable_beads": final_result.num_movable_beads,
        "windows": windows,
        "blob_max_diam": blob.get_maximum_diameter(),
        "pore_max_rad": pore.get_maximum_distance_to_com(),
        "pore_mean_rad": pore.get_mean_distance_to_com(),
        "pore_volume": pore.get_volume(),
        "asphericity": pore.get_asphericity(),
        "shape": pore.get_relative_shape_anisotropy(),
    }
    with open(output_file, "w") as f:
        json.dump(pore_data, f)
    return pore_data


def add_traj_pore(
    name,
    input_xyz_template,
    optimizer,
    md_traj_data,
    output_dir,
):
    atom_types = optimizer.get_xyz_atom_types(input_xyz_template)
    for step in md_traj_data:
        sdict = md_traj_data[step]
        xyz_file = os.path.join(output_dir, f"{name}_{step}.xyz")
        optimizer.write_conformer_xyz_file(
            ts=step,
            ts_data=sdict,
            filename=xyz_file,
            atom_types=atom_types,
        )
        pore_data = calculate_pore(xyz_file)
        md_traj_data[step]["windows"] = pore_data["windows"]
        md_traj_data[step]["num_windows"] = len(pore_data["windows"])
        md_traj_data[step]["blob_max_diam"] = pore_data["blob_max_diam"]
        md_traj_data[step]["pore_max_rad"] = pore_data["pore_max_rad"]
        md_traj_data[step]["pore_mean_rad"] = pore_data["pore_mean_rad"]
        md_traj_data[step]["pore_volume"] = pore_data["pore_volume"]
        md_traj_data[step]["asphericity"] = pore_data["asphericity"]
        md_traj_data[step]["shape"] = pore_data["shape"]
    return md_traj_data


def run_optimisation(
    cage,
    ff_modifications,
    ffname,
    topo_str,
    output_dir,
):

    run_prefix = f"{topo_str}_{ffname}"
    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")
    opt_xyz_file = os.path.join(output_dir, f"{run_prefix}_opted.xyz")
    md_xyz_file = os.path.join(output_dir, f"{run_prefix}_final.xyz")
    opt_mol_file = os.path.join(output_dir, f"{run_prefix}_opted.mol")
    md_mol_file = os.path.join(output_dir, f"{run_prefix}_final.mol")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        if not os.path.exists(opt_xyz_file):
            logging.info(f": running optimisation of {run_prefix}")
            opt = MNL2NOptimizer(
                fileprefix=run_prefix,
                output_dir=output_dir,
                param_pool=ff_modifications["param_pool"],
            )
            run_data = opt.optimize(cage)
        opted = cage.with_structure_from_file(opt_xyz_file)
        opted.write(opt_mol_file)

        distances = get_distances(optimizer=opt, cage=opted)
        angles = get_angles(optimizer=opt, cage=opted)

        num_steps = len(run_data["traj"])
        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        opt_traj_data = run_data["traj"]
        logging.info(
            f"{run_prefix}: {num_steps} {fin_energy} {fin_gnorm} "
        )

        try:
            if not os.path.exists(md_xyz_file):
                logging.info("running MD..")
                opt = MNL2NMD(
                    fileprefix=run_prefix,
                    output_dir=output_dir,
                    param_pool=ff_modifications["param_pool"],
                )
                md_data = opt.optimize(opted)
            mded = cage.with_structure_from_file(md_xyz_file)
            mded.write(md_mol_file)
            md_traj_data = md_data["traj"]
            md_traj_data = add_traj_rmsd(md_traj_data)
        except ValueError:
            logging.info(f"MD of {run_prefix} failed.")
            md_traj_data = {}

        res_dict = {
            "fin_energy": fin_energy,
            "traj": opt_traj_data,
            "distances": distances,
            "angles": angles,
            "mdtraj": md_traj_data,
        }
        with open(output_file, "w") as f:
            json.dump(res_dict, f)

    return res_dict


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = mnl2n_structures()
    figure_output = mnl2n_figures()

    # Make cage of each symmetry.
    topologies = cage_topologies(fourc_bb(), twoc_bb(sites=5))
    bite_angles = np.arange(0, 181, 10)
    ff_options = {}
    for ba in bite_angles:
        ff_options[f"ba{ba}"] = {
            "param_pool": {
                "bonds": {},
                "angles": {
                    ("C", "B", "N"): (20, (ba / 2) + 90),
                },
                "torsions": {},
                "pairs": {},
            },
            "notes": f"bite-angle change: {ba}, rigid",
            "name": f"ba{ba}",
        }

    results = {i: {} for i in ff_options}
    for topo_str in topologies:
        topology_graph = topologies[topo_str]
        cage = stk.ConstructedMolecule(topology_graph)
        cage.write(os.path.join(struct_output, f"{topo_str}_unopt.mol"))

        for ff_str in ff_options:
            res_dict = run_optimisation(
                cage=cage,
                ffname=ff_str,
                ff_modifications=ff_options[ff_str],
                topo_str=topo_str,
                output_dir=struct_output,
            )
            results[ff_str][topo_str] = res_dict

    topo_to_c = {
        "m2l4": ("o", "k", 0),
        "m3l6": ("D", "r", 1),
        "m4l8": ("X", "gold", 2),
        "m6l12": ("o", "skyblue", 3),
        "m12l24": ("P", "b", 4),
        "m24l48": ("X", "green", 5),
    }

    md_y_columns = ("E", "T", "KE", "rmsd")
    for ycol in md_y_columns:
        md_output_plot(
            topo_to_c=topo_to_c,
            results=results,
            output_dir=figure_output,
            filename=f"md_output_{ycol}.pdf",
            y_column=ycol,
        )

    convergence(
        results=results,
        output_dir=figure_output,
        filename="convergence.pdf",
    )

    ey_vs_property(
        results=results,
        output_dir=figure_output,
        filename="e_vs_shape.pdf",
    )

    geom_distributions(
        results=results,
        output_dir=figure_output,
        filename="dist.pdf",
    )

    heatmap(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="energy_map.pdf",
        vmin=0,
        vmax=5,
        clabel="energy (eV)",
    )

    scatter(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="energy.pdf",
        ylabel="energy (eV)",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
