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
    LennardJones,
    ThreeAngle,
    IntSet,
    CheckedThreeAngle,
)

from mnl2n_construction.topologies import cage_topologies

from mnl2n_construction.plotting import (
    convergence,
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


def run_optimisation(
    cage,
    ff_modifications,
    ffname,
    topo_str,
    output_dir,
):

    run_prefix = f"{topo_str}_{ffname}"
    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f": running optimisation of {run_prefix}")
        opt = MNL2NOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=ff_modifications["param_pool"],
        )
        run_data = opt.optimize(cage)

        # Get cube shape measure.
        opted = cage.with_structure_from_file(
            path=os.path.join(output_dir, f"{run_prefix}_opted.xyz"),
        )
        opted.write(os.path.join(output_dir, f"{run_prefix}_opted.mol"))

        distances = get_distances(optimizer=opt, cage=opted)
        angles = get_angles(optimizer=opt, cage=opted)

        num_steps = len(run_data["traj"])
        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        opt_traj_data = run_data["traj"]
        logging.info(
            f"{run_prefix}: {num_steps} {fin_energy} {fin_gnorm} "
        )

        logging.info("running MD..")
        opt = MNL2NMD(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=ff_modifications["param_pool"],
        )
        md_data = opt.optimize(opted)
        mded = cage.with_structure_from_file(
            path=os.path.join(output_dir, f"{run_prefix}_final.xyz"),
        )
        mded.write(os.path.join(output_dir, f"{run_prefix}_final.mol"))
        print(md_data)
        raise SystemExit()
        md_fin_energy = run_data["final_energy"]
        md_traj_data = md_data["traj"]

        res_dict = {
            "fin_energy": fin_energy,
            "traj": opt_traj_data,
            "distances": distances,
            "angles": angles,
            "md_fin_energy": md_fin_energy,
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

    results = {i: {} for i in bite_angles}
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
            continue
            results[ba][topo_str] = res_dict
        raise SystemExit()

    topo_to_c = {
        "m2l4": ("o", "k", 0),
        "m3l6": ("D", "r", 1),
        "m4l8": ("X", "gold", 2),
        "m6l12": ("o", "skyblue", 3),
        "m12l24": ("P", "b", 4),
        "m24l48": ("X", "green", 5),
    }

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
