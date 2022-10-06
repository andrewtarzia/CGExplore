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
from gulp_optimizer import CGGulpOptimizer

from mnl2n_construction.topologies import cage_topologies

from mnl2n_construction.plotting import (
    convergence,
    scatter,
    geom_distributions,
    heatmap,
    ey_vs_property,
)

from precursor_db.precursors import twoc_bb, fourc_bb


class MNL2NOptimizer(CGGulpOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        biteangle,
    ):
        self._biteangle = float(biteangle)
        super().__init__(fileprefix, output_dir)

    def define_bond_potentials(self):
        bond_ks_ = {
            ("N", "Pd"): 10,
            ("C", "N"): 10,
            ("B", "C"): 10,
        }
        bond_rs_ = {
            ("N", "Pd"): 2,
            ("C", "N"): 2,
            ("B", "C"): 3,
        }
        return bond_ks_, bond_rs_

    def define_angle_potentials(self):
        angle_ks_ = {
            ("B", "C", "C"): 20,
            ("N", "N", "Pd"): 20,
            ("B", "C", "N"): 20,
            ("C", "N", "Pd"): 20,
        }
        angle_thetas_ = {
            ("B", "C", "C"): 180,
            ("N", "N", "Pd"): (
                "check",
                {"cut": 130, "min": 90, "max": 180},
            ),
            ("B", "C", "N"): (self._biteangle / 2) + 90,
            ("C", "N", "Pd"): 180,
        }

        return angle_ks_, angle_thetas_


def run_optimisation(cage, biteangle, topo_str, output_dir):

    run_prefix = f"{topo_str}_{biteangle}"
    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f": running optimisation of {run_prefix}")
        opt = MNL2NOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            biteangle=biteangle,
        )
        run_data = opt.optimize(cage)

        # Get cube shape measure.
        opted = cage.with_structure_from_file(
            path=os.path.join(output_dir, f"{run_prefix}_final.xyz"),
        )
        opted.write(os.path.join(output_dir, f"{run_prefix}_final.mol"))
        distances = get_distances(optimizer=opt, cage=opted)
        angles = get_angles(optimizer=opt, cage=opted)

        num_steps = len(run_data["traj"])
        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        traj_data = run_data["traj"]
        logging.info(
            f"{run_prefix}: {num_steps} {fin_energy} {fin_gnorm} "
        )
        res_dict = {
            "fin_energy": fin_energy,
            "traj": traj_data,
            "distances": distances,
            "angles": angles,
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

    results = {i: {} for i in bite_angles}
    for topo_str in topologies:
        topology_graph = topologies[topo_str]
        cage = stk.ConstructedMolecule(topology_graph)
        cage.write(os.path.join(struct_output, f"{topo_str}_unopt.mol"))

        for ba in bite_angles:
            res_dict = run_optimisation(
                cage=cage,
                biteangle=ba,
                topo_str=topo_str,
                output_dir=struct_output,
            )
            results[ba][topo_str] = res_dict

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
