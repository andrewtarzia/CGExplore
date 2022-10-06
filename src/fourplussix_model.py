#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate CG models of fourplussix systems.

Author: Andrew Tarzia

"""

import sys
import stk
import os
import json
import logging
import math

from shape import ShapeMeasure

from env_set import (
    fourplussix_figures,
    fourplussix_structures,
    fourplussix_calculations,
)
from utilities import (
    get_distances,
    get_angles,
)
from gulp_optimizer import (
    CGGulpOptimizer,
    HarmBond,
    ThreeAngle,
    BondSet,
    AngleSet,
)

from fourplusix_construction.topologies import cage_topologies

from fourplusix_construction.plotting import (
    convergence,
    scatter,
    geom_distributions,
    heatmap,
    ey_vs_shape,
)

from precursor_db.precursors import threec_bb, twoc_bb


class FourPlusSixOptimizer(CGGulpOptimizer):
    def __init__(self, fileprefix, output_dir, param_pool):
        self._param_pool = param_pool
        super().__init__(fileprefix, output_dir)

    def define_bond_potentials(self):
        edge_length = 3
        to_centre = edge_length / math.sqrt(3)

        bonds = (
            # Intra-BB.
            HarmBond("B", "N", bond_r=1, bond_k=10),
            HarmBond("C", "P", bond_r=edge_length, bond_k=10),
            HarmBond("C", "S", bond_r=edge_length, bond_k=10),
            HarmBond("P", "S", bond_r=edge_length, bond_k=10),
            HarmBond("C", "Fe", bond_r=to_centre, bond_k=10),
            HarmBond("P", "Fe", bond_r=to_centre, bond_k=10),
            HarmBond("S", "Fe", bond_r=to_centre, bond_k=10),
            # Inter-BB.
            HarmBond("C", "N", bond_r=2, bond_k=10),
            HarmBond("P", "N", bond_r=2, bond_k=10),
            HarmBond("S", "N", bond_r=2, bond_k=10),
        )

        print(bonds)
        new_bonds = []
        for bond in bonds:
            bond_type = bond.get_types()
            if bond_type in self._param_pool["bonds"]:
                k, r = self._param_pool["bonds"][bond_type]
                nbond = HarmBond(bond.atom1_type, bond.atom2_type, r, k)
                new_bonds.append(nbond)
            else:
                new_bonds.append(bond)

        new_bonds = tuple(new_bonds)
        print(new_bonds)
        input("check when modified")

        return BondSet(new_bonds)

    def define_angle_potentials(self):
        angles = (
            # Intra-BB.
            ThreeAngle(
                atom1_type="B",
                atom2_type="N",
                atom3_type="N",
                theta=100,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="C",
                atom2_type="P",
                atom3_type="S",
                theta=60,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="Fe",
                atom2_type="P",
                atom3_type="S",
                theta=120,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="Fe",
                atom2_type="P",
                atom3_type="C",
                theta=120,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="Fe",
                atom2_type="C",
                atom3_type="S",
                theta=120,
                angle_k=20,
            ),
            # Inter-BB.
            ThreeAngle(
                atom1_type="N",
                atom2_type="C",
                atom3_type="B",
                theta=180,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="N",
                atom2_type="P",
                atom3_type="B",
                theta=180,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="N",
                atom2_type="S",
                atom3_type="B",
                theta=180,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="C",
                atom2_type="N",
                atom3_type="Fe",
                theta=180,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="P",
                atom2_type="N",
                atom3_type="Fe",
                theta=180,
                angle_k=20,
            ),
            ThreeAngle(
                atom1_type="S",
                atom2_type="N",
                atom3_type="Fe",
                theta=180,
                angle_k=20,
            ),
        )
        #     # ("C", "N", "P"): 120,
        #     # ("C", "N", "S"): 120,
        #     # ("N", "P", "S"): 120,
        #     # ("B", "N", "S"): 180,
        #     # ("B", "N", "P"): 180,
        #     # ("B", "C", "N"): 180,
        #     # ("C", "Fe", "N"): 180,
        #     # ("Fe", "N", "S"): 180,
        #     # ("Fe", "N", "P"): 180,
        # }

        print(angles)
        new_angles = []
        for angle in angles:
            angle_type = angle.get_types()
            if angle_type in self._param_pool["angles"]:
                k, theta = self._param_pool["angles"][angle_type]
                nangle = ThreeAngle(
                    angle.atom1_type,
                    angle.atom2_type,
                    angle.atom3_type,
                    theta,
                    k,
                )
                new_angles.append(nangle)
            else:
                new_angles.append(angle)

        new_angles = tuple(new_angles)
        print(new_angles)
        input("check when modified")

        return AngleSet(new_angles)


def run_optimisation(
    cage,
    ff_modifications,
    topo_str,
    output_dir,
):

    ffname = ff_modifications["name"]
    run_prefix = f"{topo_str}_{ffname}"
    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f": running optimisation of {run_prefix}")
        opt = FourPlusSixOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            param_pool=ff_modifications["param_pool"],
        )
        run_data = opt.optimize(cage)

        # Get cube shape measure.
        opted = cage.with_structure_from_file(
            path=os.path.join(output_dir, f"{run_prefix}_final.xyz"),
        )
        opted.write(os.path.join(output_dir, f"{run_prefix}_final.mol"))

        return {}
        oh6_measure = ShapeMeasure(
            output_dir=(
                fourplussix_calculations() / f"{run_prefix}_shape"
            ),
            target_atmnum=5,
            shape_string="oc6",
        ).calculate(opted)
        distances = get_distances(optimizer=opt, cage=opted)
        angles = get_angles(optimizer=opt, cage=opted)

        num_steps = len(run_data["traj"])
        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        traj_data = run_data["traj"]
        logging.info(
            f"{run_prefix}: {num_steps} {fin_energy} {fin_gnorm} "
            f"{oh6_measure}"
        )
        res_dict = {
            "fin_energy": fin_energy,
            "oh6": oh6_measure,
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
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = fourplussix_structures()
    figure_output = fourplussix_figures()

    # Make cage of each symmetry.
    topologies = cage_topologies(threec_bb(), twoc_bb(sites=3))

    ff_options = {
        "default": {
            "param_pool": {"bonds": (), "angles": ()},
            "notes": "default FF, high symmetry, rigid",
            "name": "def",
        },
    }

    results = {i: {} for i in ff_options}
    for topo_str in topologies:
        topology_graph = topologies[topo_str]
        cage = stk.ConstructedMolecule(topology_graph)
        cage.write(os.path.join(struct_output, f"{topo_str}_unopt.mol"))

        for ff_str in ff_options:
            res_dict = run_optimisation(
                cage=cage,
                ff_modifications=ff_options[ff_str],
                topo_str=topo_str,
                output_dir=struct_output,
            )
            results[ff_str][topo_str] = res_dict

    print(results)
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

    ey_vs_shape(
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
        vmax=45,
        clabel="energy (eV)",
    )

    heatmap(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="shape_map.pdf",
        vmin=0,
        vmax=2.2,
        clabel="OH-6",
    )

    scatter(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="energy.pdf",
        ylabel="energy (eV)",
    )
    scatter(
        topo_to_c=topo_to_c,
        results=results,
        output_dir=figure_output,
        filename="shape.pdf",
        ylabel="OH-6",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()