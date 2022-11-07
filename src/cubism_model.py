#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate CG models of cubism M8L6 systems.

Author: Andrew Tarzia

"""

import sys
import stk
import numpy as np
import os
import json
import logging

from shape import ShapeMeasure

from env_set import cubism
from utilities import (
    get_distances,
    get_angles,
    check_directory,
)
from gulp_optimizer import CGGulpOptimizer

from cube_construction.cube import CGM8L6Cube
from cube_construction.symmetries import symmetries

from cube_construction.plotting import (
    convergence,
    scatter,
    geom_distributions,
    heatmap,
    comp_scatter,
    ey_vs_shape,
)

from precursor_db.precursors import delta_bb, lambda_bb, plane_bb


class CubismOptimizer(CGGulpOptimizer):
    def __init__(
        self,
        fileprefix,
        output_dir,
        anisotropy,
        ortho_k,
        o_angle_k,
    ):
        self._anisotropy = anisotropy
        self._ortho_k = ortho_k
        self._o_angle_k = o_angle_k
        super().__init__(fileprefix, output_dir)

    def define_bond_potentials(self):
        bond_ks_ = {
            ("C", "C"): 10,
            ("B", "B"): 10,
            ("B", "C"): self._ortho_k,
            ("C", "Zn"): 10,
            ("B", "Zn"): 10,
            ("Fe", "Fe"): 10,
            ("Fe", "Zn"): 10,
            ("Zn", "Zn"): 10,
        }
        _base_length = 4
        _ortho_length = self._anisotropy * _base_length
        bond_rs_ = {
            ("C", "C"): _base_length,
            ("B", "B"): _base_length,
            ("B", "C"): _ortho_length,
            ("C", "Zn"): 4,
            ("B", "Zn"): 4,
            ("Fe", "Fe"): 4,
            ("Fe", "Zn"): 4,
            ("Zn", "Zn"): 4,
        }
        return bond_ks_, bond_rs_

    def define_angle_potentials(self):
        angle_ks_ = {
            ("B", "C", "C"): self._o_angle_k,
            ("B", "B", "C"): self._o_angle_k,
            ("Fe", "Fe", "Fe"): 20,
            # ('Fe', 'Fe', 'Zn'): 10,
            ("Fe", "Zn", "Zn"): 20,
            ("Zn", "Zn", "Zn"): 20,
            ("B", "C", "Zn"): self._o_angle_k,
            ("B", "B", "Zn"): self._o_angle_k,
            ("C", "C", "Zn"): self._o_angle_k,
            ("C", "Fe", "Zn"): 20,
            ("B", "Fe", "Zn"): 20,
        }
        angle_thetas_ = {
            ("B", "C", "C"): 90,
            ("B", "B", "C"): 90,
            ("Fe", "Fe", "Fe"): 60,
            # ('Fe', 'Fe', 'Zn'): 60,
            # This requires a special rule.
            ("Fe", "Zn", "Zn"): (
                "check",
                {"cut": 70, "min": 60, "max": 90},
            ),
            ("Zn", "Zn", "Zn"): 60,
            ("B", "C", "Zn"): 135,
            ("B", "B", "Zn"): 135,
            ("C", "C", "Zn"): 135,
            ("C", "Fe", "Zn"): 180,
            ("B", "Fe", "Zn"): 180,
        }
        return angle_ks_, angle_thetas_


def run_optimisation(
    cage,
    aniso,
    symm,
    flex,
    ortho_k,
    o_angle_k,
    output_dir,
    calculation_dir,
):

    run_prefix = f"{symm}_{aniso}_{flex}"
    output_file = os.path.join(output_dir, f"{run_prefix}_res.json")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            res_dict = json.load(f)
    else:
        logging.info(f": running optimisation of {run_prefix}")
        opt = CubismOptimizer(
            fileprefix=run_prefix,
            output_dir=output_dir,
            anisotropy=aniso,
            ortho_k=ortho_k,
            o_angle_k=o_angle_k,
        )
        run_data = opt.optimize(cage)

        # Get cube shape measure.
        opted = cage.with_structure_from_file(
            path=os.path.join(output_dir, f"{run_prefix}_final.xyz"),
        )
        opted.write(os.path.join(output_dir, f"{run_prefix}_final.mol"))

        cu8_measure = ShapeMeasure(
            output_dir=calculation_dir / f"{run_prefix}_shape",
            target_atmnum=30,
            shape_string="cube",
        ).calculate(opted)
        distances = get_distances(optimizer=opt, cage=opted)
        angles = get_angles(optimizer=opt, cage=opted)

        num_steps = len(run_data["traj"])
        fin_energy = run_data["final_energy"]
        fin_gnorm = run_data["final_gnorm"]
        traj_data = run_data["traj"]
        logging.info(
            f"{run_prefix}: {num_steps} {fin_energy} {fin_gnorm} "
            f"{cu8_measure}"
        )
        res_dict = {
            "fin_energy": fin_energy,
            "cu8": cu8_measure,
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

    struct_output = cubism() / "structures"
    check_directory(struct_output)
    figure_output = cubism() / "figures"
    check_directory(figure_output)
    calculation_output = cubism() / "calculations"
    check_directory(calculation_output)

    # Make cage of each symmetry.
    symms = symmetries(delta_bb(), lambda_bb(), plane_bb())
    anisotropies = np.arange(1.0, 2.01, 0.05)
    # results = {i: {} for i in symms}
    flexes = {
        "low": (10, 20),
        "high": (0.1, 2.0),
    }
    for flex in flexes:
        results = {round(i, 2): {} for i in anisotropies}
        for symm in symms:
            topology_graph = CGM8L6Cube(
                building_blocks=symms[symm]["building_blocks"],
                vertex_alignments=symms[symm]["vertex_alignments"],
                num_processes=1,
            )
            cage = stk.ConstructedMolecule(topology_graph)
            cage.write(os.path.join(struct_output, f"{symm}_unopt.mol"))

            for aniso in anisotropies:
                aniso = round(aniso, 2)
                res_dict = run_optimisation(
                    cage=cage,
                    aniso=aniso,
                    symm=symm,
                    flex=flex,
                    ortho_k=flexes[flex][0],
                    o_angle_k=flexes[flex][1],
                    output_dir=struct_output,
                    calculation_output=calculation_output,
                )
                results[aniso][symm] = res_dict

        symm_to_c = {
            "d2": ("o", "k", 0),
            "th1": ("D", "r", 1),
            "th2": ("X", "r", 2),
            "td": ("o", "r", 3),
            "tl": ("P", "r", 4),
            "s61": ("X", "gold", 5),
            "s62": ("D", "gold", 6),
            "s41": ("X", "gray", 7),
            "s42": ("D", "gray", 8),
            "d31": ("P", "skyblue", 9),
            "d32": ("o", "skyblue", 10),
            "d31n": ("P", "b", 11),
            "d32n": ("o", "b", 12),
            "c2h": ("o", "green", 13),
            "c2v": ("X", "green", 14),
        }

        convergence(
            results=results,
            output_dir=figure_output,
            filename=f"convergence_{flex}.pdf",
        )

        ey_vs_shape(
            results=results,
            output_dir=figure_output,
            filename=f"e_vs_shape_{flex}.pdf",
        )

        geom_distributions(
            results=results,
            output_dir=figure_output,
            filename=f"dist_{flex}.pdf",
        )

        heatmap(
            symm_to_c=symm_to_c,
            results=results,
            output_dir=figure_output,
            filename=f"energy_map_{flex}.pdf",
            vmin=0,
            vmax=45,
            clabel="energy (eV)",
            flex=flex,
        )

        heatmap(
            symm_to_c=symm_to_c,
            results=results,
            output_dir=figure_output,
            filename=f"energy_map_flat_{flex}.pdf",
            vmin=0,
            vmax=10,
            clabel="energy (eV)",
            flex=flex,
        )

        heatmap(
            symm_to_c=symm_to_c,
            results=results,
            output_dir=figure_output,
            filename=f"shape_map_{flex}.pdf",
            vmin=0,
            vmax=2.2,
            clabel="CU-8",
            flex=flex,
        )

        scatter(
            symm_to_c=symm_to_c,
            results=results,
            output_dir=figure_output,
            filename=f"energy_{flex}.pdf",
            ylabel="energy (eV)",
            flex=flex,
        )
        scatter(
            symm_to_c=symm_to_c,
            results=results,
            output_dir=figure_output,
            filename=f"shape_{flex}.pdf",
            ylabel="CU-8",
            flex=flex,
        )

        comp_sets = {
            "ts": ("th1", "th2", "td", "tl"),
            "ds": ("d31", "d32", "d31n", "d32n"),
            "ss": ("s61", "s62", "s41", "s42"),
            "expt": ("d2", "tl", "s62", "th2", "d32"),
        }
        for key, values in comp_sets.items():
            if flex == "high":
                eylim = (0, 10)
            else:
                eylim = (0, 45)

            comp_scatter(
                symm_to_c=symm_to_c,
                symm_set=values,
                results=results,
                output_dir=figure_output,
                filename=f"comp_energy_{flex}_{key}.pdf",
                ylabel="energy (eV)",
                flex=flex,
                ylim=eylim,
            )
            comp_scatter(
                symm_to_c=symm_to_c,
                symm_set=values,
                results=results,
                output_dir=figure_output,
                filename=f"comp_shape_{flex}_{key}.pdf",
                ylabel="CU-8",
                flex=flex,
                ylim=(0, 2),
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
