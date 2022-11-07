#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to optimise CG models of fourplussix host-guest systems.

Author: Andrew Tarzia

"""

import sys
import os
import stk
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from env_set import project_dir
from shape import ShapeMeasure


def get_all_templates(dir):
    templates = {}
    xyzs = dir.glob("*.xyz")
    for xyz in xyzs:
        with open(xyz, "r") as f:
            lines = f.readlines()
        vn = str(xyz.name).split(".")[0].split("-")[1]
        if vn not in templates:
            templates[vn] = {}

        coordinates = []
        for line in lines[3:]:
            x, y, z = [float(i) for i in line.rstrip().split()[1:]]
            coordinates.append([x, y, z])

        templates[vn][str(xyz.name).split(".")[0]] = coordinates

    return templates


def get_shape(coords, num_vertices, name):
    struct_output = project_dir() / "shape_structures"
    molecule = stk.BuildingBlock.init(
        atoms=tuple(stk.H(i) for i in range(num_vertices)),
        bonds=(),
        position_matrix=np.array(coords),
    )
    molecule.write(
        str(struct_output / f"{name}.mol"),
    )
    shape_measures = ShapeMeasure(
        output_dir=(struct_output / f"{name}_shape"),
        target_atmnums=None,
        shape_string=None,
    ).calculate(molecule)
    return shape_measures


def get_cmap(n, name="viridis"):
    return plt.cm.get_cmap(name, n)


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        print(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = project_dir() / "shape_structures"
    if not os.path.exists(struct_output):
        os.mkdir(struct_output)
    figure_output = project_dir() / "figures"
    if not os.path.exists(figure_output):
        os.mkdir(figure_output)
    test_templates = struct_output / "tests"
    if not os.path.exists(test_templates):
        os.mkdir(test_templates)

    all_templates = get_all_templates(test_templates)

    # Iterate over vertices.
    # vertex_nums = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 24)
    vertex_nums = (4, 5, 6, 8)

    # Start with an initial geometry with N vertices, and
    # interpolate (linearly or stochatstically) between possible
    # shapes based on shape vector.
    interpol_points = np.linspace(0.1, 0.9, 10)
    for vn in vertex_nums:
        n_shape_arrays = {}
        vn_temps = all_templates[str(vn)]
        all_pairs = itertools.combinations(all_templates[str(vn)], r=2)
        for pair in all_pairs:
            ini_coord = np.array(vn_temps[pair[0]])
            fin_coord = np.array(vn_temps[pair[1]])
            vectors_between = fin_coord - ini_coord
            n_shape_arrays[pair[0]] = get_shape(ini_coord, vn, pair[0])
            n_shape_arrays[pair[1]] = get_shape(fin_coord, vn, pair[1])
            for multi in interpol_points:
                new_coord = ini_coord + vectors_between * multi
                name = f"{vn}_{len(n_shape_arrays)}"
                n_shape_arrays[name] = get_shape(
                    new_coord,
                    vn,
                    name,
                )

        target_row_names = list(vn_temps.keys())
        target_individuals = ("CU-8", "OC-6", "TBPY-5", "T-4")
        cmap = get_cmap(len(target_row_names))

        # Plot shape map for N vertices.
        data_array = pd.DataFrame.from_dict(
            n_shape_arrays,
            orient="index",
        ).reset_index()

        # Separating out the features
        x = data_array.loc[:, target_row_names].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(x)
        pc_df = pd.DataFrame(
            data=pcs,
            columns=["pc1", "pc2"],
        )
        pc_df = pd.concat(
            [pc_df, data_array[["index"]]],
            axis=1,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            pc_df["pc1"],
            pc_df["pc2"],
            c="grey",
            edgecolor="none",
            s=30,
            alpha=0.8,
        )

        count = 0
        for idx, row in pc_df.iterrows():

            if row["index"] not in target_row_names:
                continue
            print(row["index"], n_shape_arrays[row["index"]])
            if row["index"] in target_individuals:
                ax.scatter(
                    row["pc1"],
                    row["pc2"],
                    color="r",
                    edgecolor="k",
                    marker="D",
                    s=60,
                    label=row["index"],
                )
            else:
                ax.scatter(
                    row["pc1"],
                    row["pc2"],
                    color=cmap(count),
                    edgecolor="k",
                    s=60,
                    label=row["index"],
                )
            count += 1

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("principal component 1", fontsize=16)
        ax.set_ylabel("principal component 2", fontsize=16)
        ax.set_title(f"PCA: {vn} vertices", fontsize=16)
        ax.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            os.path.join(figure_output, f"pca_{vn}.pdf"),
            dpi=720,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
