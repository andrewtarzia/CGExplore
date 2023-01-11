#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to plot phase spaces.

Author: Andrew Tarzia

"""

import sys
import os
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt


from env_set import cages

from analysis_utilities import (
    write_out_mapping,
    map_cltype_to_shapetopology,
    convert_topo_to_label,
    topo_to_colormap,
    max_energy,
    stoich_map,
    data_to_array,
    isomer_energy,
)


def phase_space_1(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # t_map = map_cltype_to_shapetopology()
    # fig, axs = plt.subplots(
    #     nrows=3,
    #     ncols=3,
    #     sharex=True,
    #     sharey=True,
    #     figsize=(16, 10),
    # )
    # flat_axs = axs.flatten()

    # data_dict = {}
    # for bb_triplet in all_data:
    #     b_dict = all_data[bb_triplet]
    #     cl_bbname = bb_triplet[1]
    #     c2_bbname = bb_triplet[0]
    #     present_beads_names = get_present_beads(c2_bbname)

    #     x = get_CGBead_from_string(
    #         present_beads_names[0], core_2c_beads()
    #     ).sigma
    #     # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
    #     y = (
    #         get_CGBead_from_string(
    #             present_beads_names[-1], arm_2c_beads()
    #         ).angle_centered
    #         - 90
    #     ) * 2

    #     min_energy = min(tuple(i[1] for i in b_dict.values()))
    #     min_e_dict = {
    #         i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
    #     }
    #     keys = list(min_e_dict.keys())
    #     min_energy_topo = keys[0][-1]

    #     if "3C1" in cl_bbname:
    #         if t_map["3C1"][min_energy_topo] not in data_dict:
    #             data_dict[t_map["3C1"][min_energy_topo]] = []
    #         data_dict[t_map["3C1"][min_energy_topo]].append((x, y))
    #     elif "4C1" in cl_bbname:
    #         if t_map["4C1"][min_energy_topo] not in data_dict:
    #             data_dict[t_map["4C1"][min_energy_topo]] = []
    #         data_dict[t_map["4C1"][min_energy_topo]].append((x, y))

    # for ax, t_str in zip(flat_axs, topology_labels()):
    #     ax.scatter(
    #         [i[0] for i in data_dict[t_str]],
    #         [i[1] for i in data_dict[t_str]],
    #         c="gray",
    #         edgecolor="none",
    #         s=30,
    #         alpha=0.2,
    #     )

    #     ax.tick_params(axis="both", which="major", labelsize=16)
    #     ax.set_xlabel("target 2c core size", fontsize=16)
    #     ax.set_ylabel("target 2c bite angle", fontsize=16)
    #     ax.set_title(t_str, fontsize=16)

    # fig.tight_layout()
    # fig.savefig(
    #     os.path.join(figure_output, "ps_1.pdf"),
    #     dpi=720,
    #     bbox_inches="tight",
    # )
    # plt.close()


def phase_space_2(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    axmap = (
        {
            "ax": flat_axs[0],
            "tor": "toff",
            "x": "pore",
            "y": "energy",
        },
        {
            "ax": flat_axs[4],
            "tor": "ton",
            "x": "pore",
            "y": "energy",
        },
        {
            "ax": flat_axs[1],
            "tor": "toff",
            "x": "min_b2b",
            "y": "energy",
        },
        {
            "ax": flat_axs[5],
            "tor": "ton",
            "x": "min_b2b",
            "y": "energy",
        },
        {
            "ax": flat_axs[2],
            "tor": "toff",
            "x": "sv_l_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[6],
            "tor": "ton",
            "x": "sv_l_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[3],
            "tor": "toff",
            "x": "sv_n_dist",
            "y": "energy",
        },
        {
            "ax": flat_axs[7],
            "tor": "ton",
            "x": "sv_n_dist",
            "y": "energy",
        },
    )
    for axd in axmap:
        ax = axd["ax"]
        tdata = all_data[all_data["torsions"] == axd["tor"]]
        edata = tdata[tdata["energy"] < max_energy() * 50]
        xvalues = edata[axd["x"]]
        yvalues = edata[axd["y"]]
        hb = ax.hexbin(
            xvalues,
            yvalues,
            gridsize=20,
            cmap="inferno",
            bins="log",
            vmax=len(all_data),
        )
        cbar = fig.colorbar(hb, ax=ax, label="log10(N)")
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel(f"{axd['x']}", fontsize=16)
        ax.set_ylabel(f"{axd['y']}", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_2.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_3(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

    opt_data = all_data[all_data["optimised"]]
    groups = opt_data.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"].values()},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"].values()},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"].values()},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"].values()},
    }
    for gid, dfi in groups:
        bbtitle = gid[:3]
        for tors in ("ton", "toff"):

            fin_data = dfi[dfi["torsions"] == tors]
            energies = {
                str(row["topology"]): float(row["energy"])
                / stoich_map(str(row["topology"]))
                for i, row in fin_data.iterrows()
            }
            num_mixed = len(
                tuple(
                    i
                    for i in list(energies.values())
                    if i < isomer_energy()
                )
            )
            min_energy = min(energies.values())
            if min_energy > max_energy():
                topo_str = "unstable"
            elif num_mixed > 1:
                topo_str = "mixed"
            else:
                topo_str = list(energies.keys())[
                    list(energies.values()).index(min_energy)
                ]
            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

        if "mixed" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["mixed"] = 0
        if "mixed" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["mixed"] = 0
        if "unstable" not in data[(bbtitle, "toff")]:
            data[(bbtitle, "toff")]["unstable"] = 0
        if "unstable" not in data[(bbtitle, "ton")]:
            data[(bbtitle, "ton")]["unstable"] = 0

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        ax.bar(
            [convert_topo_to_label(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        for i, key in enumerate(coords):
            val = coords[key]
            if val < 20:
                move = 20
            else:
                move = -20
            ax.text(
                i,
                val + move,
                val,
                fontsize=16,
                ha="center",
            )

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_3.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_11(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

    opt_data = all_data[all_data["optimised"]]
    groups = opt_data.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"].values()},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"].values()},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"].values()},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"].values()},
    }
    for gid, dfi in groups:
        bbtitle = gid[:3]
        for tors in ("ton", "toff"):
            fin_data = dfi[dfi["torsions"] == tors]
            for tstr in topologies[bbtitle].values():
                t_data = fin_data[fin_data["topology"] == tstr]
                if len(t_data) != 1:
                    continue
                if t_data.iloc[0]["persistent"]:
                    topo_str = tstr
                else:
                    topo_str = "not"

                if topo_str not in data[(bbtitle, tors)]:
                    data[(bbtitle, tors)][topo_str] = 0
                data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        ax.bar(
            [convert_topo_to_label(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        for i, key in enumerate(coords):
            val = coords[key]
            if val < 20:
                move = 20
            else:
                move = -20
            ax.text(
                i,
                val + move,
                val,
                fontsize=16,
                ha="center",
            )

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_11.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_12(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    topologies = map_cltype_to_shapetopology()

    opt_data = all_data[all_data["optimised"]]
    groups = opt_data.groupby(["bbpair"])
    data = {
        ("3C1", "toff"): {i: 0 for i in topologies["3C1"].values()},
        ("3C1", "ton"): {i: 0 for i in topologies["3C1"].values()},
        ("4C1", "toff"): {i: 0 for i in topologies["4C1"].values()},
        ("4C1", "ton"): {i: 0 for i in topologies["4C1"].values()},
    }
    for gid, dfi in groups:
        bbtitle = gid[:3]
        for tors in ("ton", "toff"):
            fin_data = dfi[dfi["torsions"] == tors]
            per_data = fin_data[fin_data["persistent"]]
            present_topologies = list(per_data["topology"])
            if len(present_topologies) == 1:
                topo_str = present_topologies[0]
            else:
                topo_str = "mixed"

            if topo_str not in data[(bbtitle, tors)]:
                data[(bbtitle, tors)][topo_str] = 0
            data[(bbtitle, tors)][topo_str] += 1

    for ax, (bbtitle, torsion) in zip(flat_axs, data):
        coords = data[(bbtitle, torsion)]
        ax.bar(
            [convert_topo_to_label(i) for i in coords.keys()],
            coords.values(),
            # color="#06AED5",
            # color="#086788",
            # color="#DD1C1A",
            color="#de9ed6",
            edgecolor="k",
        )

        for i, key in enumerate(coords):
            val = coords[key]
            if val < 30:
                move = 20
            else:
                move = -20
            ax.text(
                i,
                val + move,
                val,
                fontsize=16,
                ha="center",
            )

        title = (
            f"{bbtitle}, {torsion}: {isomer_energy()}eV: "
            f"{max_energy()}eV"
        )
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("count", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_12.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_13(all_data, figure_output):

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    color_map = topo_to_colormap()
    for ax, tstr in zip(flat_axs, color_map):
        t_data = all_data[all_data["topology"] == tstr]
        n_values = list(t_data["sv_n_dist"])
        ax.hist(
            x=n_values,
            bins=50,
            density=False,
            histtype="step",
            color="#DD1C1A",
            lw=3,
        )

        filt_data = t_data[t_data["sv_l_dist"].notna()]
        if len(filt_data) > 0:
            l_values = list(t_data["sv_l_dist"])
            ax.hist(
                x=l_values,
                bins=50,
                density=False,
                histtype="step",
                color="k",
                lw=2,
                linestyle="--",
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("cosine similarity", fontsize=16)
        ax.set_ylabel("log(count)", fontsize=16)
        ax.set_title(tstr, fontsize=16)
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_13.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_4(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # color_map = shapelabels_to_colormap()

    # fig, axs = plt.subplots(
    #     nrows=1,
    #     ncols=2,
    #     figsize=(16, 5),
    # )
    # flat_axs = axs.flatten()

    # s_vectors_3c0 = {}
    # s_vectors_4c0 = {}
    # row_3c0 = set()
    # row_4c0 = set()
    # for bb_triplet in bb_data:
    #     b_dict = bb_data[bb_triplet]
    #     cl_bbname = bb_triplet[1]

    #     min_energy = min(tuple(i[1] for i in b_dict.values()))
    #     min_e_dict = {
    #         i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
    #     }
    #     keys = list(min_e_dict.keys())
    #     min_energy_topo = keys[0][-1]

    #     shape_vector = get_shape_vector(b_dict)
    #     if "3C1" in cl_bbname:
    #         s_vectors_3c0[
    #             (bb_triplet[0], bb_triplet[1], min_energy_topo)
    #         ] = shape_vector
    #         for i in shape_vector:
    #             row_3c0.add(i)
    #     elif "4C1" in cl_bbname:
    #         s_vectors_4c0[
    #             (bb_triplet[0], bb_triplet[1], min_energy_topo)
    #         ] = shape_vector
    #         for i in shape_vector:
    #             row_4c0.add(i)

    # shape_vector_dicts = (
    #     ("3C1", s_vectors_3c0, row_3c0),
    #     ("4C1", s_vectors_4c0, row_4c0),
    # )
    # for ax, (title, coords, row_names) in zip(
    #     flat_axs, shape_vector_dicts
    # ):
    #     data_array = pd.DataFrame.from_dict(
    #         coords,
    #         orient="index",
    #     ).reset_index()

    #     # Separating out the features
    #     x = data_array.loc[:, row_names].values
    #     # Standardizing the features
    #     x = StandardScaler().fit_transform(x)
    #     pca = PCA(n_components=2)
    #     pcs = pca.fit_transform(x)
    #     pc_df = pd.DataFrame(
    #         data=pcs,
    #         columns=["pc1", "pc2"],
    #     )

    #     for t_final in color_map:
    #         ax.scatter(
    #             pc_df["pc1"][data_array["level_2"] == t_final],
    #             pc_df["pc2"][data_array["level_2"] == t_final],
    #             c=color_map[str(t_final)],
    #             edgecolor="none",
    #             s=30,
    #             alpha=1.0,
    #             label=t_final,
    #         )

    #     ax.tick_params(axis="both", which="major", labelsize=16)
    #     ax.set_xlabel("principal component 1", fontsize=16)
    #     ax.set_ylabel("principal component 2", fontsize=16)
    #     ax.set_title(f"PCA: {title}", fontsize=16)
    #     ax.legend(ncol=2, fontsize=16)

    # fig.tight_layout()
    # fig.savefig(
    #     os.path.join(figure_output, "ps_4.pdf"),
    #     dpi=720,
    #     bbox_inches="tight",
    # )
    # plt.close()


def phase_space_5(all_data, figure_output):
    fig, axs = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(16, 8),
    )
    flat_axs = axs.flatten()
    cmap = topo_to_colormap()
    for ax, tstr in zip(flat_axs, cmap):
        tdata = all_data[all_data["topology"] == tstr]
        tondata = tdata[tdata["torsions"] == "ton"]
        # toffdata = tdata[tdata["torsions"] == "toff"]
        x1 = tondata["c2angle"]
        # x2 = toffdata["c2angle"]
        y1 = tondata["sv_n_dist"]
        # y2 = toffdata["sv_n_dist"]
        z1 = tondata["energy"]
        # z2 = toffdata["energy"]

        ax.set_title(tstr, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("c2angle", fontsize=16)
        ax.set_ylabel("sv_n_dist", fontsize=16)

        ax.scatter(
            x1,
            y1,
            c=z1,
            vmin=0,
            vmax=30,
            alpha=0.4,
            edgecolor="none",
            s=30,
            cmap="inferno",
        )

    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=0, vmax=30)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("energy (eV)", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, "ps_5.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def phase_space_6(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # fig, axs = plt.subplots(
    #     nrows=3,
    #     ncols=3,
    #     figsize=(16, 10),
    # )
    # flat_axs = axs.flatten()

    # target_individuals = mapshape_to_topology()

    # shape_coordinates = {
    #     target_individuals[i]: [] for i in target_individuals
    # }
    # for bb_triplet in bb_data:
    #     b_dict = bb_data[bb_triplet]
    #     if "4C1" in bb_triplet[1]:
    #         shapes = target_shapes_by_cltype("4C1")
    #     elif "3C1" in bb_triplet[1]:
    #         shapes = target_shapes_by_cltype("3C1")

    #     for shape in shapes:
    #         topo_str = target_individuals[shape]
    #         try:
    #             shape_value = b_dict[shape][0]
    #         except KeyError:
    #             continue
    #         energy = b_dict[shape][1]
    #         min_distance = b_dict[shape][2]
    #         x = shape_value
    #         y = min_distance
    #         z = energy
    #         shape_coordinates[topo_str].append((x, y, z))

    # for ax, topo_str in zip(flat_axs, shape_coordinates):
    #     coords = shape_coordinates[topo_str]
    #     shape_str = list(target_individuals.keys())[
    #         list(target_individuals.values()).index(topo_str)
    #     ]
    #     ax.set_title(topo_str, fontsize=16)
    #     ax.tick_params(axis="both", which="major", labelsize=16)
    #     ax.set_xlabel(shape_str, fontsize=16)
    #     ax.set_ylabel("min. distance [A]", fontsize=16)

    #     ax.scatter(
    #         [i[0] for i in coords],
    #         [i[1] for i in coords],
    #         c=[i[2] for i in coords],
    #         vmin=0,
    #         vmax=30,
    #         alpha=0.4,
    #         edgecolor="none",
    #         s=30,
    #         cmap="inferno",
    #     )

    # cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    # cmap = mpl.cm.inferno
    # norm = mpl.colors.Normalize(vmin=0, vmax=30)
    # cbar = fig.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #     cax=cbar_ax,
    #     orientation="vertical",
    # )
    # cbar.ax.tick_params(labelsize=16)
    # cbar.set_label("energy (eV)", fontsize=16)
    # fig.tight_layout()
    # fig.savefig(
    #     os.path.join(figure_output, "ps_6.pdf"),
    #     dpi=720,
    #     bbox_inches="tight",
    # )
    # plt.close()


def phase_space_7(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # topologies = mapshape_to_topology(reverse=True)

    # t_map = map_cltype_to_shapetopology()

    # fig, axs = plt.subplots(
    #     nrows=3,
    #     ncols=3,
    #     sharex=True,
    #     sharey=True,
    #     figsize=(16, 10),
    # )
    # flat_axs = axs.flatten()

    # data_dict = {}
    # for bb_triplet in bb_data:
    #     b_dict = bb_data[bb_triplet]
    #     cl_bbname = bb_triplet[1]
    #     c2_bbname = bb_triplet[0]
    #     present_beads_names = get_present_beads(c2_bbname)

    #     x = get_CGBead_from_string(
    #         present_beads_names[0], core_2c_beads()
    #     ).sigma
    #     # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
    #     y = (
    #         get_CGBead_from_string(
    #             present_beads_names[-1], arm_2c_beads()
    #         ).angle_centered
    #         - 90
    #     ) * 2

    #     min_energy = min(tuple(i[1] for i in b_dict.values()))
    #     min_e_dict = {
    #         i: b_dict[i] for i in b_dict if b_dict[i][1] == min_energy
    #     }
    #     keys = list(min_e_dict.keys())
    #     min_energy_topo = keys[0][-1]

    #     if "3C1" in cl_bbname:
    #         topology = t_map["3C1"][min_energy_topo]
    #     elif "4C1" in cl_bbname:
    #         topology = t_map["4C1"][min_energy_topo]

    #     target_shape = topologies[topology]
    #     s = min_e_dict[target_shape][0]

    #     if topology not in data_dict:
    #         data_dict[topology] = []
    #     data_dict[topology].append((x, y, s))

    # for ax, t_str in zip(flat_axs, topologies):
    #     ax.scatter(
    #         [i[0] for i in data_dict[t_str]],
    #         [i[1] for i in data_dict[t_str]],
    #         c=[i[2] for i in data_dict[t_str]],
    #         edgecolor="none",
    #         s=30,
    #         alpha=1.0,
    #         vmin=0,
    #         vmax=20,
    #         cmap="viridis",
    #     )

    #     title = t_str + ": " + topologies[t_str].rstrip("b")

    #     ax.tick_params(axis="both", which="major", labelsize=16)
    #     ax.set_xlabel("target 2c core size", fontsize=16)
    #     ax.set_ylabel("target 2c bite angle", fontsize=16)
    #     ax.set_title(title, fontsize=16)

    # cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    # cmap = mpl.cm.viridis
    # norm = mpl.colors.Normalize(vmin=0, vmax=20)
    # cbar = fig.colorbar(
    #     mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #     cax=cbar_ax,
    #     orientation="vertical",
    # )
    # cbar.ax.tick_params(labelsize=16)
    # cbar.set_label("target shape", fontsize=16)

    # fig.tight_layout()
    # fig.savefig(
    #     os.path.join(figure_output, "ps_7.pdf"),
    #     dpi=720,
    #     bbox_inches="tight",
    # )
    # plt.close()


def phase_space_8(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # t_map = map_cltype_to_shapetopology()

    # color_map = cltypetopo_to_colormap()

    # data_dict = {}
    # for bb_triplet in bb_data:
    #     b_dict = bb_data[bb_triplet]
    #     cl_bbname = bb_triplet[1]
    #     c2_bbname = bb_triplet[0]

    #     if "3C1" in cl_bbname:
    #         title = "3C1"
    #     elif "4C1" in cl_bbname:
    #         title = "4C1"

    #     present_beads_names = get_present_beads(cl_bbname)
    #     core_cl_bead = present_beads_names[0]
    #     plot_set = (title, core_cl_bead)

    #     present_beads_names = get_present_beads(c2_bbname)
    #     core_bead_s = present_beads_names[0]
    #     arm_bead_s = present_beads_names[-1]
    #     x = get_CGBead_from_string(core_bead_s, core_2c_beads()).sigma
    #     # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
    #     y = (
    #         get_CGBead_from_string(
    #             arm_bead_s, arm_2c_beads()
    #         ).angle_centered
    #         - 90
    #     ) * 2

    #     all_energies = set(
    #         b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
    #     )
    #     num_mixed = len(
    #         tuple(i for i in all_energies if i < isomer_energy())
    #     )
    #     if num_mixed > 1:
    #         if num_mixed == 2:
    #             s = color_map["mixed"]["2"]
    #         elif num_mixed > 2:
    #             s = color_map["mixed"][">2"]

    #     else:
    #         min_energy = min(tuple(i[1] for i in b_dict.values()))
    #         min_e_dict = {
    #             i: b_dict[i]
    #             for i in b_dict
    #             if b_dict[i][1] == min_energy
    #         }
    #         keys = list(min_e_dict.keys())
    #         min_energy_topo = keys[0][-1]
    #         topology = t_map[title][min_energy_topo]

    #         if min_energy > max_energy():
    #             s = "white"
    #         else:
    #             s = color_map[title][topology]

    #     if plot_set not in data_dict:
    #         data_dict[plot_set] = []
    #     data_dict[plot_set].append((x, y, s))

    # for data_pair in data_dict:
    #     cltype, cl_bead = data_pair
    #     cbead = get_CGBead_from_string(cl_bead, beads_3c() + beads_4c())
    #     cl_bead_sigma = cbead.sigma
    #     cl_bead_angle = cbead.angle_centered

    #     figtitle_suffix = f"{cltype}_{cl_bead}"

    #     # fig, axs = plt.subplots(
    #     #     nrows=1,
    #     #     ncols=2,
    #     #     sharex=True,
    #     #     sharey=True,
    #     #     figsize=(16, 5),
    #     # )
    #     # flat_axs = axs.flatten()
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     ax.scatter(
    #         [i[0] for i in data_dict[data_pair]],
    #         [i[1] for i in data_dict[data_pair]],
    #         c=[i[2] for i in data_dict[data_pair]],
    #         edgecolor="k",
    #         s=300,
    #         marker="s",
    #         alpha=1.0,
    #     )
    #     title = (
    #         f"{cltype} : {cl_bead_sigma} : {cl_bead_angle} : "
    #         f"{max_energy()}eV : {isomer_energy()}eV"
    #     )

    #     ax.tick_params(axis="both", which="major", labelsize=16)
    #     ax.set_xlabel("target 2c core size", fontsize=16)
    #     ax.set_ylabel("target 2c bite angle", fontsize=16)
    #     ax.set_title(title, fontsize=16)

    #     for i in color_map:
    #         if i not in (cltype, "mixed"):
    #             continue
    #         for j in color_map[i]:
    #             if i == "mixed":
    #                 string = f"mixed: {j}"
    #             else:
    #                 string = j
    #             ax.scatter(
    #                 None,
    #                 None,
    #                 c=color_map[i][j],
    #                 edgecolor="k",
    #                 s=300,
    #                 marker="s",
    #                 alpha=1.0,
    #                 label=string,
    #             )
    #     ax.scatter(
    #         None,
    #         None,
    #         c="white",
    #         edgecolor="k",
    #         s=300,
    #         marker="s",
    #         alpha=1.0,
    #         label="unstable",
    #     )

    #     fig.legend(
    #         bbox_to_anchor=(0, 1.02, 2, 0.2),
    #         loc="lower left",
    #         ncol=3,
    #         fontsize=16,
    #     )

    #     fig.tight_layout()
    #     fig.savefig(
    #         os.path.join(figure_output, f"ps_8_{figtitle_suffix}.pdf"),
    #         dpi=720,
    #         bbox_inches="tight",
    #     )
    #     plt.close()


def phase_space_9(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # t_map = map_cltype_to_shapetopology()
    # color_map = cltypetopo_to_colormap()

    # data_dict = {}
    # for bb_triplet in bb_data:
    #     b_dict = bb_data[bb_triplet]
    #     cl_bbname = bb_triplet[1]
    #     c2_bbname = bb_triplet[0]

    #     if "3C1" in cl_bbname:
    #         title = "3C1"
    #     elif "4C1" in cl_bbname:
    #         title = "4C1"

    #     present_beads_names = get_present_beads(cl_bbname)
    #     core_cl_bead = present_beads_names[0]
    #     plot_set = (title, core_cl_bead)

    #     present_beads_names = get_present_beads(c2_bbname)
    #     core_bead_s = present_beads_names[0]
    #     arm_bead_s = present_beads_names[-1]
    #     x = get_CGBead_from_string(core_bead_s, core_2c_beads()).sigma
    #     # y = get_CGBead_from_string(c3_core_name, beads_3c()).sigma
    #     y = (
    #         get_CGBead_from_string(
    #             arm_bead_s, arm_2c_beads()
    #         ).angle_centered
    #         - 90
    #     ) * 2

    #     all_energies = set(
    #         b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
    #     )
    #     num_mixed = len(
    #         tuple(i for i in all_energies if i < isomer_energy())
    #     )
    #     if num_mixed > 1:
    #         if num_mixed == 2:
    #             s = color_map["mixed"]["2"]
    #         elif num_mixed > 2:
    #             s = color_map["mixed"][">2"]

    #     else:
    #         min_energy = min(tuple(i[1] for i in b_dict.values()))
    #         min_e_dict = {
    #             i: b_dict[i]
    #             for i in b_dict
    #             if b_dict[i][1] == min_energy
    #         }
    #         keys = list(min_e_dict.keys())
    #         min_energy_topo = keys[0][-1]
    #         min_e_distance = min_e_dict[keys[0]][2]
    #         topology = t_map[title][min_energy_topo]
    #         if min_e_distance < min_radius:
    #             s = "k"
    #         elif min_energy > max_energy():
    #             s = "white"
    #         else:
    #             s = color_map[title][topology]

    #     if plot_set not in data_dict:
    #         data_dict[plot_set] = []
    #     data_dict[plot_set].append((x, y, s))

    # for data_pair in data_dict:
    #     cltype, cl_bead = data_pair
    #     cbead = get_CGBead_from_string(cl_bead, beads_3c() + beads_4c())
    #     cl_bead_sigma = cbead.sigma
    #     cl_bead_angle = cbead.angle_centered

    #     figtitle_suffix = f"{cltype}_{cl_bead}"

    #     # fig, axs = plt.subplots(
    #     #     nrows=1,
    #     #     ncols=2,
    #     #     sharex=True,
    #     #     sharey=True,
    #     #     figsize=(16, 5),
    #     # )
    #     # flat_axs = axs.flatten()
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     ax.scatter(
    #         [i[0] for i in data_dict[data_pair]],
    #         [i[1] for i in data_dict[data_pair]],
    #         c=[i[2] for i in data_dict[data_pair]],
    #         edgecolor="k",
    #         s=300,
    #         marker="s",
    #         alpha=1.0,
    #     )
    #     title = (
    #         f"{cltype} : {cl_bead_sigma} : {cl_bead_angle} : "
    #         f"{max_energy()}eV : {isomer_energy()}eV : {min_radius()}A"
    #     )

    #     ax.tick_params(axis="both", which="major", labelsize=16)
    #     ax.set_xlabel("target 2c core size", fontsize=16)
    #     ax.set_ylabel("target 2c bite angle", fontsize=16)
    #     ax.set_title(title, fontsize=16)

    #     for i in color_map:
    #         if i not in (cltype, "mixed"):
    #             continue
    #         for j in color_map[i]:
    #             if i == "mixed":
    #                 string = f"mixed: {j}"
    #             else:
    #                 string = j
    #             ax.scatter(
    #                 None,
    #                 None,
    #                 c=color_map[i][j],
    #                 edgecolor="k",
    #                 s=300,
    #                 marker="s",
    #                 alpha=1.0,
    #                 label=string,
    #             )
    #     ax.scatter(
    #         None,
    #         None,
    #         c="white",
    #         edgecolor="k",
    #         s=300,
    #         marker="s",
    #         alpha=1.0,
    #         label="unstable",
    #     )
    #     ax.scatter(
    #         None,
    #         None,
    #         c="k",
    #         edgecolor="k",
    #         s=300,
    #         marker="s",
    #         alpha=1.0,
    #         label=f"min distance < {min_radius()}A",
    #     )

    #     fig.legend(
    #         bbox_to_anchor=(0, 1.02, 2, 0.2),
    #         loc="lower left",
    #         ncol=3,
    #         fontsize=16,
    #     )

    #     fig.tight_layout()
    #     fig.savefig(
    #         os.path.join(figure_output, f"ps_9_{figtitle_suffix}.pdf"),
    #         dpi=720,
    #         bbox_inches="tight",
    #     )
    #     plt.close()


def phase_space_10(all_data, figure_output):
    print(all_data.head())
    print(all_data.columns)
    print(all_data.iloc[1])
    raise SystemExit()
    # t_map = map_cltype_to_shapetopology()
    # for t_cltopo, torsions in itertools.product(
    #     (3, 4), ("ton", "toff")
    # ):
    #     input_dict = {}
    #     data_dict = {}
    #     for bb_triplet in bb_data:
    #         b_dict = bb_data[bb_triplet]
    #         cl_bbname = bb_triplet[1]
    #         c2_bbname = bb_triplet[0]
    #         torsion = bb_triplet[2]
    #         if torsion != torsions:
    #             continue
    #         bb_string = f"{cl_bbname}_{c2_bbname}"

    #         present_c2_beads = get_present_beads(c2_bbname)
    #         present_cl_beads = get_present_beads(cl_bbname)

    #         cltopo = int(cl_bbname[0])
    #         if cltopo != t_cltopo:
    #             continue
    #         if cltopo == 4:
    #             clangle = get_CGBead_from_string(
    #                 present_cl_beads[0],
    #                 beads_4c(),
    #             ).angle_centered[0]
    #             cltitle = "4C1"
    #         elif cltopo == 3:
    #             clangle = get_CGBead_from_string(
    #                 present_cl_beads[0],
    #                 beads_3c(),
    #             ).angle_centered
    #             cltitle = "3C1"

    #         all_energies = set(
    #             b_dict[i][1] / int(i.rstrip("b")[-1]) for i in b_dict
    #         )
    #         num_mixed = len(
    #             tuple(i for i in all_energies if i < isomer_energy())
    #         )
    #         if num_mixed > 1:
    #             if num_mixed == 2:
    #                 topology = "mixed (2)"
    #             elif num_mixed > 2:
    #                 topology = "mixed (>2)"
    #             min_e_distance = None

    #         else:
    #             min_energy = min(tuple(i[1] for i in b_dict.values()))
    #             min_e_dict = {
    #                 i: b_dict[i]
    #                 for i in b_dict
    #                 if b_dict[i][1] == min_energy
    #             }
    #             keys = list(min_e_dict.keys())
    #             min_energy_topo = keys[0][-1]
    #             min_e_distance = min_e_dict[keys[0]][2]
    #             topology = t_map[cltitle][min_energy_topo]

    #             if min_energy > max_energy():
    #                 topology = "unstable"

    #         cl_bead_libs = beads_3c().copy()
    #         cl_bead_libs.update(beads_4c())
    #         row = {
    #             "cltopo": cltopo,
    #             "clsigma": get_CGBead_from_string(
    #                 present_cl_beads[0],
    #                 cl_bead_libs,
    #             ).sigma,
    #             "clangle": clangle,
    #             "c2sigma": get_CGBead_from_string(
    #                 present_c2_beads[0],
    #                 core_2c_beads(),
    #             ).sigma,
    #             "c2angle": (
    #                 get_CGBead_from_string(
    #                     present_c2_beads[1],
    #                     arm_2c_beads(),
    #                 ).angle_centered
    #                 - 90
    #             )
    #             * 2,
    #             "pref_topology": topology,
    #             "pref_topology_pore": min_e_distance,
    #         }

    #         input_dict[bb_string] = row
    #         data_dict[bb_string] = b_dict

    #     data_array = pd.DataFrame.from_dict(
    #         data_dict,
    #         orient="index",
    #     ).reset_index()
    #     print(data_array.head())

    #     input_array = pd.DataFrame.from_dict(
    #         input_dict,
    #         orient="index",
    #     ).reset_index()
    #     print(input_array.head())
    #     target_row_names = (
    #         "cltopo",
    #         "clsigma",
    #         "clangle",
    #         "c2sigma",
    #         "c2angle",
    #     )
    #     # Separating out the features
    #     x = input_array.loc[:, target_row_names].values
    #     # Standardizing the features
    #     x = StandardScaler().fit_transform(x)
    #     pca = PCA(n_components=2)
    #     pcs = pca.fit_transform(x)
    #     pc_df = pd.DataFrame(
    #         data=pcs,
    #         columns=["pc1", "pc2"],
    #     )

    #     properties = {
    #         "clangle": "cat",
    #         "c2sigma": "cat",
    #         "c2angle": "cat",
    #         "pref_topology": "cat",
    #         "pref_topology_pore": "cts",
    #     }

    #     for prop in properties:
    #         prop_type = properties[prop]
    #         if prop_type == "cat":
    #             fig, ax = plt.subplots(figsize=(8, 5))

    #             categories = {}
    #             for i, prop_set in enumerate(
    #                 sorted(set(input_array[prop]))
    #             ):
    #                 categories[prop_set] = len(
    #                     input_array[input_array[prop] == prop_set]
    #                 )

    #             ax.bar(
    #                 categories.keys(),
    #                 categories.values(),
    #                 # color="#06AED5",
    #                 color="#086788",
    #                 # color="#DD1C1A",
    #                 # color="#320E3B",
    #                 edgecolor="k",
    #             )

    #             ax.tick_params(axis="both", which="major", labelsize=16)
    #             ax.set_xlabel(prop, fontsize=16)
    #             ax.set_ylabel("count", fontsize=16)

    #             fig.tight_layout()
    #             fig.savefig(
    #                 os.path.join(
    #                     figure_output,
    #                     f"dist_10_{t_cltopo}_{torsions}_{prop}.pdf",
    #                 ),
    #                 dpi=720,
    #                 bbox_inches="tight",
    #             )
    #             plt.close()

    #             fig, ax = plt.subplots(figsize=(8, 5))
    #             ax.scatter(
    #                 pc_df["pc1"],
    #                 pc_df["pc2"],
    #                 c="k",
    #                 edgecolor="none",
    #                 s=60,
    #                 alpha=1.0,
    #             )

    #             for i, prop_set in enumerate(
    #                 sorted(set(input_array[prop]))
    #             ):
    #                 ax.scatter(
    #                     pc_df["pc1"][input_array[prop] == prop_set],
    #                     pc_df["pc2"][input_array[prop] == prop_set],
    #                     # c=color_map[str(t_final)],
    #                     edgecolor="none",
    #                     s=20,
    #                     alpha=1.0,
    #                     label=f"{prop}: {prop_set}",
    #                 )

    #             ax.tick_params(axis="both", which="major", labelsize=16)
    #             ax.set_xlabel("pc1", fontsize=16)
    #             ax.set_ylabel("pc2", fontsize=16)
    #             ax.legend(fontsize=16)

    #             fig.tight_layout()
    #             fig.savefig(
    #                 os.path.join(
    #                     figure_output,
    #                     f"ps_10_{t_cltopo}_{torsions}_{prop}.pdf",
    #                 ),
    #                 dpi=720,
    #                 bbox_inches="tight",
    #             )
    #             plt.close()
    #         elif prop_type == "cts":
    #             fig, ax = plt.subplots(figsize=(8, 5))

    #             ax.hist(
    #                 x=list(input_array[prop]),
    #                 bins=50,
    #                 density=False,
    #                 histtype="step",
    #                 lw=3,
    #             )

    #             ax.tick_params(axis="both", which="major", labelsize=16)
    #             ax.set_xlabel(prop, fontsize=16)
    #             ax.set_ylabel("count", fontsize=16)

    #             fig.tight_layout()
    #             fig.savefig(
    #                 os.path.join(
    #                     figure_output,
    #                     f"dist_10_{t_cltopo}_{torsions}_{prop}.pdf",
    #                 ),
    #                 dpi=720,
    #                 bbox_inches="tight",
    #             )
    #             plt.close()

    #             fig, ax = plt.subplots(figsize=(8, 5))
    #             ax.scatter(
    #                 pc_df["pc1"][pd.notna(input_array[prop])],
    #                 pc_df["pc2"][pd.notna(input_array[prop])],
    #                 c="k",
    #                 edgecolor="none",
    #                 s=60,
    #                 alpha=1.0,
    #             )

    #             ax.scatter(
    #                 pc_df["pc1"][pd.notna(input_array[prop])],
    #                 pc_df["pc2"][pd.notna(input_array[prop])],
    #                 c=list(
    #                     input_array[pd.notna(input_array[prop])][prop]
    #                 ),
    #                 edgecolor="none",
    #                 s=20,
    #                 alpha=1.0,
    #                 vmin=0,
    #                 vmax=20,
    #                 cmap="viridis",
    #             )

    #             cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    #             cmap = mpl.cm.viridis
    #             norm = mpl.colors.Normalize(vmin=0, vmax=20)
    #             cbar = fig.colorbar(
    #                 mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #                 cax=cbar_ax,
    #                 orientation="vertical",
    #             )
    #             cbar.ax.tick_params(labelsize=16)
    #             cbar.set_label(prop, fontsize=16)

    #             ax.tick_params(axis="both", which="major", labelsize=16)
    #             ax.set_xlabel("pc1", fontsize=16)
    #             ax.set_ylabel("pc2", fontsize=16)

    #             fig.tight_layout()
    #             fig.savefig(
    #                 os.path.join(
    #                     figure_output,
    #                     f"ps_10_{t_cltopo}_{torsions}_{prop}.pdf",
    #                 ),
    #                 dpi=720,
    #                 bbox_inches="tight",
    #             )
    #             plt.close()


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    figure_output = cages() / "figures"
    calculation_output = cages() / "calculations"

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=calculation_output,
    )
    logging.info(f"there are {len(all_data)} collected data")
    opt_data = all_data[all_data["optimised"]]
    logging.info(f"there are {len(opt_data)} successfully opted")
    write_out_mapping(opt_data)

    phase_space_2(opt_data, figure_output)
    phase_space_5(opt_data, figure_output)
    phase_space_3(opt_data, figure_output)
    phase_space_13(opt_data, figure_output)
    phase_space_11(opt_data, figure_output)
    phase_space_12(opt_data, figure_output)
    phase_space_10(opt_data, figure_output)
    phase_space_9(opt_data, figure_output)
    phase_space_1(opt_data, figure_output)
    phase_space_6(opt_data, figure_output)
    phase_space_7(opt_data, figure_output)
    phase_space_8(opt_data, figure_output)
    phase_space_4(opt_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
