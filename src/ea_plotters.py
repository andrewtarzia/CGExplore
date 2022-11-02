#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for new stk EA plotters.

Author: Andrew Tarzia

"""


import matplotlib.pyplot as plt
import stk
import json
import os
import numpy as np
import logging


class CgProgressPlotter(stk.ProgressPlotter):
    def write(self, path, dpi=500):

        fig, ax = plt.subplots(figsize=(8, 5))

        # It's possible that all values were filtered out, and trying
        # to plot an empty dataframe would raise an exception.
        max_data = self._plot_data[self._plot_data["Type"] == "Max"]
        mean_data = self._plot_data[self._plot_data["Type"] == "Mean"]
        min_data = self._plot_data[self._plot_data["Type"] == "Min"]
        ax.plot(
            max_data["Generation"],
            max_data[self._y_label],
            c="#087E8B",
            label="max",
            # edgecolor="white",
            # s=100,
            alpha=1.0,
            lw=3,
            marker="o",
            markersize=4,
        )
        ax.plot(
            mean_data["Generation"],
            mean_data[self._y_label],
            c="#FF5A5F",
            label="mean",
            # edgecolor="white",
            # s=100,
            alpha=1.0,
            lw=3,
            marker="o",
            markersize=4,
        )
        ax.plot(
            min_data["Generation"],
            min_data[self._y_label],
            c="#6D435A",
            label="min",
            # edgecolor="white",
            # s=100,
            alpha=1.0,
            lw=3,
            marker="o",
            markersize=4,
        )

        # Set the length of the axes to account for all generations,
        # as its possible the first or last ones were not included
        # due to being filtered out.
        ax.set_xlim(0, self._num_generations)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("generation", fontsize=16)
        ax.set_ylabel(self._y_label, fontsize=16)
        ax.legend(loc=0, fontsize=16)

        fig.tight_layout()
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close("all")
        return self


def plot_existing_data_distributions(
    calculation_dir,
    figures_dir,
    suffix=None,
):
    if suffix is None:
        suffix = "res"
    all_jsons = list(calculation_dir.glob(f"*_{suffix}.json"))
    count = len(all_jsons)
    logging.info(f"there are {count} existing data points")

    if count == 0:
        return

    target_radius = 2.5
    target_energy = 0
    target_shape = 0

    scores = []
    energies = []
    oh6_measures = []
    pore_radii = []
    num_atoms = []
    for json_file in all_jsons:
        with open(json_file, "r") as f:
            res_dict = json.load(f)

        pore_radius = res_dict["opt_pore_data"]["pore_max_rad"]
        pore_size_diff = abs(5 - pore_radius * 2) / 5
        score = (
            1
            / (
                res_dict["fin_energy"]
                + res_dict["oh6_measure"] * 100
                + pore_size_diff * 100
            )
            * 10
        )
        scores.append(score)
        energies.append(res_dict["fin_energy"])
        oh6_measures.append(res_dict["oh6_measure"])
        pore_radii.append(pore_radius)

        xyz_file = str(json_file).replace("_res.json", "_opted.xyz")
        with open(xyz_file, "r") as f:
            na = int(f.readline().strip())
        num_atoms.append(na)

    fig, axs = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(8, 10),
    )

    axs[0].hist(
        x=scores,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("fitness", fontsize=16)
    axs[0].set_ylabel("frequency", fontsize=16)

    axs[1].hist(
        x=energies,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("energies", fontsize=16)
    axs[1].set_ylabel("frequency", fontsize=16)
    axs[1].axvline(x=target_energy, linestyle="--", lw=2, c="k")

    axs[2].hist(
        x=oh6_measures,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_xlabel("oh6_measures", fontsize=16)
    axs[2].set_ylabel("frequency", fontsize=16)
    axs[2].axvline(x=target_shape, linestyle="--", lw=2, c="k")

    axs[3].hist(
        x=pore_radii,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[3].tick_params(axis="both", which="major", labelsize=16)
    axs[3].set_xlabel("pore_radii", fontsize=16)
    axs[3].set_ylabel("frequency", fontsize=16)
    axs[3].axvline(x=target_radius, linestyle="--", lw=2, c="k")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_dists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10),
    )

    hb = axs[0][0].hexbin(
        energies,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[0][0].tick_params(axis="both", which="major", labelsize=16)
    axs[0][0].set_xlabel("energies", fontsize=16)
    axs[0][0].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[0][0], label="log10(N)")
    axs[0][0].axvline(x=target_energy, linestyle="--", lw=2, c="k")

    hb = axs[0][1].hexbin(
        oh6_measures,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[0][1].tick_params(axis="both", which="major", labelsize=16)
    axs[0][1].set_xlabel("oh6_measures", fontsize=16)
    axs[0][1].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[0][1], label="log10(N)")
    axs[0][1].axvline(x=target_shape, linestyle="--", lw=2, c="k")

    hb = axs[1][0].hexbin(
        pore_radii,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[1][0].tick_params(axis="both", which="major", labelsize=16)
    axs[1][0].set_xlabel("pore_radii", fontsize=16)
    axs[1][0].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[1][0], label="log10(N)")
    axs[1][0].axvline(x=target_radius, linestyle="--", lw=2, c="k")

    hb = axs[1][1].hexbin(
        pore_radii,
        oh6_measures,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[1][1].tick_params(axis="both", which="major", labelsize=16)
    axs[1][1].set_xlabel("pore_radii", fontsize=16)
    axs[1][1].set_ylabel("oh6_measures", fontsize=16)
    fig.colorbar(hb, ax=axs[1][1], label="log10(N)")
    axs[1][1].axvline(x=target_radius, linestyle="--", lw=2, c="k")
    axs[1][1].axhline(y=target_shape, linestyle="--", lw=2, c="k")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_hexdists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(figsize=(5, 5))
    hb = axs.hexbin(
        num_atoms,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs.tick_params(axis="both", which="major", labelsize=16)
    axs.set_xlabel("num. atoms", fontsize=16)
    axs.set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs, label="log10(N)")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_na_hexdists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def flatten(li):
    return [item for sublist in li for item in sublist]


def plot_existing_guest_data_distributions(
    calculation_dir,
    figures_dir,
    suffix=None,
):
    if suffix is None:
        suffix = "res"
    all_jsons = list(calculation_dir.glob(f"*_{suffix}.json"))
    count = len(all_jsons)
    logging.info(f"there are {count} existing data points")

    if count == 0:
        return

    target_volume_ratio = 0.55
    target_distance = 2.0
    target_energy = 0
    target_shape = 0

    scores = []
    energies = []
    oh6_measures = []
    volume_ratios = []
    min_distances = []
    for json_file in all_jsons:
        with open(json_file, "r") as f:
            res_dict = json.load(f)

        pore_data = res_dict["opt_pore_data"]
        volume_ratio = (
            res_dict["guest_volume"] / pore_data["pore_volume"]
        )
        volume_ratio_score = abs(volume_ratio - 0.55)

        min_distance = np.min(flatten(res_dict["hg_distances"]))
        min_distance_score = abs((min_distance - 2)) / 2.0
        score = 10 / (
            res_dict["fin_energy"]
            + res_dict["oh6_measure"] * 100
            + volume_ratio_score * 100
            + min_distance_score * 100
        )

        scores.append(score)
        energies.append(res_dict["fin_energy"])
        oh6_measures.append(res_dict["oh6_measure"])
        volume_ratios.append(volume_ratio)
        min_distances.append(min_distance)

    fig, axs = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(8, 10),
    )

    axs[0].hist(
        x=scores,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("fitness", fontsize=16)
    axs[0].set_ylabel("frequency", fontsize=16)

    axs[1].hist(
        x=energies,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("energies", fontsize=16)
    axs[1].set_ylabel("frequency", fontsize=16)
    axs[1].axvline(x=target_energy, linestyle="--", lw=2, c="k")

    axs[2].hist(
        x=oh6_measures,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_xlabel("oh6_measures", fontsize=16)
    axs[2].set_ylabel("frequency", fontsize=16)
    axs[2].axvline(x=target_shape, linestyle="--", lw=2, c="k")

    axs[3].hist(
        x=volume_ratios,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[3].tick_params(axis="both", which="major", labelsize=16)
    axs[3].set_xlabel("volume_ratios", fontsize=16)
    axs[3].set_ylabel("frequency", fontsize=16)
    axs[3].axvline(x=target_volume_ratio, linestyle="--", lw=2, c="k")

    axs[4].hist(
        x=min_distances,
        bins=50,
        # range=(0, 4.4),
        density=True,
        histtype="step",
        color="k",
        lw=3,
    )
    axs[4].tick_params(axis="both", which="major", labelsize=16)
    axs[4].set_xlabel("min_distances", fontsize=16)
    axs[4].set_ylabel("frequency", fontsize=16)
    axs[4].axvline(x=target_distance, linestyle="--", lw=2, c="k")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_guest_dists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10),
    )

    hb = axs[0][0].hexbin(
        energies,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[0][0].tick_params(axis="both", which="major", labelsize=16)
    axs[0][0].set_xlabel("energies", fontsize=16)
    axs[0][0].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[0][0], label="log10(N)")
    axs[0][0].axvline(x=target_energy, linestyle="--", lw=2, c="k")

    hb = axs[0][1].hexbin(
        oh6_measures,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[0][1].tick_params(axis="both", which="major", labelsize=16)
    axs[0][1].set_xlabel("oh6_measures", fontsize=16)
    axs[0][1].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[0][1], label="log10(N)")
    axs[0][1].axvline(x=target_shape, linestyle="--", lw=2, c="k")

    hb = axs[1][0].hexbin(
        volume_ratios,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[1][0].tick_params(axis="both", which="major", labelsize=16)
    axs[1][0].set_xlabel("volume_ratios", fontsize=16)
    axs[1][0].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[1][0], label="log10(N)")
    axs[1][0].axvline(
        x=target_volume_ratio, linestyle="--", lw=2, c="k"
    )

    hb = axs[1][1].hexbin(
        min_distances,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[1][1].tick_params(axis="both", which="major", labelsize=16)
    axs[1][1].set_xlabel("min_distances", fontsize=16)
    axs[1][1].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[1][1], label="log10(N)")
    axs[1][1].axvline(x=target_distance, linestyle="--", lw=2, c="k")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_guest_hexdists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 5),
    )

    hb = axs[0].hexbin(
        oh6_measures,
        volume_ratios,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("oh6_measures", fontsize=16)
    axs[0].set_ylabel("volume_ratios", fontsize=16)
    fig.colorbar(hb, ax=axs[0], label="log10(N)")
    axs[0].axvline(x=target_shape, linestyle="--", lw=2, c="k")
    axs[0].axhline(y=target_volume_ratio, linestyle="--", lw=2, c="k")

    hb = axs[1].hexbin(
        min_distances,
        volume_ratios,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("min_distances", fontsize=16)
    axs[1].set_ylabel("volume_ratios", fontsize=16)
    fig.colorbar(hb, ax=axs[1], label="log10(N)")
    axs[1].axvline(x=target_distance, linestyle="--", lw=2, c="k")
    axs[1].axhline(y=target_volume_ratio, linestyle="--", lw=2, c="k")

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figures_dir, "known_library_guest_maphexdists.pdf"
        ),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
