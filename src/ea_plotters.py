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
            max_data["Fitness Value"],
            c="#087E8B",
            label="max",
            # edgecolor="white",
            # s=100,
            alpha=1.0,
            lw=3,
            marker="o",
            marker_size=3,
        )
        ax.plot(
            mean_data["Generation"],
            mean_data["Fitness Value"],
            c="#FF5A5F",
            label="mean",
            # edgecolor="white",
            # s=100,
            alpha=1.0,
            lw=3,
            marker="o",
            marker_size=3,
        )
        ax.plot(
            min_data["Generation"],
            min_data["Fitness Value"],
            c="#6D435A",
            label="min",
            # edgecolor="white",
            # s=100,
            alpha=1.0,
            lw=3,
            marker="o",
            marker_size=3,
        )

        # Set the length of the axes to account for all generations,
        # as its possible the first or last ones were not included
        # due to being filtered out.
        ax.set_xlim(0, self._num_generations)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("generation", fontsize=16)
        ax.set_ylabel("fitness value", fontsize=16)
        fig.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close("all")
        return self


def plot_existing_data_distributions(calculation_dir, figures_dir):
    all_jsons = list(calculation_dir.glob("*_res.json"))
    logging.info(f"there are {len(all_jsons)} existing data points")

    scores = []
    energies = []
    oh6_measures = []
    pore_radii = []
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

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_dists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        sharey=True,
        figsize=(16, 5),
    )

    hb = axs[0].hexbin(
        energies,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("energies", fontsize=16)
    axs[0].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[0], label="log10(N)")

    hb = axs[1].hexbin(
        oh6_measures,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("oh6_measures", fontsize=16)
    axs[1].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[1], label="log10(N)")

    hb = axs[2].hexbin(
        pore_radii,
        scores,
        gridsize=20,
        cmap="inferno",
        bins="log",
    )
    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_xlabel("pore_radii", fontsize=16)
    axs[2].set_ylabel("fitness", fontsize=16)
    fig.colorbar(hb, ax=axs[2], label="log10(N)")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "known_library_hexdists.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
