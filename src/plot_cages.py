#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to visulize structures in matplotlib.

Author: Andrew Tarzia

"""

import sys
import os
import logging
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from env_set import cages
from analysis_utilities import (
    isomer_energy,
    data_to_array,
    topology_labels,
    write_out_mapping,
    get_lowest_energy_data,
    convert_topo,
)
from visualisation import Pymol
from utilities import check_directory


def generate_images_of_all(
    all_data,
    struct_output,
    struct_figure_output,
):

    von = all_data[all_data["vdws"] == "von"]

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.7,
        "vdw": 0,
        "zoom_string": "custom",
    }

    for i, row in von.iterrows():
        struct_name = str(row["index"])
        if "2P3" in struct_name:
            orient_atoms = "C"
        elif "2P4" in struct_name:
            orient_atoms = "Pd"
        else:
            orient_atoms = None

        struct_file = struct_output / f"{struct_name}_optc.mol"
        _ = generate_image(
            struct_name=struct_name,
            struct_file=struct_file,
            struct_figure_output=struct_figure_output,
            orient_atoms=orient_atoms,
            settings=settings,
        )


def visualise_low_and_high(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):

    von = all_data[all_data["vdws"] == "von"]
    tlabels = topology_labels(short="P")

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    fig, axs = plt.subplots(
        ncols=len(tlabels),
        nrows=2,
        figsize=(16, 4),
    )

    for i, tstr in enumerate(tlabels):
        if tstr in ("6P8",):
            ton = von[von["torsions"] == "toff"]
        else:
            ton = von[von["torsions"] == "ton"]

        tdata = ton[ton["topology"] == tstr]
        min_e = min(tdata["energy_per_bb"])
        max_e = max(tdata["energy_per_bb"])
        low_e = tdata[tdata["energy_per_bb"] == min_e].iloc[0]
        high_e = tdata[tdata["energy_per_bb"] == max_e].iloc[0]

        logging.info(
            f"low E: {str(low_e.cage_name)}; "
            f"E={round(low_e.energy_per_bb, 2)}"
        )
        logging.info(
            f"high E: {str(high_e.cage_name)}; "
            f"E={round(high_e.energy_per_bb, 2)}"
        )
        high_e_name = str(high_e["index"])
        low_e_name = str(low_e["index"])

        add_structure_to_ax(
            ax=axs[0][i],
            struct_name=low_e_name,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=min_e,
            settings=settings,
        )
        add_structure_to_ax(
            ax=axs[1][i],
            struct_name=high_e_name,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=max_e,
            settings=settings,
        )
        # add_energy_to_ax(
        #     ax=axs[0][i],
        #     energy=min_e,
        # )
        # add_energy_to_ax(
        #     ax=axs[1][i],
        #     energy=max_e,
        # )
        axs[0][i].axis("off")
        axs[1][i].axis("off")
        axs[1][i].text(
            x=0,
            y=0,
            s=convert_topo(tstr),
            fontsize=16,
            transform=axs[1][i].transAxes,
        )

    fig.tight_layout()
    filename = "vlh.pdf"
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def fig2_a(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):
    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    structure_names = (
        "2P4_4C1m0400b0000_2C1c0000a0000",
        "3P6_4C1m0400b0000_2C1c0000a0400",
        "4P8_4C1m0400b0000_2C1c0000a0600",
        "6P12_4C1m0400b0000_2C1c0000a0900",
        "12P24_4C1m0400b0000_2C1c0000a01200",
    )
    tor_opt = "ton"
    ton = all_data[all_data["torsions"] == tor_opt]

    fig, axs = plt.subplots(
        ncols=len(structure_names),
        nrows=1,
        figsize=(16, 4),
    )

    for sname, ax in zip(structure_names, axs):

        tdata = ton[ton["cage_name"] == sname]
        sindex = str(tdata.iloc[0]["index"])
        add_structure_to_ax(
            ax=ax,
            struct_name=sindex,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=None,
            settings=settings,
        )
        ax.axis("off")

    fig.tight_layout()
    filename = "vfig2a.pdf"
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()


def fig2_cd(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):
    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    structure_names = (
        "4P6_3C1n0100b0000_2C1c0000a01800",
        "4P6_3C1n0700b0000_2C1c0000a0700",
        "4P6_3C1n0500b0000_2C1c0000a0200",
        "4P6_3C1n0500b0000_2C1c0000a01300",
    )
    tor_opt = "ton"
    ton = all_data[all_data["torsions"] == tor_opt]

    fig, axs = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(10, 10),
    )
    flat_axs = axs.flatten()

    for sname, ax in zip(structure_names, flat_axs):

        tdata = ton[ton["cage_name"] == sname]
        sindex = str(tdata.iloc[0]["index"])
        add_structure_to_ax(
            ax=ax,
            struct_name=sindex,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=None,
            settings=settings,
        )
        ax.axis("off")

    fig.tight_layout()
    filename = "vfig2cd.pdf"
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()


def si_ar_fig(
    all_data,
    structure_names,
    nrows,
    ncols,
    filename,
    struct_output,
    struct_figure_output,
    figure_output,
):
    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(4, 10),
    )
    flat_axs = axs.flatten()

    for sname, ax in zip(structure_names, flat_axs):
        ton = all_data[all_data["torsions"] == sname[1]]
        tdata = ton[ton["cage_name"] == sname[0]]
        sindex = str(tdata.iloc[0]["index"])
        add_structure_to_ax(
            ax=ax,
            struct_name=sindex,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=min(tdata["energy_per_bb"]),
            settings=settings,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()


def si_ar_fig_gen(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):

    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("2P3_3C1n0000b0000_2C1c0000a01200", "ton"),
            ("2P3_3C1n0000b0000_2C1c0000a0500", "toff"),
            ("2P3_3C1n0100b0000_2C1c0000a01100", "ton"),
            ("2P3_3C1n0100b0000_2C1c0000a0200", "toff"),
            ("2P3_3C1n0200b0000_2C1c0000a01000", "ton"),
            ("2P3_3C1n0200b0000_2C1c0000a0200", "toff"),
            ("2P3_3C1n0300b0000_2C1c0000a0900", "ton"),
            ("2P3_3C1n0300b0000_2C1c0000a0000", "toff"),
            ("2P3_3C1n0400b0000_2C1c0000a0700", "ton"),
            ("2P3_3C1n0400b0000_2C1c0000a0000", "toff"),
            ("2P3_3C1n0500b0000_2C1c0000a0600", "ton"),
            ("2P3_3C1n0500b0000_2C1c0000a0000", "toff"),
            ("2P3_3C1n0600b0000_2C1c0000a0400", "ton"),
            ("2P3_3C1n0600b0000_2C1c0000a0000", "toff"),
            ("2P3_3C1n0700b0000_2C1c0000a0000", "ton"),
            ("2P3_3C1n0700b0000_2C1c0000a0000", "toff"),
        ),
        nrows=8,
        ncols=2,
        filename="2P3_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("4P6_3C1n0000b0000_2C1c0000a01700", "ton"),
            ("4P6_3C1n0000b0000_2C1c0000a0500", "toff"),
            ("4P6_3C1n0100b0000_2C1c0000a01800", "ton"),
            ("4P6_3C1n0100b0000_2C1c0000a0000", "toff"),
            ("4P6_3C1n0200b0000_2C1c0000a01700", "ton"),
            ("4P6_3C1n0200b0000_2C1c0000a0000", "toff"),
            ("4P6_3C1n0300b0000_2C1c0000a01500", "ton"),
            ("4P6_3C1n0300b0000_2C1c0000a0000", "toff"),
            ("4P6_3C1n0400b0000_2C1c0000a01400", "ton"),
            ("4P6_3C1n0400b0000_2C1c0000a0000", "toff"),
            ("4P6_3C1n0500b0000_2C1c0000a01300", "ton"),
            ("4P6_3C1n0500b0000_2C1c0000a0000", "toff"),
            ("4P6_3C1n0600b0000_2C1c0000a01100", "ton"),
            ("4P6_3C1n0600b0000_2C1c0000a0000", "toff"),
            ("4P6_3C1n0700b0000_2C1c0000a0700", "ton"),
            ("4P6_3C1n0700b0000_2C1c0000a0000", "toff"),
        ),
        nrows=8,
        ncols=2,
        filename="4P6_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("4P62_3C1n0000b0000_2C1c0000a01000", "ton"),
            ("4P62_3C1n0000b0000_2C1c0000a0000", "toff"),
            ("4P62_3C1n0100b0000_2C1c0000a01000", "ton"),
            ("4P62_3C1n0100b0000_2C1c0000a0000", "toff"),
            ("4P62_3C1n0200b0000_2C1c0000a0900", "ton"),
            ("4P62_3C1n0200b0000_2C1c0000a0000", "toff"),
            ("4P62_3C1n0300b0000_2C1c0000a0800", "ton"),
            ("4P62_3C1n0300b0000_2C1c0000a0000", "toff"),
            ("4P62_3C1n0400b0000_2C1c0000a0800", "ton"),
            ("4P62_3C1n0400b0000_2C1c0000a0000", "toff"),
            ("4P62_3C1n0500b0000_2C1c0000a0600", "ton"),
            ("4P62_3C1n0500b0000_2C1c0000a0000", "toff"),
            ("4P62_3C1n0600b0000_2C1c0000a0700", "ton"),
            ("4P62_3C1n0600b0000_2C1c0000a0200", "ton"),
            ("4P62_3C1n0700b0000_2C1c0000a0500", "ton"),
            ("4P62_3C1n0700b0000_2C1c0000a0000", "toff"),
        ),
        nrows=8,
        ncols=2,
        filename="4P62_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("6P9_3C1n0000b0000_2C1c0000a01400", "ton"),
            ("6P9_3C1n0000b0000_2C1c0000a0000", "toff"),
            ("6P9_3C1n0100b0000_2C1c0000a01500", "ton"),
            ("6P9_3C1n0100b0000_2C1c0000a0000", "toff"),
            ("6P9_3C1n0200b0000_2C1c0000a01600", "ton"),
            ("6P9_3C1n0200b0000_2C1c0000a0000", "toff"),
            ("6P9_3C1n0300b0000_2C1c0000a01600", "ton"),
            ("6P9_3C1n0300b0000_2C1c0000a0700", "ton"),
            ("6P9_3C1n0400b0000_2C1c0000a01400", "ton"),
            ("6P9_3C1n0400b0000_2C1c0000a0000", "toff"),
            ("6P9_3C1n0500b0000_2C1c0000a01400", "ton"),
            ("6P9_3C1n0500b0000_2C1c0000a0000", "toff"),
            ("6P9_3C1n0600b0000_2C1c0000a01300", "ton"),
            ("6P9_3C1n0600b0000_2C1c0000a0100", "ton"),
            ("6P9_3C1n0700b0000_2C1c0000a01000", "ton"),
            ("6P9_3C1n0700b0000_2C1c0000a0000", "toff"),
        ),
        nrows=8,
        ncols=2,
        filename="6P9_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("8P12_3C1n0000b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0000b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0100b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0100b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0200b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0200b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0300b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0300b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0400b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0400b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0500b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0500b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0600b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0600b0000_2C1c0000a01000", "toff"),
            ("8P12_3C1n0700b0000_2C1c0000a01000", "ton"),
            ("8P12_3C1n0700b0000_2C1c0000a01000", "toff"),
        ),
        nrows=8,
        ncols=2,
        filename="8P12_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )

    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("2P4_4C1m0000b0000_2C1c0000a01100", "ton"),
            ("2P4_4C1m0000b0000_2C1c0000a0500", "toff"),
            ("2P4_4C1m0100b0000_2C1c0000a0900", "ton"),
            ("2P4_4C1m0100b0000_2C1c0000a0000", "toff"),
            ("2P4_4C1m0200b0000_2C1c0000a0700", "ton"),
            ("2P4_4C1m0200b0000_2C1c0000a0000", "toff"),
            ("2P4_4C1m0300b0000_2C1c0000a0500", "ton"),
            ("2P4_4C1m0300b0000_2C1c0000a0000", "toff"),
            ("2P4_4C1m0400b0000_2C1c0000a0000", "ton"),
            ("2P4_4C1m0400b0000_2C1c0000a0000", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="2P4_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("3P6_4C1m0000b0000_2C1c0000a01300", "ton"),
            ("3P6_4C1m0000b0000_2C1c0000a0300", "toff"),
            ("3P6_4C1m0100b0000_2C1c0000a01200", "ton"),
            ("3P6_4C1m0100b0000_2C1c0000a01200", "toff"),
            ("3P6_4C1m0200b0000_2C1c0000a01000", "ton"),
            ("3P6_4C1m0200b0000_2C1c0000a0800", "toff"),
            ("3P6_4C1m0300b0000_2C1c0000a0900", "ton"),
            ("3P6_4C1m0300b0000_2C1c0000a0000", "toff"),
            ("3P6_4C1m0400b0000_2C1c0000a0400", "ton"),
            ("3P6_4C1m0400b0000_2C1c0000a0400", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="3P6_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("4P8_4C1m0000b0000_2C1c0000a01200", "ton"),
            ("4P8_4C1m0000b0000_2C1c0000a0800", "toff"),
            ("4P8_4C1m0100b0000_2C1c0000a01200", "ton"),
            ("4P8_4C1m0100b0000_2C1c0000a0700", "toff"),
            ("4P8_4C1m0200b0000_2C1c0000a01000", "ton"),
            ("4P8_4C1m0200b0000_2C1c0000a0500", "ton"),
            ("4P8_4C1m0300b0000_2C1c0000a01000", "ton"),
            ("4P8_4C1m0300b0000_2C1c0000a0000", "toff"),
            ("4P8_4C1m0400b0000_2C1c0000a0600", "ton"),
            ("4P8_4C1m0400b0000_2C1c0000a0000", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="4P8_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("6P12_4C1m0000b0000_2C1c0000a01600", "ton"),
            ("6P12_4C1m0000b0000_2C1c0000a01600", "toff"),
            ("6P12_4C1m0100b0000_2C1c0000a0000", "ton"),
            ("6P12_4C1m0100b0000_2C1c0000a01800", "ton"),
            ("6P12_4C1m0200b0000_2C1c0000a0200", "ton"),
            ("6P12_4C1m0200b0000_2C1c0000a01600", "ton"),
            ("6P12_4C1m0300b0000_2C1c0000a0400", "ton"),
            ("6P12_4C1m0300b0000_2C1c0000a01400", "ton"),
            ("6P12_4C1m0400b0000_2C1c0000a0900", "ton"),
            ("6P12_4C1m0400b0000_2C1c0000a0900", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="6P12_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("8P16_4C1m0000b0000_2C1c0000a01500", "ton"),
            ("8P16_4C1m0000b0000_2C1c0000a01400", "toff"),
            ("8P16_4C1m0100b0000_2C1c0000a01600", "ton"),
            ("8P16_4C1m0100b0000_2C1c0000a0900", "toff"),
            ("8P16_4C1m0200b0000_2C1c0000a01600", "ton"),
            ("8P16_4C1m0200b0000_2C1c0000a0000", "toff"),
            ("8P16_4C1m0300b0000_2C1c0000a01500", "ton"),
            ("8P16_4C1m0300b0000_2C1c0000a0200", "toff"),
            ("8P16_4C1m0400b0000_2C1c0000a01000", "ton"),
            ("8P16_4C1m0400b0000_2C1c0000a0400", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="8P16_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
    si_ar_fig(
        all_data=all_data,
        structure_names=(
            ("12P24_4C1m0000b0000_2C1c0000a0200", "ton"),
            ("12P24_4C1m0000b0000_2C1c0000a01200", "toff"),
            ("12P24_4C1m0100b0000_2C1c0000a01500", "ton"),
            ("12P24_4C1m0100b0000_2C1c0000a0000", "toff"),
            ("12P24_4C1m0200b0000_2C1c0000a01600", "ton"),
            ("12P24_4C1m0200b0000_2C1c0000a0000", "toff"),
            ("12P24_4C1m0300b0000_2C1c0000a01200", "ton"),
            ("12P24_4C1m0300b0000_2C1c0000a0000", "toff"),
            ("12P24_4C1m0400b0000_2C1c0000a01200", "ton"),
            ("12P24_4C1m0400b0000_2C1c0000a0000", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="12P24_ar.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )


def si_shape_fig(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):

    structure_names = (
        # ("12P24_4C1m0000b0000_2C1c0000a0200", "ton"),
        # ("12P24_4C1m0000b0000_2C1c0000a01200", "toff"),
        # ("12P24_4C1m0100b0000_2C1c0000a01500", "ton"),
        # ("12P24_4C1m0100b0000_2C1c0000a0000", "toff"),
        # ("12P24_4C1m0200b0000_2C1c0000a01600", "ton"),
        # ("12P24_4C1m0200b0000_2C1c0000a0000", "toff"),
        # ("12P24_4C1m0300b0000_2C1c0000a01200", "ton"),
        # ("12P24_4C1m0300b0000_2C1c0000a0000", "toff"),
        # ("12P24_4C1m0400b0000_2C1c0000a01200", "ton"),
        ("4P6_3C1n0700b0000_2C1c0000a0200", "ton"),
        ("4P6_3C1n0700b0000_2C1c0000a0200", "toff"),
        ("4P6_3C1n0700b0000_2C1c0000a0700", "ton"),
        ("4P6_3C1n0700b0000_2C1c0000a0700", "toff"),
        ("4P6_3C1n0100b0000_2C1c0000a0800", "ton"),
        ("4P6_3C1n0100b0000_2C1c0000a0800", "toff"),
        ("4P62_3C1n0000b0000_2C1c0000a01300", "ton"),
        ("4P62_3C1n0000b0000_2C1c0000a01300", "toff"),
        ("4P62_3C1n0700b0000_2C1c0000a0300", "ton"),
        ("4P62_3C1n0700b0000_2C1c0000a0300", "toff"),
        ("4P62_3C1n0300b0000_2C1c0000a0300", "ton"),
        ("4P62_3C1n0300b0000_2C1c0000a0300", "toff"),
        ("6P12_4C1m0400b0000_2C1c0000a0900", "ton"),
        ("6P12_4C1m0400b0000_2C1c0000a0900", "toff"),
        ("6P12_4C1m0300b0000_2C1c0000a0900", "ton"),
        ("6P12_4C1m0300b0000_2C1c0000a0900", "toff"),
        ("6P12_4C1m0200b0000_2C1c0000a0400", "ton"),
        ("6P12_4C1m0200b0000_2C1c0000a0400", "toff"),
        ("6P9_3C1n0000b0000_2C1c0000a0000", "ton"),
        ("6P9_3C1n0000b0000_2C1c0000a0000", "toff"),
        ("6P9_3C1n0100b0000_2C1c0000a0000", "ton"),
        ("6P9_3C1n0100b0000_2C1c0000a0000", "toff"),
        ("6P9_3C1n0200b0000_2C1c0000a0000", "ton"),
        ("6P9_3C1n0200b0000_2C1c0000a0000", "toff"),
        ("4P8_4C1m0400b0000_2C1c0000a0600", "ton"),
        ("4P8_4C1m0400b0000_2C1c0000a0600", "toff"),
        ("4P8_4C1m0300b0000_2C1c0000a0600", "ton"),
        ("4P8_4C1m0300b0000_2C1c0000a0600", "toff"),
        ("4P8_4C1m0200b0000_2C1c0000a0600", "ton"),
        ("4P8_4C1m0200b0000_2C1c0000a0600", "toff"),
    )
    nrows = 5
    ncols = 6
    filename = "shape_structure_fig.pdf"

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(16, 14),
    )
    flat_axs = axs.flatten()

    for sname, ax in zip(structure_names, flat_axs):
        ton = all_data[all_data["torsions"] == sname[1]]
        tdata = ton[ton["cage_name"] == sname[0]]
        sindex = str(tdata.iloc[0]["index"])
        add_structure_to_ax(
            ax=ax,
            struct_name=sindex,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=min(tdata["energy_per_bb"]),
            settings=settings,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figure_output, filename),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()


def add_energy_to_ax(ax, energy):
    if energy <= isomer_energy():
        colorcode = "#345995"
    else:
        # colorcode = "#F9A03F"
        colorcode = "#CA1551"

    ax.set_title(round(energy, 1), fontsize=16, color=colorcode)


def add_structure_to_ax(
    ax,
    struct_name,
    struct_output,
    struct_figure_output,
    settings,
    energy=None,
):

    if "2P3" in struct_name:
        orient_atoms = "C"
    elif "2P4" in struct_name:
        orient_atoms = "Pd"
    else:
        orient_atoms = None

    struct_file = struct_output / f"{struct_name}_optc.mol"
    png_file = generate_image(
        struct_name=struct_name,
        struct_file=struct_file,
        struct_figure_output=struct_figure_output,
        orient_atoms=orient_atoms,
        settings=settings,
    )

    img = mpimg.imread(png_file)
    ax.imshow(img)
    if energy is not None:
        add_energy_to_ax(
            ax=ax,
            energy=energy,
        )


def generate_image(
    struct_name,
    struct_file,
    struct_figure_output,
    orient_atoms,
    settings,
):
    png_file = struct_figure_output / f"{struct_name}_f.png"
    if not os.path.exists(png_file):
        viz = Pymol(
            output_dir=struct_figure_output,
            file_prefix=f"{struct_name}_f",
            settings=settings,
        )
        viz.visualise(
            [struct_file],
            orient_atoms=orient_atoms,
            # big_colour=colorcode,
        )
    return png_file


def webapp_csv(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):
    logging.info("running webapp_csv")

    github_base_url = (
        "https://github.com/andrewtarzia/cgmodels/blob/main/"
        "cg_model_apr2023/"
    )
    github_selfsort_url = github_base_url + "self_sort_outcomes/"

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    csv_files = {"3C1_4C1": {}, "2C1_3C1": {}, "2C1_4C1": {}}

    bbpairs = set(all_data["bbpair"])
    count = 0
    total = len(bbpairs)

    for bbpair in bbpairs:
        logging.info(f"viz self sort of {bbpair} ({count} of {total})")
        bbdata = all_data[all_data["bbpair"] == bbpair]

        bbdict = {}
        for tors in ("ton", "toff"):
            if "3C1" in bbpair and "4C1" in bbpair:
                ncols = nrows = 1
                topo_type = "3C1_4C1"
            elif "2C1" in bbpair and "3C1" in bbpair:
                nrows = 1
                ncols = 5
                topo_type = "2C1_3C1"
            elif "2C1" in bbpair and "4C1" in bbpair:
                nrows = 1
                ncols = 6
                topo_type = "2C1_4C1"

            if topo_type == "3C1_4C1" and tors == "ton":
                bbdict[tors] = {
                    "topologies": "none",
                    "selfsort_url": "none",
                }
                continue

            tdata = bbdata[bbdata["torsions"] == tors]
            bite_angle = next(iter(set(tdata["target_bite_angle"])))
            clangle = next(iter(set(tdata["clangle"])))
            c3angle = next(iter(set(tdata["c3angle"])))

            energies = {
                str(row["topology"]): float(row["energy_per_bb"])
                for i, row in tdata.iterrows()
            }
            index_energies = {
                str(row["index"]): float(row["energy_per_bb"])
                for i, row in tdata.iterrows()
            }

            mixed_energies = {
                i: energies[i]
                for i in energies
                if energies[i] < isomer_energy()
            }

            min_energy = min(energies.values())

            if min_energy > isomer_energy():
                topologies = "none"
            else:
                topologies = "/".join(sorted(mixed_energies))

            vss_output = figure_output / "vss_figures"
            check_directory(vss_output)
            figure_file = os.path.join(
                vss_output, f"vss_{bbpair}_{tors}"
            )
            if not os.path.exists(figure_file):

                if ncols == 1:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    flat_axs = [ax]
                else:
                    fig, axs = plt.subplots(
                        ncols=ncols,
                        nrows=nrows,
                        figsize=(16, 5),
                    )
                    flat_axs = axs.flatten()

                for sindx, ax in zip(
                    sorted(index_energies.keys()), flat_axs
                ):
                    add_structure_to_ax(
                        ax=ax,
                        struct_name=sindx,
                        struct_output=struct_output,
                        struct_figure_output=struct_figure_output,
                        energy=index_energies[sindx],
                        settings=settings,
                    )
                    ax.axis("off")

                fig.tight_layout()
                fig.savefig(
                    figure_file,
                    dpi=120,
                    bbox_inches="tight",
                )
                plt.close()

            selfsort_url = (
                github_selfsort_url + f"vss_{bbpair}_{tors}.png"
            )
            bbdict[tors] = {
                "topologies": topologies,
                "clangle": clangle,
                "bite_angle": bite_angle,
                "c3angle": c3angle,
                "selfsort_url": selfsort_url,
            }

        csv_files[topo_type][bbpair] = bbdict
        count += 1

    for csv_name in csv_files:
        filename = figure_output / f"{csv_name}_bbdata.csv"
        with open(filename, "w") as f:
            f.write(
                "name,clangle,bite_angle,c3angle,ton_topologies,"
                "toff_topologies,"
                "ton_selfsort_url,toff_selfsort_url\n"
            )
            for bbpair in csv_files[csv_name]:
                bbdict = csv_files[csv_name][bbpair]
                f.write(
                    f"{bbpair},{bbdict['toff']['clangle']},"
                    f"{bbdict['toff']['bite_angle']},"
                    f"{bbdict['toff']['c3angle']},"
                    f"{bbdict['ton']['topologies']},"
                    f"{bbdict['toff']['topologies']},"
                    f"{bbdict['ton']['selfsort_url']},"
                    f"{bbdict['toff']['selfsort_url']}"
                    "\n"
                )


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    figure_output = cages() / "ommfigures"
    struct_output = cages() / "ommstructures"
    struct_figure_output = cages() / "ommfigures" / "structures"
    check_directory(struct_figure_output)
    calculation_output = cages() / "ommcalculations"

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=calculation_output,
    )
    low_e_data = get_lowest_energy_data(
        all_data=all_data,
        output_dir=calculation_output,
    )
    logging.info(f"there are {len(all_data)} collected data")
    write_out_mapping(all_data)

    generate_images_of_all(
        low_e_data,
        struct_output,
        struct_figure_output,
    )

    si_shape_fig(
        low_e_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    visualise_low_and_high(
        low_e_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    fig2_a(
        low_e_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    fig2_cd(
        low_e_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    si_ar_fig_gen(
        low_e_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )

    webapp_csv(
        low_e_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    raise SystemExit(
        "want to print out problematic structures, e.g. a-a distances "
        "of zero, or nulll angles"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
