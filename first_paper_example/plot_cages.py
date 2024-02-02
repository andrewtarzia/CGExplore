#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Script to visulize structures in matplotlib.

Author: Andrew Tarzia

"""

import logging
import math
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from analysis import (
    convert_topo,
    data_to_array,
    get_paired_cage_name,
    isomer_energy,
    mapshape_to_topology,
    topology_labels,
)
from cgexplore.utilities import check_directory
from cgexplore.visualisation import Pymol
from env_set import (
    calculations,
    figures,
    outputdata,
    pymol_path,
    structures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


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

    for _i, row in von.iterrows():
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
            f"low E: {low_e.cage_name!s}; "
            f"E={round(low_e.energy_per_bb, 2)}"
        )
        logging.info(
            f"high E: {high_e.cage_name!s}; "
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

        axs[0][i].axis("off")
        axs[1][i].axis("off")

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

    for sname, ax in zip(structure_names, axs, strict=True):
        tdata = ton[ton["cage_name"] == naming_convention_map(sname, "ton")]
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
        "4P6_3C1n0400b0000_2C1c0000a0000",
        "4P6_3C1n0500b0000_2C1c0000a0100",
        "4P6_3C1n0600b0000_2C1c0000a0300",
        "4P6_3C1n0700b0000_2C1c0000a0700",
        "4P6_3C1n0600b0000_2C1c0000a01100",
        "4P6_3C1n0500b0000_2C1c0000a01300",
        "4P6_3C1n0400b0000_2C1c0000a01400",
        "4P6_3C1n0300b0000_2C1c0000a01500",
        "4P6_3C1n0200b0000_2C1c0000a01700",
        "4P6_3C1n0100b0000_2C1c0000a01800",
    )
    tor_opt = "ton"
    ton = all_data[all_data["torsions"] == tor_opt]

    fig, axs = plt.subplots(
        ncols=5,
        nrows=2,
        figsize=(16, 8),
    )
    flat_axs = axs.flatten()

    for sname, ax in zip(structure_names, flat_axs, strict=True):
        tdata = ton[ton["cage_name"] == naming_convention_map(sname, "ton")]
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
    figsize=None,
    titles=None,
):
    if figsize is None:
        figsize = (4, 10)

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
        figsize=figsize,
    )
    flat_axs = axs.flatten()

    for i, (sname, ax) in enumerate(zip(structure_names, flat_axs)):
        ton = all_data[all_data["torsions"] == sname[1]]
        tdata = ton[
            ton["cage_name"] == naming_convention_map(sname[0], sname[1])
        ]
        sindex = str(tdata.iloc[0]["index"])

        title = None if titles is None else titles[i]

        add_structure_to_ax(
            ax=ax,
            struct_name=sindex,
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            energy=min(tdata["energy_per_bb"]),
            settings=settings,
            title=title,
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
            ("6P8_4C1m0000b0000_3C1n0700b0000", "toff"),
            ("6P8_4C1m0100b0000_3C1n0700b0000", "toff"),
            ("6P8_4C1m0200b0000_3C1n0600b0000", "toff"),
            ("6P8_4C1m0300b0000_3C1n0500b0000", "toff"),
            ("6P8_4C1m0400b0000_3C1n0100b0000", "toff"),
        ),
        nrows=5,
        ncols=1,
        filename="6P8_cr.pdf",
        struct_output=struct_output,
        struct_figure_output=struct_figure_output,
        figure_output=figure_output,
    )
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
            ("4P82_4C1m0000b0000_2C1c0000a01500", "ton"),
            ("4P82_4C1m0000b0000_2C1c0000a0500", "toff"),
            ("4P82_4C1m0100b0000_2C1c0000a01400", "ton"),
            ("4P82_4C1m0100b0000_2C1c0000a01400", "toff"),
            ("4P82_4C1m0200b0000_2C1c0000a01300", "ton"),
            ("4P82_4C1m0200b0000_2C1c0000a01000", "toff"),
            ("4P82_4C1m0300b0000_2C1c0000a01100", "ton"),
            ("4P82_4C1m0300b0000_2C1c0000a0100", "ton"),
            ("4P82_4C1m0400b0000_2C1c0000a0600", "ton"),
            ("4P82_4C1m0400b0000_2C1c0000a0600", "toff"),
        ),
        nrows=5,
        ncols=2,
        filename="4P82_ar.pdf",
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
        tdata = ton[
            ton["cage_name"] == naming_convention_map(sname[0], sname[1])
        ]
        sindex = str(tdata.iloc[0]["index"])
        tstr = str(tdata.iloc[0]["topology"])

        for shape_type in ("n", "l"):
            try:
                shape = mapshape_to_topology(shape_type, False)[tstr]
            except KeyError:
                continue

            c_column = f"{shape_type}_{shape}"
            svalue = tdata.iloc[0][c_column]
            logging.info(f"{sindex}: {shape_type}: {shape}, {svalue}")

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
    colorcode = "#345995" if energy <= isomer_energy() else "#CA1551"

    ax.set_title(round(energy, 1), fontsize=16, color=colorcode)


def add_text_to_ax(ax, text):
    x = 0.05
    y = 0.25
    ax.text(x=x, y=y, s=text, fontsize=16, transform=ax.transAxes)


def add_structure_to_ax(
    ax,
    struct_name,
    struct_output,
    struct_figure_output,
    settings,
    energy=None,
    title=None,
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
    if title is not None:
        add_text_to_ax(
            ax=ax,
            text=title,
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
            pymol_path=pymol_path(),
        )
        viz.visualise(
            [struct_file],
            orient_atoms=orient_atoms,
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
        "cg_model_jul2023/"
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
                ncols = 7
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
            c2angle = next(iter(set(tdata["c2angle"])))
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
            figure_file = os.path.join(vss_output, f"vss_{bbpair}_{tors}.png")
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
                    sorted(index_energies.keys()), flat_axs, strict=True
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

            selfsort_url = github_selfsort_url + f"vss_{bbpair}_{tors}.png"
            bbdict[tors] = {
                "topologies": topologies,
                "clangle": clangle,
                "bite_angle": bite_angle,
                "c3angle": c3angle,
                "c2angle": c2angle,
                "selfsort_url": selfsort_url,
            }

        csv_files[topo_type][bbpair] = bbdict
        count += 1

    for csv_name in csv_files:
        filename = figure_output / f"{csv_name}_bbdata.csv"
        with open(filename, "w") as f:
            f.write(
                "name,clangle,bite_angle,c2angle,c3angle,"
                "restricted topologies,unrestricted topologies,"
                "restricted URL,unrestricted URL\n"
            )
            for bbpair in csv_files[csv_name]:
                bbdict = csv_files[csv_name][bbpair]
                f.write(
                    f"{bbpair},{bbdict['toff']['clangle']},"
                    f"{bbdict['toff']['bite_angle']},"
                    f"{bbdict['toff']['c2angle']},"
                    f"{bbdict['toff']['c3angle']},"
                    f"{bbdict['ton']['topologies']},"
                    f"{bbdict['toff']['topologies']},"
                    f"{bbdict['ton']['selfsort_url']},"
                    f"{bbdict['toff']['selfsort_url']}"
                    "\n"
                )


def check_odd_outcomes(
    all_data,
    figure_output,
    struct_output,
    struct_figure_output,
):
    topologies = [i for i in topology_labels(short="P") if i != "6P8"]

    for tstr in topologies:
        tdata = all_data[all_data["topology"] == tstr]
        outcomes = []

        ton_data = tdata[tdata["torsions"] == "ton"]
        cage_names = set(ton_data["cage_name"])

        for cage_name in sorted(cage_names):
            cdata = ton_data[ton_data["cage_name"] == cage_name]
            pair_name = get_paired_cage_name(cage_name)
            pdata = tdata[tdata["cage_name"] == pair_name]
            ton_energy = float(cdata["energy_per_bb"].iloc[0])
            toff_energy = float(pdata["energy_per_bb"].iloc[0])

            # Ignore rounding errors in near zero cases.
            if ton_energy < 1e-1:
                continue
            # Not interested in high energy states, which are just a
            # mess.
            if toff_energy > isomer_energy() * 5:
                continue
            if toff_energy - ton_energy > 0.001:
                ba = int(next(iter(cdata["target_bite_angle"])))
                clangle = int(next(iter(cdata["clangle"])))
                tonlbl = f"{convert_topo(tstr)}:{ba}:{clangle}:rest."
                tofflbl = f"{convert_topo(tstr)}:{ba}:{clangle}:not rest."
                logging.info(
                    f"for {cage_name}: ton: {ton_energy}, "
                    f"toff: {toff_energy}"
                )
                outcomes.append((cage_name, "ton", tonlbl))
                outcomes.append((pair_name, "toff", tofflbl))

        logging.info(f"{tstr}: {len(outcomes)} odd outcomes")
        if len(outcomes) == 0:
            continue
        si_ar_fig(
            all_data=all_data,
            structure_names=outcomes,
            nrows=math.ceil(len(outcomes) / 4),
            ncols=4,
            filename=f"odd_outcomes_{tstr}.pdf",
            struct_output=struct_output,
            struct_figure_output=struct_figure_output,
            figure_output=figure_output,
            figsize=(16, 10),
            titles=[i[2] for i in outcomes],
        )


def generate_movies(figure_output):
    logging.info("running generate_movies")
    vss_output = figure_output / "vss_figures"
    astr = [f"a0{i}00" for i in range(19)]

    for cltopo in ("3C1", "4C1"):
        if cltopo == "3C1":
            sequence = [f"n0{i}00" for i in range(8)]
        elif cltopo == "4C1":
            sequence = [f"m0{i}00" for i in range(5)]

        for clseq in sequence:
            for tors in ("ton", "toff"):
                files = [
                    vss_output
                    / f"vss_{cltopo}{clseq}b00002C1c0000{i}_{tors}.png"
                    for i in astr
                ]
                output_file = f"vss_{cltopo}{clseq}b00002C1c0000a_{tors}.mkv"
                logging.info(f"gen movie to {output_file}")
                output_file = figure_output / output_file
                concat_file = (
                    figure_output
                    / f"vss_{cltopo}{clseq}b00002C1c0000a_{tors}.txt"
                )
                with open(concat_file, "w") as f:
                    for fi in files:
                        f.write(f"file {fi}\n")

                if os.path.exists(output_file):
                    # Delete previous video.
                    os.remove(output_file)

                # Make video.
                ffmpeg_cmd = (
                    "ffmpeg -safe 0 -f concat "
                    f"-i {concat_file} "
                    '-vf "settb=AVTB,setpts=N/2/TB,fps=2"'
                    f" {output_file}"
                )
                os.system(ffmpeg_cmd)


def naming_convention_map(old_name, tors="toff"):
    """This only applies because of the change of naming convention.

    In future, users should stick to the `new` naming convention.

    """
    if "_f" in old_name:
        # You do not need to convert this.
        return old_name

    tstr, bb1, bb2 = old_name.split("_")
    if "4C1" in old_name and "3C1" in old_name:
        bb1name = "4C1m1b1"
        tettopic_count = [f"0{i}00" for i in range(8)].index(
            bb1[3:].split("b")[0].split("m")[-1]
        )
        bb2name = "3C1n1b1"
        tritopic_count = [f"0{i}00" for i in range(8)].index(
            bb2[3:].split("b")[0].split("n")[-1]
        )
        ffid = (tritopic_count * 5) + (tettopic_count)
    elif "4C1" in old_name:
        bb1name = "4C1m1b1"
        tettopic_count = [f"0{i}00" for i in range(8)].index(
            bb1[3:].split("b")[0].split("m")[-1]
        )
        bb2name = "2C1c1a1"
        ditopic_count = [f"0{i}00" for i in range(19)].index(
            bb2[3:].split("a")[-1]
        )
        ffid = (ditopic_count * 5 * 2) + (tettopic_count * 2)
        if tors == "toff":
            ffid += 1
    elif "3C1" in old_name:
        bb1name = "3C1n1b1"
        tritopic_count = [f"0{i}00" for i in range(8)].index(
            bb1[3:].split("b")[0].split("n")[-1]
        )
        bb2name = "2C1c1a1"
        ditopic_count = [f"0{i}00" for i in range(19)].index(
            bb2[3:].split("a")[-1]
        )
        ffid = (ditopic_count * 8 * 2) + (tritopic_count * 2)
        if tors == "toff":
            ffid += 1

    new_name = f"{tstr}_{bb1name}_{bb2name}_f{ffid}"
    logging.info(f"analysing {old_name} as {new_name}")
    return new_name


def main():
    first_line = f"Usage: {__file__}.py"
    if len(sys.argv) != 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = structures()
    figure_output = figures()
    calculation_output = calculations()
    data_output = outputdata()
    struct_figure_output = figures() / "structures"
    check_directory(struct_figure_output)

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=data_output,
    )
    logging.info(f"there are {len(all_data)} collected data")

    generate_images_of_all(
        all_data,
        struct_output,
        struct_figure_output,
    )

    check_odd_outcomes(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    si_shape_fig(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    visualise_low_and_high(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    fig2_a(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    fig2_cd(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    si_ar_fig_gen(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )
    msg = "not implemented from here."
    raise SystemExit(msg)
    webapp_csv(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )

    generate_movies(
        all_data,
        figure_output,
        struct_output,
        struct_figure_output,
    )


if __name__ == "__main__":
    main()
