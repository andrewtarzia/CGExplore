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
    cltypetopo_to_colormap,
    write_out_mapping,
    get_lowest_energy_data,
)
from visualisation import Pymol
from utilities import check_directory


def visualise_bite_angle(all_data, figure_output):
    raise NotImplementedError()
    struct_output = cages() / "structures"
    topologies = (
        "TwoPlusThree",
        "FourPlusSix",
        "FourPlusSix2",
        "SixPlusNine",
        "EightPlusTwelve",
        "TwoPlusFour",
        "ThreePlusSix",
        "FourPlusEight",
        "SixPlusTwelve",
        "M12L24",
    )

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    trim = all_data[all_data["clr0"] == 2]
    trim = trim[trim["c2r0"] == 5]

    for tstr in topologies:
        tdata = trim[trim["topology"] == tstr]
        clangles = set(tdata["clangle"])

        for clangle in clangles:
            fig, axs = plt.subplots(
                ncols=19,
                nrows=2,
                figsize=(16, 5),
            )
            cdata = tdata[tdata["clangle"] == clangle]

            for i, tors in enumerate(("ton", "toff")):
                flat_axs = axs[i].flatten()

                show = cdata[cdata["torsions"] == tors]
                names_energies = [
                    (
                        str(row["index"]),
                        float(row["energy"]),
                        float(row["target_bite_angle"]),
                    )
                    for idx, row in show.iterrows()
                ]
                names_energies = sorted(
                    names_energies, key=lambda tup: tup[2]
                )

                for cage_data, ax in zip(names_energies, flat_axs):
                    name, energy, ba = cage_data
                    structure_file = struct_output / f"{name}_optc.mol"
                    structure_colour = colour_by_energy(energy)
                    png_file = figure_output / f"{name}_f.png"
                    if not os.path.exists(png_file):
                        viz = Pymol(
                            output_dir=figure_output,
                            file_prefix=f"{name}_f",
                            settings=settings,
                        )
                        viz.visualise(
                            [structure_file], [structure_colour]
                        )

                    img = mpimg.imread(png_file)
                    ax.imshow(img)
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(f"{ba}", fontsize=16)

            ax.plot(
                [None, None],
                [None, None],
                c=colour_by_energy(max_energy() + 1),
                lw=3,
                label=f"energy per bond > {max_energy()}eV",
            )
            ax.plot(
                [None, None],
                [None, None],
                c=colour_by_energy(isomer_energy() + 0.1),
                lw=3,
                label=f"energy per bond <= {max_energy()}eV",
            )
            ax.plot(
                [None, None],
                [None, None],
                c=colour_by_energy(0),
                lw=3,
                label=f"energy per bond <= {isomer_energy()}eV",
            )

            fig.legend(
                bbox_to_anchor=(0, 1.02, 2, 0.2),
                loc="lower left",
                ncol=3,
                fontsize=16,
            )
            fig.tight_layout()
            filename = f"vba_{tstr}_{clangle}.pdf"
            fig.savefig(
                os.path.join(figure_output, filename),
                dpi=360,
                bbox_inches="tight",
            )
            plt.close()


def visualise_self_sort(all_data, figure_output):
    raise NotImplementedError()
    struct_output = cages() / "structures"
    topology_dict = cltypetopo_to_colormap()

    settings = {
        "grid_mode": 0,
        "rayx": 1000,
        "rayy": 1000,
        "stick_rad": 0.8,
        "vdw": 0,
        "zoom_string": "custom",
    }

    bbpairs = set(all_data["bbpair"])

    for bbpair in bbpairs:
        ctitle = "4C1" if "4C1" in bbpair else "3C1"
        ncols = 5 if ctitle == "3C1" else 6
        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=2,
            figsize=(16, 5),
        )
        bdata = all_data[all_data["bbpair"] == bbpair]

        for i, tors in enumerate(("ton", "toff")):
            flat_axs = axs[i].flatten()

            show = bdata[bdata["torsions"] == tors]
            ctitle = "4C1" if "4C1" in bbpair else "3C1"
            target_bite_angle = float(
                list(show["target_bite_angle"])[0]
            )
            names_energies = []
            for tstr in topology_dict[ctitle]:
                rowdata = show[show["topology"] == tstr]
                if len(rowdata) > 1:
                    raise ValueError(f"{rowdata} too long")
                elif len(rowdata) == 1:
                    row = rowdata.iloc[0]
                    names_energies.append(
                        (str(row["index"]), float(row["energy"])),
                    )
                else:
                    names_energies.append(None)

            for j, cage_data in enumerate(names_energies):
                ax = flat_axs[j]
                ax.axis("off")
                if i == 0 and j == 0:
                    ax.set_title(
                        f"ba: {target_bite_angle}", fontsize=16
                    )
                if cage_data is None:
                    continue
                name, energy = cage_data
                structure_file = struct_output / f"{name}_optc.mol"
                structure_colour = colour_by_energy(energy)
                png_file = figure_output / f"{name}_f.png"
                if not os.path.exists(png_file):
                    viz = Pymol(
                        output_dir=figure_output,
                        file_prefix=f"{name}_f",
                        settings=settings,
                    )
                    viz.visualise([structure_file], [structure_colour])

                img = mpimg.imread(png_file)
                ax.imshow(img)

        ax.plot(
            [None, None],
            [None, None],
            c=colour_by_energy(max_energy() + 1),
            lw=3,
            label=f"energy > {max_energy()}eV",
        )
        ax.plot(
            [None, None],
            [None, None],
            c=colour_by_energy(isomer_energy() + 0.1),
            lw=3,
            label=f"energy <= {max_energy()}eV",
        )
        ax.plot(
            [None, None],
            [None, None],
            c=colour_by_energy(0),
            lw=3,
            label=f"energy <= {isomer_energy()}eV",
        )

        fig.legend(
            bbox_to_anchor=(0, 1.02, 2, 0.2),
            loc="lower left",
            ncol=3,
            fontsize=16,
        )
        fig.tight_layout()
        filename = f"vss_{bbpair}.pdf"
        fig.savefig(
            os.path.join(figure_output, filename),
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()


def visualise_high_energy(all_data, figure_output):
    raise NotImplementedError()
    struct_output = cages() / "structures"
    high_energy = all_data[all_data["energy"] > 500]
    high_e_names = list(high_energy["index"])
    high_e_energies = list(high_energy["energy"])
    logging.info(
        f"there are {len(high_e_names)} high energy structures"
    )
    with open(figure_output / "high_energy_names.txt", "w") as f:
        f.write("_opted3.mol ".join(high_e_names))
        f.write("_opted3.mol")

    cage_name = high_e_names[0]
    structure_files = []
    structure_colours = []
    for i, cage_name in enumerate(high_e_names):
        structure_files.append(struct_output / f"{cage_name}_optc.mol")
        structure_colours.append(colour_by_energy(high_e_energies[i]))
    viz = Pymol(output_dir=figure_output, file_prefix="highe")
    viz.visualise(structure_files, structure_colours)


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

    raise SystemExit(
        "want to print out problematic structures, e.g. a-a distances "
        "of zero, or nulll angles"
    )
    raise SystemExit(
        "use same struct output below and struct_figure_output"
    )

    visualise_self_sort(all_data, figure_output)
    visualise_bite_angle(all_data, figure_output)
    visualise_high_energy(all_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
