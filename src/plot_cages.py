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
    max_energy,
    data_to_array,
    cltypetopo_to_colormap,
    write_out_mapping,
)
from visualisation import Pymol


def colour_by_energy(energy):
    if energy <= isomer_energy():
        colorcode = "#345995"
    elif energy <= max_energy():
        colorcode = "#F9A03F"
    else:
        colorcode = "#CA1551"

    return colorcode


def visualise_bite_angle(all_data, figure_output):
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

    trim = all_data[all_data["clsigma"] == 2]
    trim = trim[trim["c2sigma"] == 5]

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
                        float(row["c2angle"]),
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
            c2angle = float(list(show["c2angle"])[0])
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
                    ax.set_title(f"ba: {c2angle}", fontsize=16)
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

    visualise_self_sort(opt_data, figure_output)
    visualise_bite_angle(opt_data, figure_output)
    visualise_high_energy(opt_data, figure_output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
