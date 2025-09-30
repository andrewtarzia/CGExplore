# Distributed under the terms of the MIT License.

"""Script to plot parities.

Author: Andrew Tarzia

"""

import logging
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from analysis import (
    convert_topo,
    data_to_array,
    eb_str,
    isomer_energy,
)
from env_set import cages, calculations, figures, outputdata
from matplotlib.lines import Line2D

from cgexplore.utilities import check_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def energy_parity(
    all_data: pd.DataFrame,
    dupl_data: pd.DataFrame,
    figure_output: pathlib.Path,
) -> None:
    """Plot energy parity."""
    logger.info("running energy_parity")
    print(len(all_data), len(dupl_data))  # noqa: T201

    e_threshold = 0.01

    cmap = {
        "4P6": "#086788",
        "4P62": "#F9A03F",
        "8P12": "#CA1551",
        # "#CA1551"
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    count_ = 0
    for _idx, row in dupl_data.iterrows():
        index_name = str(row["index"])
        tstr = str(row["topology"])
        orig_row = all_data[all_data["index"] == index_name]
        dupl_energy = float(row["energy_per_bb"])
        orig_energy = float(orig_row["energy_per_bb"])
        if dupl_energy < e_threshold and orig_energy < e_threshold:
            continue
        if abs(dupl_energy - orig_energy) > e_threshold:
            print(index_name, dupl_energy, orig_energy)  # noqa: T201
            ax.scatter(
                orig_energy,
                dupl_energy,
                c=cmap[tstr],
                edgecolor="white",
                s=60,
                alpha=0.8,
            )
            count_ += 1
        else:
            ax.scatter(
                orig_energy,
                dupl_energy,
                c="gray",
                edgecolor="none",
                s=20,
                alpha=0.1,
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"first run {eb_str()}", fontsize=16)
    ax.set_ylabel(f"second run {eb_str()}", fontsize=16)
    ax.set_xlim(0.01, 20)
    ax.set_ylim(0.01, 20)
    ax.axhline(y=isomer_energy(), c="k", ls="--", lw=2, alpha=0.2)
    ax.axvline(x=isomer_energy(), c="k", ls="--", lw=2, alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{count_} (of {len(dupl_data)}) far examples", fontsize=16)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=convert_topo(tstr),
            markerfacecolor=cmap[tstr],
            markersize=7,
            markeredgecolor="white",
            alpha=0.8,
        )
        for tstr in cmap
    ]

    ax.legend(handles=legend_elements, fontsize=16, ncol=1)

    fig.tight_layout()
    fig.savefig(
        figure_output / "energy_parity.pdf",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    figure_output = figures()
    calculation_output = calculations()
    data_output = outputdata()
    dupl_calculation_output = cages() / "duplicate_calculations"
    dupl_data_output = cages() / "duplicate_outputdata"
    check_directory(dupl_data_output)

    all_data = data_to_array(
        json_files=calculation_output.glob("*_res.json"),
        output_dir=data_output,
    )

    dupl_all_data = data_to_array(
        json_files=dupl_calculation_output.glob("*_res.json"),
        output_dir=dupl_data_output,
    )

    energy_parity(all_data, dupl_all_data, figure_output)


if __name__ == "__main__":
    main()
