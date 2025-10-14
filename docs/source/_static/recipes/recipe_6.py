"""Copiable code from Recipe #5."""  # noqa: INP001

import argparse
import logging
import pathlib
from collections import abc, defaultdict

import matplotlib.pyplot as plt

import cgexplore as cgx

logger = logging.getLogger(__name__)


def make_topt_plot(
    database_dir: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    parameter_sets: list[abc.Sequence[str]],
    chosen_name: str,
) -> dict:
    """Visualise energies."""
    possible_modifiable = ["nb", "bnb", "bac", "ba", "ac"]
    units = {
        "nb": r"$\mathrm{\AA}$",
        "bnb": "$^\\circ$",
        "bac": "$^\\circ$",
        "ba": r"$\mathrm{\AA}$",
        "ac": r"$\mathrm{\AA}$",
    }
    colours = {
        ("bac",): "tab:blue",
        ("bac", "ba"): "tab:orange",
        ("ba", "ac"): "tab:green",
        ("bac", "ba", "nb", "bnb"): "tab:red",
        ("bnb",): "tab:purple",
        ("bnb", "nb"): "tab:brown",
        ("nb", "ba"): "tab:pink",
    }

    fig, axs = plt.subplots(
        ncols=len(possible_modifiable),
        sharey=True,
        figsize=(16, 5),
    )
    flat_axs = axs.flatten()

    for ps, parameters in enumerate(parameter_sets):
        database_path = database_dir / f"set_{ps}.db"
        entry = cgx.utilities.AtomliteDatabase(database_path).get_entry(
            chosen_name
        )

        if "optimisation_energy_per_bb" not in entry.properties:
            raise RuntimeError

        term_dict = {
            term: entry.properties["optimisation_x"][int(i)]
            for i, term in entry.properties["optimisation_map"].items()
        }

        ffdict = entry.properties["forcefield_dict"]["v_dict"]
        init_term_dict = {
            term: ffdict["_".join(list(term))] for term in term_dict
        }

        init_term_dict.update(
            {
                term: ffdict["_".join(list(term))]
                for term in possible_modifiable
                if term not in init_term_dict
            }
        )

        term_dict.update(
            {
                term: ffdict["_".join(list(term))]
                for term in possible_modifiable
                if term not in term_dict
            }
        )

        for i, ax in enumerate(flat_axs):
            term_key = possible_modifiable[i]
            if ps == 0:
                ax.axvline(x=init_term_dict[term_key], c="gray", zorder=-1)
            ax.scatter(
                term_dict[term_key],
                entry.properties["optimisation_energy_per_bb"],
                c=colours[parameters],
                alpha=1,
                ec="k",
                s=80,
                zorder=2,
            )
            ax.plot(
                (init_term_dict[term_key], term_dict[term_key]),
                (
                    entry.properties["optimisation_energy_per_bb"],
                    entry.properties["optimisation_energy_per_bb"],
                ),
                c=colours[parameters],
                alpha=1,
                lw=1,
                zorder=-2,
                marker="s",
                markersize=3,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel(f"${term_key}$ [{units[term_key]}]", fontsize=16)

            ax.set_yscale("log")
            if i == 0:
                ax.set_ylabel(r"final $E_{\mathrm{b}}$", fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=360, bbox_inches="tight")
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()
    # Define working directories.
    wd = (
        pathlib.Path(__file__).resolve().parent
        / ".."
        / ".."
        / "recipes"
        / "recipe_6_output"
    )
    cgx.utilities.check_directory(wd)
    struct_output = wd / "structures"
    cgx.utilities.check_directory(struct_output)
    calc_dir = wd / "calculations"
    cgx.utilities.check_directory(calc_dir)
    data_dir = wd / "data"
    cgx.utilities.check_directory(data_dir)
    figure_dir = wd / "figures"
    cgx.utilities.check_directory(figure_dir)

    # Define a definer dictionary.
    # These are constants, while different systems can override these
    # parameters.
    cg_scale = 2
    constant_definer_dict = {
        # Bonds.
        "nb": ("bond", 1.0, 1e5),
        # Angles.
        "aca": ("angle", 180, 1e2),
        "nba": ("angle", 180, 1e2),
        # Nonbondeds.
        "n": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
    }

    # Define beads.
    bead_library = cgx.molecular.BeadLibrary.from_bead_types(
        # Type and coordination.
        {"n": 3, "a": 2, "b": 2, "c": 2}
    )

    # Define your forcefield alterations as building blocks.
    building_block_library = {
        "ditopic": {
            "precursor": cgx.molecular.TwoC1Arm(
                bead=bead_library.get_from_type("c"),
                abead1=bead_library.get_from_type("a"),
            ),
            "mod_definer_dict": {
                "ba": ("bond", 1.5 / cg_scale, 1e5),
                "ac": ("bond", 1.5 / 2 / cg_scale, 1e5),
                "bac": ("angle", 115, 1e2),
            },
        },
        "tritopic": {
            "precursor": cgx.molecular.ThreeC1Arm(
                bead=bead_library.get_from_type("n"),
                abead1=bead_library.get_from_type("b"),
            ),
            "mod_definer_dict": {
                "nb": ("bond", 3.0 / cg_scale, 1e5),
                "bnb": ("angle", 120, 1e2),
            },
        },
    }

    # Define systems to predict the structure of.
    systems = {
        "cc3": {
            "stoichiometry_map": {"tritopic": 2, "ditopic": 3},
            "multipliers": (2,),
            "vdw_cutoff": 2,
        },
    }

    # Define a series of parameter explorations.
    parameter_sets = [
        ("bac",),
        ("bac", "ba"),
        ("ba", "ac"),
        ("bac", "ba", "nb", "bnb"),
        ("bnb",),
        ("bnb", "nb"),
        ("nb", "ba"),
    ]
    if args.run:
        for system_name, syst_d in systems.items():
            logger.info("doing system: %s", system_name)
            # Merge constant dict with modifications from different systems.
            merged_definer_dicts = (
                cgx.systems_optimisation.merge_definer_dicts(
                    original_definer_dict=constant_definer_dict,
                    new_definer_dicts=[
                        building_block_library[i]["mod_definer_dict"]
                        for i in syst_d["stoichiometry_map"]
                    ],
                )
            )

            forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
                identifier=f"{system_name}ff",
                prefix=f"{system_name}ff",
                vdw_bond_cutoff=syst_d["vdw_cutoff"],
                present_beads=bead_library.get_present_beads(),
                definer_dict=merged_definer_dicts,
            )

            # A structure i have predicted earlier (using the same approach as
            # recipe 2/5).
            chosen_name = "cc3_2_4"
            conformer_db_path = calc_dir / f"{chosen_name}.db"
            conformer_db = cgx.utilities.AtomliteDatabase(conformer_db_path)
            min_energy_structure = None
            min_energy = float("inf")
            for entry in conformer_db.get_entries():
                if entry.properties["energy_per_bb"] < min_energy:
                    min_energy = entry.properties["energy_per_bb"]
                    min_energy_structure = conformer_db.get_molecule(
                        key=entry.key
                    )
                    num_bbs = entry.properties["num_bbs"]

            for ps, parameters in enumerate(parameter_sets):
                logger.info("doing %s", parameters)
                database_path = data_dir / f"set_{ps}.db"
                ffoptcalculation_dir = calc_dir / f"set_{ps}"
                ffoptcalculation_dir.mkdir(exist_ok=True)
                # To database.
                cgx.utilities.AtomliteDatabase(database_path).add_molecule(
                    key=chosen_name,
                    molecule=min_energy_structure,
                )
                cgx.utilities.AtomliteDatabase(database_path).add_properties(
                    key=chosen_name,
                    property_dict={
                        "energy_per_bb": min_energy,
                        "num_bbs": num_bbs,
                        "forcefield_dict": (
                            forcefield.get_forcefield_dictionary()
                        ),
                    },
                )
                cgx.scram.target_optimisation(
                    database_path=database_path,
                    target_key=chosen_name,
                    calculation_dir=ffoptcalculation_dir,
                    definer_dict=merged_definer_dicts,
                    modifiable_terms=parameters,
                    forcefield=forcefield,
                )

    # Now let's make a plot given those datasets.
    fig, ax = plt.subplots(figsize=(8, 5))
    properties = defaultdict(list)
    structures = []
    chosen_name = "cc3_2_4"
    for ps, parameters in enumerate(parameter_sets):  # noqa: B007
        database_path = data_dir / f"set_{ps}.db"
        ffoptcalculation_dir = calc_dir / f"set_{ps}"

        database = cgx.utilities.AtomliteDatabase(database_path)
        ff_database = cgx.utilities.AtomliteDatabase(
            ffoptcalculation_dir / f"{chosen_name}_ffopt_ffopt.db"
        )

        entry = database.get_entry(key=chosen_name)
        if len(structures) == 0:
            # Add the input.
            structures.append(database.get_molecule(key=chosen_name))
            properties["E_b / kjmol-1"].append(
                entry.properties["energy_per_bb"]
            )
            properties["rmsd / AA"].append(0)

        # Add the final structure and rmsd to chemiscope.
        structures.append(ff_database.get_molecule(key=chosen_name))
        properties["E_b / kjmol-1"].append(
            entry.properties["optimisation_energy_per_bb"]
        )
        properties["rmsd / AA"].append(entry.properties["optimisation_rmsd"])

        ax.plot(
            entry.properties["optimisation_energies"],
            alpha=1.0,
            ms=3,
            marker="o",
            label=f"set_{ps}",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("step", fontsize=16)
    ax.set_ylabel(r"$E_{\mathrm{b}}$ [AA]", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        figure_dir / "recipe_6_image.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    logger.info("saving %s entries", len(structures))
    cgx.utilities.write_chemiscope_json(
        json_file=data_dir / "opt_structures.json.gz",
        structures=structures,
        properties=properties,
        bonds_as_shapes=True,
        meta_dict={
            "name": "Recipe 6 structures.",
            "description": (
                "Parameter optimised minimal models from recipe 6."
            ),
            "authors": ["Andrew Tarzia"],
            "references": [],
        },
        x_axis_dict={"property": "rmsd / AA"},
        y_axis_dict={"property": "E_b / kjmol-1"},
        z_axis_dict={"property": ""},
        color_dict={"property": "E_b / kjmol-1", "min": 0, "max": 1.0},
        bond_hex_colour="#919294",
    )

    make_topt_plot(
        database_dir=data_dir,
        figure_dir=figure_dir,
        chosen_name=chosen_name,
        filename="recipe_6_image_2.png",
        parameter_sets=parameter_sets,
    )


if __name__ == "__main__":
    main()
