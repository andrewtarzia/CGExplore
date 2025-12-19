"""Copiable code from Recipe #5."""  # noqa: INP001

import argparse
import itertools as it
import logging
import pathlib
from typing import TYPE_CHECKING

import agx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openmm
import stko

import cgexplore as cgx

if TYPE_CHECKING:
    import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def optimisation_workflow(  # noqa: PLR0913
    config_name: str,
    conformer_db_path: pathlib.Path,
    topology_code: cgx.scram.TopologyCode,
    iterator: cgx.scram.TopologyIterator,
    bb_config: cgx.scram.Configuration,
    calculation_dir: pathlib.Path,
    forcefield: cgx.forcefields.ForceField,
) -> None:
    """Geometry optimise a configuration."""
    attempts = (
        (None, None, None),
        ("set", "kamada", 10.0),
        ("set", "spring", 10.0),
        ("set", "spectral", 10.0),
        ("regraphed", "spring", 10.0),
        ("regraphed", "kamada", 10.0),
    )

    for midx, (method, layout_type, scale) in enumerate(attempts):
        name = f"{config_name}_{midx}"

        try:
            if method == "regraphed":
                constructed_molecule = cgx.scram.get_regraphed_molecule(
                    layout_type=layout_type,  # type:ignore[arg-type]
                    scale=scale,  # type:ignore[arg-type]
                    topology_code=topology_code,
                    iterator=iterator,
                    configuration=bb_config,
                )

            elif method == "set":
                constructed_molecule = cgx.scram.get_vertexset_molecule(
                    layout_type=layout_type,  # type:ignore[arg-type]
                    scale=scale,  # type:ignore[arg-type]
                    topology_code=topology_code,
                    iterator=iterator,
                    configuration=bb_config,
                )

            else:
                constructed_molecule = cgx.scram.try_except_construction(
                    iterator=iterator,
                    topology_code=topology_code,
                    configuration=bb_config,
                    vertex_positions=None,
                ).with_centroid(np.array((0.0, 0.0, 0.0)))

        except ValueError:
            continue

        # Optimise and save.
        logger.info("building %s", name)

        try:
            # Check all the other possible mashes.
            potential_names = [
                f"{config_name}_{nmash_idx}"
                for nmash_idx in range(len(attempts))
            ]
            if method is None:
                conformer = cgx.scram.graph_optimise_cage(
                    molecule=constructed_molecule,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=conformer_db_path,
                )

            else:
                conformer = cgx.scram.optimise_cage(
                    molecule=constructed_molecule,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=conformer_db_path,
                    potential_names=potential_names,
                )

            if conformer is not None:
                num_components = len(
                    stko.Network.init_from_molecule(
                        conformer.molecule
                    ).get_connected_components()
                )
                energy_per_bb = cgx.utilities.get_energy_per_bb(
                    energy_decomposition=(conformer.energy_decomposition),
                    number_building_blocks=(
                        iterator.get_num_building_blocks()
                    ),
                )

                properties = {
                    "forcefield_dict": (
                        forcefield.get_forcefield_dictionary()
                    ),
                    "energy_per_bb": energy_per_bb,
                    "num_components": num_components,
                    "mash_idx": midx,
                    "config_name": config_name,
                    "num_bbs": (iterator.get_num_building_blocks()),
                    "topology_code_vmap": tuple(
                        (int(i[0]), int(i[1]))
                        for i in topology_code.vertex_map
                    ),
                    "bb_config_idx": bb_config.idx,
                }
                cgx.utilities.AtomliteDatabase(
                    conformer_db_path
                ).add_properties(
                    key=name,
                    property_dict=properties,  # type:ignore[arg-type]
                )

        except openmm.OpenMMException:
            logger.info("failed optimisation of %s", name)


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    min_energy_key: str,
    conformer_db_path: pathlib.Path,
) -> None:
    """Analyse toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    initial_properties = (
        cgx.utilities.AtomliteDatabase(conformer_db_path)
        .get_entry(min_energy_key)
        .properties
    )

    database.add_properties(key=name, property_dict=initial_properties)
    database.add_properties(key=name, property_dict={"lowest_e_of_mash": True})
    properties = database.get_entry(name).properties

    # Here you can add custom analysis.
    if "min_distance" not in properties:
        database.add_properties(
            key=name,
            property_dict={
                "min_distance": (
                    cgx.analysis.GeomMeasure().calculate_min_distance(
                        database.get_molecule(key=name)
                    )["min_distance"]
                ),
            },
        )


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    width_height: tuple[float, float] = (7, 10),
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=width_height)
    energies: dict[tuple[tuple[str, ...], str], list[tuple]] = {}

    xs = []
    ys = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "lowest_e_of_mash" not in entry.properties:
            continue
        if "multiplier" not in entry.properties:
            continue
        multi = str(entry.properties["multiplier"])
        if multi not in xs:
            xs.append(multi)

        config_name = tuple(entry.properties["config_name"].split("_"))  # type:ignore[union-attr]
        if config_name not in ys:
            ys.append(config_name)

        tidx = int(entry.properties["topology_idx"])  # type:ignore[arg-type]
        bidx = int(entry.properties["bb_config_idx"])  # type:ignore[arg-type]
        midx = int(entry.properties["mash_idx"])  # type:ignore[arg-type]
        energy = float(entry.properties["energy_per_bb"])  # type:ignore[arg-type]

        if (config_name, multi) not in energies:
            energies[(config_name, multi)] = []

        if int(entry.properties["num_components"]) > 1:  # type:ignore[arg-type]
            continue
        energies[(config_name, multi)].append(
            (round(energy, 4), tidx, bidx, midx)  # type:ignore[arg-type]
        )

    # create the new map
    cmap = plt.cm.Blues_r  # type:ignore[attr-defined]
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", cmaplist, cmap.N
    )

    # define the bins and normalize
    bounds = np.arange(0, 5, 0.5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    for (pair, multi), evalues in energies.items():
        sorted_energies = sorted(evalues, key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = ys.index(pair)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap=cmap,
            norm=norm,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(ys))))
    ax.set_yticklabels(["_".join(i) for i in ys])

    ax.axhline(4.5, c="k", alpha=0.5)
    ax.axhline(8.5, c="k", alpha=0.5)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])  # type:ignore[call-overload]
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("energy", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
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
        / "recipe_5_output"
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
    ligand_dir = wd / "ligands"
    cgx.utilities.check_directory(ligand_dir)

    # Define a database, and a prefix for naming structure, forcefield and
    # output files.
    database_path = data_dir / "test.db"

    # Define a definer dictionary.
    # These are constants, while different systems can override these
    # parameters.
    cg_scale = 2
    constant_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "aca": ("angle", 180, 1e2),
        "ede": ("angle", 180, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbe": ("angle", 180, 1e2),
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "f": ("nb", 10.0, 1.0),
    }

    # Define beads.
    bead_library = cgx.molecular.BeadLibrary.from_bead_types(
        # Type and coordination.
        {"m": 4, "a": 2, "b": 2, "c": 2, "d": 2, "e": 2, "f": 2}
    )

    # Define your forcefield alterations as building blocks.
    building_block_library = {
        "lin": {
            "precursor": cgx.molecular.TwoC1Arm(
                bead=bead_library.get_from_type("c"),
                abead1=bead_library.get_from_type("a"),
            ),
            "mod_definer_dict": {
                "ba": ("bond", 2.8 / cg_scale, 1e5),
                "ac": ("bond", 1.5 / 2 / cg_scale, 1e5),
                "bac": ("angle", 180, 1e2),
            },
        },
        "mxy": {
            "precursor": cgx.molecular.TwoC1Arm(
                bead=bead_library.get_from_type("d"),
                abead1=bead_library.get_from_type("e"),
            ),
            "mod_definer_dict": {
                "be": ("bond", 7.6 / cg_scale, 1e5),
                "ed": ("bond", 5.0 / 2 / cg_scale, 1e5),
                "bed": ("angle", 90, 1e2),
            },
        },
        "tetra": {
            "precursor": cgx.molecular.FourC1Arm(
                bead=bead_library.get_from_type("m"),
                abead1=bead_library.get_from_type("b"),
            ),
            "mod_definer_dict": {
                "mb": ("bond", 2.0 / cg_scale, 1e5),
                "bmb": ("pyramid", 90, 1e2),
            },
        },
        "corner": {
            "precursor": cgx.molecular.TwoC0Arm(
                bead=bead_library.get_from_type("f"),
            ),
            "mod_definer_dict": {
                "bf": ("bond", 2.0 / cg_scale, 1e5),
                "bfb": ("angle", 90, 1e2),
                "fbm": ("angle", 90, 1e2),
            },
        },
    }

    # Define systems to predict the structure of.
    systems = {
        "mix1_2-2-1-1": {
            "stoichiometry_map": {"tetra": 2, "corner": 2, "lin": 1, "mxy": 1},
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "mix1_2-2-2-0": {
            "stoichiometry_map": {"tetra": 2, "corner": 2, "lin": 2},
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
        "mix1_2-2-0-2": {
            "stoichiometry_map": {"tetra": 2, "corner": 2, "mxy": 2},
            "multipliers": (1,),
            "vdw_cutoff": 2,
        },
    }

    if args.run:
        for system_name, syst_d in systems.items():
            logger.info("doing system: %s", system_name)
            # Merge constant dict with modifications from different systems.
            merged_definer_dicts = (
                cgx.systems_optimisation.merge_definer_dicts(
                    original_definer_dict=constant_definer_dict,
                    new_definer_dicts=[
                        building_block_library[i]["mod_definer_dict"]  # type:ignore[misc]
                        for i in syst_d["stoichiometry_map"]  # type:ignore[attr-defined]
                    ],
                )
            )

            forcefield = cgx.systems_optimisation.get_forcefield_from_dict(
                identifier=f"{system_name}ff",
                prefix=f"{system_name}ff",
                vdw_bond_cutoff=syst_d["vdw_cutoff"],  # type:ignore[arg-type]
                present_beads=bead_library.get_present_beads(),
                definer_dict=merged_definer_dicts,
            )

            # Build all the building blocks and pre optimise their structures.
            bb_map = {}
            for prec_name in syst_d["stoichiometry_map"]:  # type:ignore[attr-defined]
                prec: cgx.molecular.Precursor = building_block_library[  # type:ignore[assignment]
                    prec_name
                ]["precursor"]
                if prec_name == "corner":
                    bb: stk.BuildingBlock = prec.get_building_block()
                else:
                    bb = cgx.utilities.optimise_ligand(  # type:ignore[assignment]
                        molecule=prec.get_building_block(),
                        name=f"{system_name}_{prec.get_name()}",
                        output_dir=calc_dir,
                        forcefield=forcefield,
                        platform=None,
                    ).clone()
                    bb.write(
                        str(
                            ligand_dir
                            / f"{system_name}_{prec.get_name()}_optl.mol"
                        )
                    )
                bb_map[prec_name] = bb

            for multiplier in syst_d["multipliers"]:  # type:ignore[attr-defined]
                logger.info(
                    "doing system: %s, multi: %s", system_name, multiplier
                )

                # Define a connectivity based on a multiplier.
                iterator = cgx.scram.TopologyIterator(
                    building_block_counts={
                        bb_map[name]: stoich * multiplier
                        for name, stoich in syst_d["stoichiometry_map"].items()  # type:ignore[attr-defined]
                    },
                )
                logger.info(
                    "graph iteration has %s graphs", iterator.count_graphs()
                )

                possible_bbdicts = iterator.get_configurations()
                logger.info(
                    "building block iteration has %s options",
                    len(possible_bbdicts),
                )

                logger.info(
                    "iterating over %s graphs and bb configurations...",
                    iterator.count_graphs() * len(possible_bbdicts),
                )
                run_topology_codes: list[agx.ConfiguredCode] = []
                for bb_config, topology_code in it.product(
                    possible_bbdicts,
                    iterator.yield_graphs(),
                ):
                    configured = agx.ConfiguredCode(topology_code, bb_config)
                    if not agx.utilities.is_configured_code_isomoprhic(
                        test_code=configured,
                        run_topology_codes=run_topology_codes,
                    ):
                        continue

                    run_topology_codes.append(configured)

                    # Here we apply a multi-initial state, multi-step geometry
                    # optimisation algorithm.
                    config_name = (
                        f"{system_name}_{multiplier}_"
                        f"{topology_code.idx}_b{bb_config.idx}"
                    )
                    # Each conformer is stored here.
                    conformer_db_path = calc_dir / f"{config_name}.db"
                    optimisation_workflow(
                        config_name=config_name,
                        conformer_db_path=conformer_db_path,
                        topology_code=topology_code,
                        iterator=iterator,
                        bb_config=bb_config,
                        calculation_dir=calc_dir,
                        forcefield=forcefield,
                    )

                    conformer_db = cgx.utilities.AtomliteDatabase(
                        conformer_db_path
                    )
                    min_energy_structure = None
                    min_energy = float("inf")
                    min_energy_key = None
                    for entry in conformer_db.get_entries():
                        if entry.properties["energy_per_bb"] < min_energy:  # type:ignore[operator]
                            min_energy = entry.properties["energy_per_bb"]  # type:ignore[assignment]
                            min_energy_structure = conformer_db.get_molecule(
                                key=entry.key
                            )
                            min_energy_key = entry.key

                    # To file.
                    min_energy_structure.write(  # type:ignore[union-attr]
                        str(struct_output / f"{config_name}_optc.mol")
                    )

                    # To database.
                    cgx.utilities.AtomliteDatabase(database_path).add_molecule(
                        molecule=min_energy_structure,  # type:ignore[arg-type]
                        key=config_name,
                    )
                    properties = {
                        "multiplier": multiplier,
                        "topology_idx": topology_code.idx,
                    }
                    cgx.utilities.AtomliteDatabase(
                        database_path
                    ).add_properties(key=config_name, property_dict=properties)

                    analyse_cage(
                        database_path=database_path,
                        name=config_name,
                        conformer_db_path=conformer_db_path,
                        min_energy_key=min_energy_key,  # type:ignore[arg-type]
                    )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="recipe_5_test.png",
    )


if __name__ == "__main__":
    main()
