"""Copiable code from Recipe #3."""  # noqa: INP001

import argparse
import logging
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import cgexplore as cgx

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    args = _parse_args()
    wd = (
        pathlib.Path(__file__).resolve().parent
        / ".."
        / ".."
        / "recipes"
        / "recipe_3_output"
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

    database_path = data_dir / "test.db"
    database = cgx.utilities.AtomliteDatabase(database_path)

    # Define beads.
    bead_library = cgx.molecular.BeadLibrary.from_bead_types(
        # Type and coordination.
        {"m": 4, "a": 2, "e": 2, "b": 2, "c": 2, "d": 2}
    )

    # Define a definer dictionary.
    # These are constants, while different systems can override these
    # parameters.
    cg_scale = 2
    constant_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbe": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "ede": ("angle", 180, 1e2),
        # Torsions.
        "bacab": ("tors", "0134", 180, 50, 1),
        "bedeb": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
    }

    # Define your forcefield alterations as building blocks.
    building_block_library = {
        "deg90": {
            "precursor": cgx.molecular.TwoC1Arm(
                bead=bead_library.get_from_type("d"),
                abead1=bead_library.get_from_type("e"),
            ),
            "mod_definer_dict": {
                "be": ("bond", 1.5 / cg_scale, 1e5),
                "de": ("bond", 5.9 / 2 / cg_scale, 1e5),
                "deb": ("angle", 135, 1e2),
            },
        },
        "l1a": {
            "precursor": cgx.molecular.TwoC1Arm(
                bead=bead_library.get_from_type("c"),
                abead1=bead_library.get_from_type("a"),
            ),
            "mod_definer_dict": {
                "ba": ("bond", 1.5 / cg_scale, 1e5),
                "ac": ("bond", 9.5 / 2 / cg_scale, 1e5),
                "bac": ("angle", 120, 1e2),
            },
        },
        "tetra": {
            "precursor": cgx.molecular.FourC1Arm(
                bead=bead_library.get_from_type("m"),
                abead1=bead_library.get_from_type("b"),
            ),
            "mod_definer_dict": {},
        },
    }

    # Define systems to predict the structure of.
    # Only focussing on m=9.
    multiplier = 1
    systems = {
        "l1a_90_9-9-9": {
            "stoichiometry_map": {"tetra": 9, "l1a": 9, "deg90": 9},
            "vdw_cutoff": 2,
        },
        "l1a_90_6-12-9": {
            "stoichiometry_map": {"tetra": 9, "l1a": 6, "deg90": 12},
            "vdw_cutoff": 2,
        },
        "l1a_90_12-6-9": {
            "stoichiometry_map": {"tetra": 9, "l1a": 12, "deg90": 6},
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

            # Build all the building blocks and pre optimise their structures.
            bb_map = {}
            for prec_name in syst_d["stoichiometry_map"]:
                prec = building_block_library[prec_name]["precursor"]
                bb = cgx.utilities.optimise_ligand(
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

            # Define the chromosome generator, holding all the changeable genes.
            chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
                prefix=system_name,
                present_beads=bead_library.get_present_beads(),
                vdw_bond_cutoff=syst_d["vdw_cutoff"],
            )

            # Automate the graph type naming.
            graph_type = cgx.scram.generate_graph_type(
                stoichiometry_map=syst_d["stoichiometry_map"],
                multiplier=multiplier,
                bb_library=bb_map,
            )
            print(graph_type)
            # Add graphs.
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bb_map[name]: stoich * multiplier
                    for name, stoich in syst_d["stoichiometry_map"].items()
                },
                graph_type=graph_type,
                graph_set="rxx",
            )
            all_topology_codes = tuple(enumerate(iterator.yield_graphs()))
            topology_codes = []
            for tidx, tc in all_topology_codes:
                if tc.contains_parallels():
                    continue
                topology_codes.append((tidx, tc))

            logger.info(
                "graph iteration has %s graphs (from %s)",
                len(topology_codes),
                len(all_topology_codes),
            )
            continue
            chromo_it.add_gene(
                iteration=topology_codes,
                gene_type="topology",
            )

            # Add building block configurations.
            possible_bbdicts = cgx.scram.get_custom_bb_configurations(
                iterator=iterator
            )
            logger.info(
                "building block iteration has %s options",
                len(possible_bbdicts),
            )
            chromo_it.add_gene(
                iteration=possible_bbdicts,
                gene_type="vertex_alignment",
            )
            raise SystemExit

            # Define fitness calculator.
            fitness_calculator = cgx.systems_optimisation.FitnessCalculator(
                fitness_function=fitness_function,
                chromosome_generator=chromo_it,
                structure_output=struct_output,
                calculation_output=calc_dir,
                database_path=database_path,
                options={},
            )

            # Define structure calculator.
            structure_calculator = (
                cgx.systems_optimisation.StructureCalculator(
                    structure_function=structure_function,
                    structure_output=struct_output,
                    calculation_output=calc_dir,
                    database_path=database_path,
                    options={},
                )
            )

            seeds = [4]
            num_generations = 10
            selection_size = 5
            num_processes = 1
            num_to_operate = 2
            for seed in seeds:
                generator = np.random.default_rng(seed)

                initial_population = chromo_it.select_random_population(
                    generator,
                    size=selection_size,
                )

                # Yield this.
                generations = []
                generation = cgx.systems_optimisation.Generation(
                    chromosomes=initial_population,
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    num_processes=num_processes,
                )

                generation.run_structures()
                _ = generation.calculate_fitness_values()
                generations.append(generation)

                for generation_id in range(1, num_generations + 1):
                    logger.info(
                        "doing generation %s of seed %s", generation_id, seed
                    )
                    logger.info(
                        "initial size is %s.", generation.get_generation_size()
                    )
                    logger.info("doing mutations.")
                    merged_chromosomes = []
                    merged_chromosomes.extend(
                        chromo_it.mutate_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            gene_range=chromo_it.get_term_ids(),
                            selection="random",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )
                    merged_chromosomes.extend(
                        chromo_it.mutate_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            gene_range=chromo_it.get_topo_ids(),
                            selection="random",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )
                    merged_chromosomes.extend(
                        chromo_it.mutate_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            gene_range=chromo_it.get_prec_ids(),
                            selection="random",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )
                    merged_chromosomes.extend(
                        chromo_it.mutate_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            gene_range=chromo_it.get_term_ids(),
                            selection="roulette",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )
                    merged_chromosomes.extend(
                        chromo_it.mutate_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            gene_range=chromo_it.get_topo_ids(),
                            selection="roulette",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )
                    merged_chromosomes.extend(
                        chromo_it.mutate_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            gene_range=chromo_it.get_prec_ids(),
                            selection="roulette",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )

                    merged_chromosomes.extend(
                        chromo_it.crossover_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            selection="random",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )

                    merged_chromosomes.extend(
                        chromo_it.crossover_population(
                            chromosomes={
                                f"{chromosome.prefix}"
                                f"_{chromosome.get_separated_string()}": chromosome
                                for chromosome in generation.chromosomes
                            },
                            generator=generator,
                            selection="roulette",
                            num_to_select=num_to_operate,
                            database=database,
                        )
                    )

                    # Add the best 5 to the new generation.
                    merged_chromosomes.extend(
                        generation.select_best(selection_size=5)
                    )

                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=chromo_it.dedupe_population(
                            merged_chromosomes
                        ),
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=num_processes,
                    )
                    logger.info(
                        "new size is %s.", generation.get_generation_size()
                    )

                    # Build, optimise and analyse each structure.
                    generation.run_structures()
                    _ = generation.calculate_fitness_values()

                    # Add final state to generations.
                    generations.append(generation)

                    # Select the best of the generation for the next generation.
                    best = generation.select_best(
                        selection_size=selection_size
                    )
                    generation = cgx.systems_optimisation.Generation(
                        chromosomes=chromo_it.dedupe_population(best),
                        fitness_calculator=fitness_calculator,
                        structure_calculator=structure_calculator,
                        num_processes=num_processes,
                    )
                    logger.info(
                        "final size is %s.", generation.get_generation_size()
                    )

                    progress_plot(
                        generations=generations,
                        output=figure_dir / f"fitness_progress_{seed}.png",
                        num_generations=num_generations,
                    )

                    # Output best structures as images.
                    best_chromosome = generation.select_best(selection_size=1)[
                        0
                    ]
                    best_name = (
                        f"{best_chromosome.prefix}_"
                        f"{best_chromosome.get_separated_string()}"
                    )

                logger.info("top scorer is %s (seed: %s)", best_name, seed)

            # Report.
            found = set()
            for generation in generations:
                for chromo in generation.chromosomes:
                    found.add(chromo.name)
            logger.info(
                "%s chromosomes found in EA (of %s)",
                len(found),
                chromo_it.get_num_chromosomes(),
            )

            fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
            ax, ax1 = axs
            xys = defaultdict(list)
            plotted = 0
            for entry in database.get_entries():
                tstr = entry.properties["topology"]

                if "fitness" not in entry.properties:
                    continue
                # Caution here, this is the fitness at point of calculation of each
                # entry. Not after the reevaluation of the self-sorting.
                fitness = entry.properties["fitness"]

                ax.scatter(
                    entry.properties["opt_pore_data"]["min_distance"],
                    entry.properties["energy_per_bb"],
                    c=fitness,
                    edgecolor="none",
                    s=100,
                    marker="o",
                    alpha=1.0,
                    vmin=0,
                    vmax=40,
                    cmap="Blues",
                )
                xys[
                    (
                        entry.properties["forcefield_dict"]["v_dict"]["b_c_o"],
                        entry.properties["forcefield_dict"]["v_dict"]["o_a_o"],
                    )
                ].append(
                    (
                        entry.properties["topology"],
                        entry.properties["energy_per_bb"],
                        fitness,
                    )
                )
                plotted += 1

            for x, y in xys:
                fitness_threshold = 10
                stable = [
                    i[0]
                    for i in xys[(x, y)]
                    if i[1] < isomer_energy() and i[2] > fitness_threshold
                ]

                if len(stable) == 0:
                    cmaps = ["white"]
                else:
                    cmaps = sorted([colours()[i] for i in stable])

                if len(cmaps) > 8:  # noqa: PLR2004
                    cmaps = ["k"]
                cgx.utilities.draw_pie(
                    colours=cmaps,
                    xpos=x,
                    ypos=y,
                    size=400,
                    ax=ax1,
                )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_xlabel("min centroid distance [angstrom]", fontsize=16)
            ax.set_ylabel(r"$E_{\mathrm{b}}$ [kjmol-1]", fontsize=16)
            ax.set_yscale("log")
            ax.set_title(
                f"plotted: {plotted}, found: {len(found)}, "
                f"possible: {chromo_it.get_num_chromosomes()}",
                fontsize=16,
            )
            ax1.tick_params(axis="both", which="major", labelsize=16)
            ax1.set_xlabel("b_c_o [deg]", fontsize=16)
            ax1.set_ylabel("o_a_o [deg]", fontsize=16)
            ax1.set_title(
                f"E: {isomer_energy()}, F: {fitness_threshold}", fontsize=16
            )

            for tstr in colours():
                ax1.scatter(
                    None,
                    None,
                    c=colours()[tstr],
                    edgecolor="none",
                    s=60,
                    marker="o",
                    alpha=1.0,
                    label=tstr,
                )

            ax1.legend(fontsize=16, loc="lower left")
            fig.tight_layout()
            fig.savefig(
                figure_dir / "space_explored.png",
                dpi=360,
                bbox_inches="tight",
            )
            plt.close()

            # Write chemiscope.
            properties = defaultdict(list)
            structures = []
            for entry in database.get_entries():
                tstr = entry.properties["topology"]

                if "fitness" not in entry.properties:
                    continue

                structures.append(database.get_molecule(key=entry.key))
                properties["key"].append(entry.key)
                properties["E_b / kjmol-1"].append(
                    entry.properties["energy_per_bb"]
                )
                # Caution here, this is the fitness at point of calculation of each
                # entry. Not after the reevaluation of the self-sorting.
                properties["fitness"].append(entry.properties["fitness"])
                properties["min_distance"].append(
                    entry.properties["opt_pore_data"]["min_distance"]
                )
                properties["num_bbs"].append(int(entry.properties["num_bbs"]))
                properties["b_c_o / deg"].append(
                    entry.properties["forcefield_dict"]["v_dict"]["b_c_o"]
                )
                properties["o_a_o / deg"].append(
                    entry.properties["forcefield_dict"]["v_dict"]["o_a_o"]
                )

            logger.info("saving %s entries", len(structures))
            cgx.utilities.write_chemiscope_json(
                json_file=data_dir / "space_explored.json.gz",
                structures=structures,
                properties=properties,
                bonds_as_shapes=True,
                meta_dict={
                    "name": "Recipe 3 structures.",
                    "description": ("Minimal models from recipe 3."),
                    "authors": ["Andrew Tarzia"],
                    "references": [],
                },
                x_axis_dict={"property": "b_c_o / deg"},
                y_axis_dict={"property": "o_a_o / deg"},
                z_axis_dict={"property": "num_bbs"},
                color_dict={"property": "E_b / kjmol-1", "min": 0, "max": 1.0},
                bond_hex_colour="#919294",
            )
    raise SystemExit("clean up")


if __name__ == "__main__":
    main()
