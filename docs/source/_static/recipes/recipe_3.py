"""Copiable code from Recipe #3."""  # noqa: INP001

import argparse
import logging
import pathlib
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stko
from openmm import OpenMMException

import cgexplore as cgx

if TYPE_CHECKING:
    import stk

logger = logging.getLogger(__name__)
seed_cs = {
    4: "tab:blue",
    12689: "tab:orange",
    18: "tab:green",
    999: "tab:red",
    142: "tab:purple",
    6582: "tab:cyan",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def progress_plot(
    seeded_generations: dict[int, list],
    output: pathlib.Path,
) -> None:
    """Draw optimisation progress."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (seed, generations) in enumerate(seeded_generations.items()):
        fitnesses = [
            generation.calculate_fitness_values() for generation in generations
        ]

        ax.plot(
            [max(i) for i in fitnesses],
            label=f"{seed}",
            lw=2,
            c=seed_cs[seed],
            marker="o",
            markersize=8,
            markeredgecolor="w",
        )
        ax.plot(
            [np.mean(i) for i in fitnesses],
            lw=2,
            ls="--",
            c=seed_cs[seed],
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(None, 10)
    ax.set_xlabel("generation", fontsize=16)
    ax.legend(ncols=2, fontsize=16)

    fig.tight_layout()
    fig.savefig(output, dpi=360, bbox_inches="tight")
    plt.close("all")


def fitness_function(  # noqa: PLR0913
    chromosome: cgx.systems_optimisation.Chromosome,
    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator,  # noqa: ARG001
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,  # noqa: ARG001
    structure_output: pathlib.Path,  # noqa: ARG001
    options: dict,
) -> float:
    """Calculate fitness."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    topology_idx, _ = chromosome.get_topology_information()
    building_block_config = chromosome.get_vertex_alignments()[0]
    name = f"{chromosome.prefix}_{topology_idx}_b{building_block_config.idx}"

    entry = database.get_entry(name)

    if not entry.properties["opt_passed"]:
        energy = 100.0

    elif entry.properties["is_duplicate"]:
        energy = float(
            database.get_entry(entry.properties["duplicate_of"]).properties[  # type:ignore[arg-type]
                "energy_per_bb"
            ]
        )

    else:
        energy = float(entry.properties["energy_per_bb"])  # type:ignore[arg-type]

    fitness = np.exp(-energy * options["beta"])
    database.add_properties(key=name, property_dict={"fitness": fitness})

    return fitness


def structure_function(  # noqa: C901, PLR0915
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,
) -> None:
    """Geometry optimisation."""
    database = cgx.utilities.AtomliteDatabase(database_path)

    topology_idx, topology_code = chromosome.get_topology_information()
    building_block_config = chromosome.get_vertex_alignments()[0]

    base_name = (
        f"{chromosome.prefix}_{topology_idx}_b{building_block_config.idx}"
    )
    l1, l2, stoichstring = chromosome.prefix.split("_")
    multiplier = stoichstring.split("-")[2]

    print(database.has_molecule(base_name), base_name)
    if database.has_molecule(base_name):
        print("he")
        return

    # Check if this has been run before.
    known_entry = None
    for entry in database.get_entries():
        # Only do base entries.
        if "is_base" not in entry.properties:
            continue

        try:
            entry_tc = options["topology_codes"][
                entry.properties["topology_idx"]
            ]
            entry_bb_config = options["bb_configs"][
                entry.properties["bb_config_idx"]
            ]
        except (KeyError, IndexError):
            continue

        known_stoichstring = entry.properties["stoichstring"]
        known_pair = entry.properties["pair"]
        if f"{l1}_{l2}" != known_pair:
            continue
        if stoichstring != known_stoichstring:
            continue

        # Testing bb-config aware graph check.
        if not cgx.scram.passes_graph_bb_iso(
            topology_code=topology_code,
            bb_config=building_block_config,
            run_topology_codes=[(entry_tc[1], entry_bb_config)],
        ):
            known_entry = entry
            break

    # Try to avoid recalculation if possible.
    if (
        known_entry is not None
        and known_entry.properties["base_name"] != base_name
    ):
        database.add_molecule(
            key=base_name,
            molecule=database.get_molecule(known_entry.key),
        )
        database.add_properties(
            key=base_name,
            property_dict={
                "is_duplicate": True,
                "duplicate_of": known_entry.key,
                "opt_passed": True,
            },
        )

        try:
            nd_ = int(known_entry.properties["num_duplicates"]) + 1  # type:ignore[arg-type]
        except KeyError:
            nd_ = 1
        database.add_properties(
            key=known_entry.key,
            property_dict={"num_duplicates": nd_},
        )

        logger.info("%s is duplicate", base_name)
        return

    # Actually do the calculation, now, just because we have too.
    constructed_molecule = cgx.scram.get_regraphed_molecule(
        graph_type="kamada",
        scale=10,
        topology_code=topology_code,
        iterator=options["iterator"],
        bb_config=building_block_config,
    )

    constructed_molecule.write(calculation_output / f"{base_name}_unopt.mol")
    opt_file = structure_output / f"{base_name}_optc.mol"

    # Optimise and save.
    logger.info("building %s", base_name)

    try:
        conformer = cgx.utilities.run_optimisation(
            assigned_system=options["forcefield"].assign_terms(
                molecule=constructed_molecule,
                name=base_name,
                output_dir=calculation_output,
            ),
            name=base_name,
            file_suffix="opt1",
            output_dir=calculation_output,
            platform=None,
        )
        opt_passed = True

    except OpenMMException:
        logger.info("failed optimisation of %s", base_name)
        opt_passed = False

    if opt_passed:
        properties = {
            "base_name": base_name,
            "energy_per_bb": cgx.utilities.get_energy_per_bb(
                energy_decomposition=conformer.energy_decomposition,
                number_building_blocks=(
                    options["iterator"].get_num_building_blocks()
                ),
            ),
            "num_components": len(
                stko.Network.init_from_molecule(
                    conformer.molecule
                ).get_connected_components()
            ),
            "forcefield_dict": (
                options["forcefield"].get_forcefield_dictionary()
            ),
            "l1": l1,
            "l2": l2,
            "pair": f"{l1}_{l2}",
            "num_bbs": (options["iterator"].get_num_building_blocks()),
            "stoichstring": stoichstring,
            "multiplier": multiplier,
            "topology_idx": topology_idx,
            "topology_code_vmap": tuple(
                (int(i[0]), int(i[1])) for i in topology_code.vertex_map
            ),
            "bb_config_idx": building_block_config.idx,
            # Add here, if it gets here, then it is not duplicate.
            "is_duplicate": False,
            "num_duplicates": 0,
        }
        database.add_molecule(key=base_name, molecule=conformer.molecule)
        conformer.molecule.write(opt_file)
    else:
        database.add_molecule(key=base_name, molecule=constructed_molecule)

    # Write base name to database.
    database.add_properties(key=base_name, property_dict=properties)
    database.add_properties(
        key=base_name,
        property_dict={"is_base": True, "opt_passed": opt_passed},
    )


def run_genetic_algorithm(  # noqa: PLR0913
    seed: int,
    chromo_it: cgx.systems_optimisation.ChromosomeGenerator,
    fitness_calculator: cgx.systems_optimisation.FitnessCalculator,
    structure_calculator: cgx.systems_optimisation.StructureCalculator,
    scan_config: dict,
    elite_population: cgx.systems_optimisation.Generation | None,
    database: cgx.utilities.AtomliteDatabase,
    neighbour_opt: bool,
) -> list[cgx.systems_optimisation.Generation]:
    """A helper function for running each GA."""
    generator = np.random.default_rng(seed)

    if elite_population is None:
        initial_population = chromo_it.select_random_population(
            generator,
            size=scan_config["selection_size"],
        )
    else:
        initial_population = elite_population.select_elite(
            proportion_threshold=0.25
        )

        logger.info(
            "selected elite with f>%s",
            round(
                elite_population.calculate_elite_fitness(
                    proportion_threshold=0.25
                ),
                5,
            ),
        )

    # Yield this.
    generations = []
    generation = cgx.systems_optimisation.Generation(
        chromosomes=initial_population,
        fitness_calculator=fitness_calculator,
        structure_calculator=structure_calculator,
        num_processes=scan_config["num_processes"],
    )

    generation.run_structures()
    _ = generation.calculate_fitness_values()
    generations.append(generation)

    for generation_id in range(1, scan_config["num_generations"] + 1):
        logger.info("doing generation %s of seed %s", generation_id, seed)
        logger.info("initial size is %s.", generation.get_generation_size())
        logger.info("doing mutations.")
        if neighbour_opt:
            merged_chromosomes: list[cgx.systems_optimisation.Chromosome] = []
            merged_chromosomes.extend(
                chromo_it.get_population_neighbours(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    selection="all",
                    gene_range=chromo_it.get_va_ids(),
                )
            )
            merged_chromosomes.extend(generation.select_all())
        else:
            merged_chromosomes = []
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    generator=generator,
                    gene_range=chromo_it.get_va_ids(),
                    selection="random",
                    num_to_select=scan_config["mutations"],
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    generator=generator,
                    gene_range=chromo_it.get_topo_ids(),
                    selection="random",
                    num_to_select=scan_config["mutations"],
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    generator=generator,
                    gene_range=chromo_it.get_va_ids(),
                    selection="roulette",
                    num_to_select=scan_config["mutations"],
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    generator=generator,
                    gene_range=chromo_it.get_topo_ids(),
                    selection="roulette",
                    num_to_select=scan_config["mutations"],
                    database=database,
                )
            )

            merged_chromosomes.extend(
                chromo_it.crossover_population(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    generator=generator,
                    selection="random",
                    num_to_select=scan_config["mutations"],
                    database=database,
                )
            )

            merged_chromosomes.extend(
                chromo_it.crossover_population(
                    chromosomes={
                        (
                            f"{chromosome.prefix}"
                            f"_{chromosome.get_topology_information()[0]}"
                            f"_b{chromosome.get_vertex_alignments()[0].idx}"
                        ): chromosome
                        for chromosome in generation.chromosomes
                    },
                    generator=generator,
                    selection="roulette",
                    num_to_select=scan_config["mutations"],
                    database=database,
                )
            )

            # Add the best 5 to the new generation.
            merged_chromosomes.extend(generation.select_best(selection_size=5))

        generation = cgx.systems_optimisation.Generation(
            chromosomes=chromo_it.dedupe_population(merged_chromosomes),
            fitness_calculator=fitness_calculator,
            structure_calculator=structure_calculator,
            num_processes=scan_config["num_processes"],
        )
        logger.info("new size is %s.", generation.get_generation_size())

        # Build, optimise and analyse each structure.
        generation.run_structures()
        _ = generation.calculate_fitness_values()

        # Add final state to generations.
        generations.append(generation)
        # Select the best of the generation for the next
        # generation.
        best = generation.select_best(
            selection_size=scan_config["selection_size"]
        )
        generation = cgx.systems_optimisation.Generation(
            chromosomes=chromo_it.dedupe_population(best),
            fitness_calculator=fitness_calculator,
            structure_calculator=structure_calculator,
            num_processes=scan_config["num_processes"],
        )
        logger.info("final size is %s.", generation.get_generation_size())

        # Output best structures as images.
        best_chromosome = generation.select_best(selection_size=1)[0]
        best_name = (
            f"{best_chromosome.prefix}_"
            f"{best_chromosome.get_topology_information()[0]}_"
            f"b{best_chromosome.get_vertex_alignments()[0].idx}"
        )

    logger.info("top scorer is %s (seed: %s)", best_name, seed)
    return generations


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
                "bac": ("angle", 150, 1e2),
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
        "l1a_90_6-12-9": {
            "stoichiometry_map": {"tetra": 9, "l1a": 6, "deg90": 12},
            "vdw_cutoff": 2,
        },
        "l1a_90_12-6-9": {
            "stoichiometry_map": {"tetra": 9, "l1a": 12, "deg90": 6},
            "vdw_cutoff": 2,
        },
        "l1a_90_9-9-9": {
            "stoichiometry_map": {"tetra": 9, "l1a": 9, "deg90": 9},
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
                opt_bb_file = (
                    ligand_dir / f"{system_name}_{prec.get_name()}_optl.mol"
                )
                if opt_bb_file.exists():
                    bb_map[prec_name] = (
                        prec.get_building_block().with_structure_from_file(
                            opt_bb_file
                        )
                    )
                else:
                    bb: stk.BuildingBlock = cgx.utilities.optimise_ligand(  # type:ignore[assignment]
                        molecule=prec.get_building_block(),
                        name=f"{system_name}_{prec.get_name()}",
                        output_dir=calc_dir,
                        forcefield=forcefield,
                        platform=None,
                    ).clone()
                    bb.write(str(opt_bb_file))
                    bb_map[prec_name] = bb

            # Define the chromosome generator, holding all the changeable
            # genes.
            chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
                prefix=system_name,
                present_beads=bead_library.get_present_beads(),
                vdw_bond_cutoff=syst_d["vdw_cutoff"],  # type:ignore[arg-type]
            )

            # Automate the graph type naming.
            graph_type = cgx.scram.generate_graph_type(
                stoichiometry_map=syst_d["stoichiometry_map"],  # type:ignore[arg-type]
                multiplier=multiplier,
                bb_library=bb_map,
            )
            # Add graphs.
            iterator = cgx.scram.TopologyIterator(
                building_block_counts={
                    bb_map[name]: stoich * multiplier
                    for name, stoich in syst_d["stoichiometry_map"].items()  # type:ignore[attr-defined]
                },
                graph_type=graph_type,
                graph_set="rxx",
            )
            all_topology_codes = tuple(enumerate(iterator.yield_graphs()))
            topology_codes = []
            for tidx, tc in all_topology_codes:
                if tc.contains_parallels():
                    continue
                # Also exlcude double walls to lower the search space.
                if tc.contains_doubles():
                    continue
                topology_codes.append((tidx, tc))

            logger.info(
                "graph iteration has %s graphs (from %s)",
                len(topology_codes),
                len(all_topology_codes),
            )
            chromo_it.add_gene(iteration=topology_codes, gene_type="topology")

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

            # Define fitness calculator.
            fitness_calculator = cgx.systems_optimisation.FitnessCalculator(
                fitness_function=fitness_function,
                chromosome_generator=chromo_it,
                structure_output=struct_output,
                calculation_output=calc_dir,
                database_path=database_path,
                options={"beta": 5},
            )

            # Define structure calculator.
            structure_calculator = (
                cgx.systems_optimisation.StructureCalculator(
                    structure_function=structure_function,
                    structure_output=struct_output,
                    calculation_output=calc_dir,
                    database_path=database_path,
                    options={
                        "topology_codes": list(all_topology_codes),
                        "bb_configs": possible_bbdicts,
                        "iterator": iterator,
                        "forcefield": forcefield,
                    },
                )
            )

            # Short runs.
            seeded_generations: dict[
                int, list[cgx.systems_optimisation.Generation]
            ] = {}
            # Very short runs!
            scan_config = {
                "seeds": [4, 12689],
                "mutations": 2,
                "num_generations": 5,
                "selection_size": 5,
                "num_processes": 1,
                "long_seeds": [142],
                "neighbour_seeds": [6582],
            }
            for seed in scan_config["seeds"]:  # type:ignore[attr-defined]
                seeded_generations[seed] = run_genetic_algorithm(
                    seed=seed,
                    chromo_it=chromo_it,
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    scan_config=scan_config,
                    database=database,
                    elite_population=None,
                    neighbour_opt=False,
                )
                progress_plot(
                    seeded_generations=seeded_generations,
                    output=figure_dir / f"fp_{system_name}.png",
                )

            # Run longer GA from elites.
            chromosomes: list[cgx.systems_optimisation.Chromosome] = []
            for generations in seeded_generations.values():
                for generation in generations:
                    chromosomes.extend(generation.chromosomes)
            elite_population = cgx.systems_optimisation.Generation(
                chromosomes=chromo_it.dedupe_population(chromosomes),
                fitness_calculator=fitness_calculator,
                structure_calculator=structure_calculator,
                num_processes=scan_config["num_processes"],  # type:ignore[arg-type]
            )
            for seed in scan_config["long_seeds"]:  # type:ignore[attr-defined]
                temp_scan_config = scan_config.copy()
                temp_scan_config.update(
                    {"num_generations": scan_config["num_generations"] * 2}  # type:ignore[operator]
                )
                seeded_generations[seed] = run_genetic_algorithm(
                    seed=seed,
                    chromo_it=chromo_it,
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    scan_config=temp_scan_config,
                    database=database,
                    elite_population=elite_population,
                    neighbour_opt=False,
                )
                progress_plot(
                    seeded_generations=seeded_generations,
                    output=figure_dir / f"fp_{system_name}.png",
                )

            # And then again, but only over neighbours.
            chromosomes = []
            for generations in seeded_generations.values():
                for generation in generations:
                    chromosomes.extend(generation.chromosomes)
            elite_population = cgx.systems_optimisation.Generation(
                chromosomes=chromo_it.dedupe_population(chromosomes),
                fitness_calculator=fitness_calculator,
                structure_calculator=structure_calculator,
                num_processes=scan_config["num_processes"],  # type:ignore[arg-type]
            )
            for seed in scan_config["neighbour_seeds"]:  # type:ignore[attr-defined]
                temp_scan_config = scan_config.copy()
                temp_scan_config.update({"selection_size": 50})
                temp_scan_config.update(
                    {
                        "num_generations": int(scan_config["num_generations"])  # type:ignore[call-overload]
                        * 2
                    }
                )
                seeded_generations[seed] = run_genetic_algorithm(
                    seed=seed,
                    chromo_it=chromo_it,
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    scan_config=temp_scan_config,
                    database=database,
                    elite_population=elite_population,
                    neighbour_opt=True,
                )
                progress_plot(
                    seeded_generations=seeded_generations,
                    output=figure_dir / f"fp_{system_name}.png",
                )

            # Report.
            found = set()
            for generations in seeded_generations.values():
                for generation in generations:
                    for chromo in generation.chromosomes:
                        found.add(chromo.name)
            logger.info(
                "%s chromosomes found in EA (of %s)",
                len(found),
                chromo_it.get_num_chromosomes(),
            )

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5), sharex=True, sharey=True)
    ss_axes = {"9-9-9": axs[0], "12-6-9": axs[1], "6-12-9": axs[2]}
    top_fit = {"9-9-9": None, "12-6-9": None, "6-12-9": None}
    for entry in database.get_entries():
        if (
            "stoichstring" not in entry.properties
            or "fitness" not in entry.properties
        ):
            continue

        ss_str = entry.properties["stoichstring"]
        ax = ss_axes[ss_str]  # type:ignore[index]
        ax.scatter(
            entry.properties["topology_idx"],
            entry.properties["bb_config_idx"],
            c="gray",
            edgecolor="none",
            s=20,
            marker="o",
            alpha=1.0,
            zorder=-2,
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("topology idx", fontsize=16)
        if ss_str == "9-9-9":
            ax.set_ylabel(r"config idx", fontsize=16)

        fitness = entry.properties["fitness"]
        if top_fit[ss_str] is None or fitness > top_fit[ss_str][2]:  # type:ignore[index]
            top_fit[ss_str] = (  # type:ignore[assignment, index]
                entry.properties["topology_idx"],
                entry.properties["bb_config_idx"],
                fitness,
            )
        continue

    for ss_str, ax in ss_axes.items():
        if top_fit[ss_str] is None:
            continue
        ax.scatter(
            top_fit[ss_str][0],  # type:ignore[index]
            top_fit[ss_str][1],  # type:ignore[index]
            marker="P",
            s=80,
            c="tab:red",
            zorder=2,
        )
        ax.set_title(
            f"$s=${ss_str} ({top_fit[ss_str][0]}, {top_fit[ss_str][1]})",  # type:ignore[index]
            fontsize=16,
        )

    fig.tight_layout()
    fig.savefig(
        figure_dir / "space_explored.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    energies: dict[str, list[float]] = {
        "9-9-9": [],
        "12-6-9": [],
        "6-12-9": [],
    }
    cs = {
        "9-9-9": "tab:blue",
        "12-6-9": "tab:orange",
        "6-12-9": "tab:green",
    }
    for entry in database.get_entries():
        if (
            "stoichstring" not in entry.properties
            or "fitness" not in entry.properties
        ):
            continue

        ss_str = entry.properties["stoichstring"]
        energies[ss_str].append(entry.properties["energy_per_bb"])  # type:ignore[index,arg-type]

    steps = range(len(energies) - 1, -1, -1)
    xmin = 0
    xmax = 10
    xwidth = 0.2
    ystep = 1
    xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
    for i, (ss_str, xdata) in enumerate(energies.items()):
        ax.hist(
            x=xdata,
            bins=xbins,  # type:ignore[arg-type]
            density=True,
            bottom=steps[i] * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
            facecolor=cs[ss_str],
            label=f"{ss_str}",
        )
        if len(xdata) == 0:
            continue
        ax.text(
            x=6,
            y=(steps[i] * ystep) + 0.2,
            s=f"min.: {round(min(xdata), 3)}",
            fontsize=16,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$E_{\mathrm{b}}$ [kjmol-1]", fontsize=16)
    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "energy_dist.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
