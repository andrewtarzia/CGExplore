"""Copiable code from Recipe #1."""  # noqa: INP001

import logging
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import openmm
import stk

import cgexplore as cgx

logger = logging.getLogger(__name__)


def isomer_energy() -> float:
    """Define threshold."""
    return 0.3


def colours() -> dict[str, str]:
    """Colours map to topologies."""
    return {
        "2P3": "tab:blue",
        "4P6": "tab:orange",
        "4P62": "tab:green",
        "6P9": "tab:red",
        "8P12": "tab:purple",
    }


def analyse_cage(
    database: cgx.utilities.AtomliteDatabase,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    chromosome: cgx.systems_optimisation.Chromosome,
) -> None:
    """Analyse a cage."""
    entry = database.get_entry(key=name)
    properties = entry.properties
    molecule = database.get_molecule(key=name)

    topology_str, _ = chromosome.get_topology_information()
    database.add_properties(
        key=name,
        property_dict={
            "cage_name": name,
            "prefix": name.split("_")[0],
            "chromosome": tuple(int(i) for i in chromosome.name),
            "topology": topology_str,
            "num_bbs": cgx.topologies.stoich_map(topology_str),
            "energy_per_bb": cgx.utilities.get_energy_per_bb(
                energy_decomposition=properties["energy_decomposition"],
                number_building_blocks=cgx.topologies.stoich_map(topology_str),
            ),
            "forcefield_dict": forcefield.get_forcefield_dictionary(),
            "opt_pore_data": cgx.analysis.GeomMeasure().calculate_min_distance(
                molecule
            ),
        },
    )


def optimise_cage(  # noqa: PLR0913
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: cgx.forcefields.ForceField,
    platform: str | None,
    database: cgx.utilities.AtomliteDatabase,
) -> cgx.molecular.Conformer:
    """Run geometry optimisation."""
    fina_mol_file = output_dir / f"{name}_final.mol"

    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)

        return cgx.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if fina_mol_file.exists():
        ensemble = cgx.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=output_dir / f"{name}_base.mol",
            conformer_xyz=output_dir / f"{name}_ensemble.xyz",
            data_json=output_dir / f"{name}_ensemble.json",
            overwrite=False,
        )
        conformer = ensemble.get_lowest_e_conformer()
        database.add_molecule(molecule=conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": conformer.energy_decomposition,
                "source": conformer.source,
                "optimised": True,
            },
        )
        return ensemble.get_lowest_e_conformer()

    logger.info("optimising %s", name)
    assigned_system = forcefield.assign_terms(molecule, name, output_dir)
    ensemble = cgx.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemble.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
        overwrite=True,
    )
    temp_molecule = cgx.utilities.run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=50,
        angle_ff_scale=50,
        max_iterations=20,
        platform=platform,
    )

    try:
        conformer = cgx.utilities.run_optimisation(
            assigned_system=cgx.forcefields.AssignedSystem(
                molecule=temp_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="opt1",
            output_dir=output_dir,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="opt1")
    except openmm.OpenMMException as error:
        if "Particle coordinate is NaN. " not in str(error):
            raise

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    for test_molecule in cgx.utilities.yield_shifted_models(
        temp_molecule,
        forcefield,
        kicks=(1, 2, 3, 4),
    ):
        try:
            conformer = cgx.utilities.run_optimisation(
                assigned_system=cgx.forcefields.AssignedSystem(
                    molecule=test_molecule,
                    forcefield_terms=assigned_system.forcefield_terms,
                    system_xml=assigned_system.system_xml,
                    topology_xml=assigned_system.topology_xml,
                    bead_set=assigned_system.bead_set,
                    vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
                ),
                name=name,
                file_suffix="sopt",
                output_dir=output_dir,
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="shifted")
        except openmm.OpenMMException as error:
            if "Particle coordinate is NaN. " not in str(error):
                raise

    min_energy_conformer = ensemble.get_lowest_e_conformer()

    # Add to atomlite database.
    database.add_molecule(molecule=min_energy_conformer.molecule, key=name)
    database.add_properties(
        key=name,
        property_dict={
            "energy_decomposition": min_energy_conformer.energy_decomposition,
            "source": min_energy_conformer.source,
            "optimised": True,
        },
    )
    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


def fitness_function(  # noqa: PLR0913
    chromosome: cgx.systems_optimisation.Chromosome,
    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,  # noqa: ARG001
) -> float:
    """Calculate fitness."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    target_pore = 2
    name = f"{chromosome.prefix}_{chromosome.get_separated_string()}"
    entry = database.get_entry(name)
    tstr = entry.properties["topology"]
    pore = entry.properties["opt_pore_data"]["min_distance"]
    energy = entry.properties["energy_per_bb"]
    pore_diff = abs(target_pore - pore) / target_pore

    # If energy is too high, return bad fitness.
    if energy > isomer_energy() * 2:
        database.add_properties(
            key=name,
            property_dict={"fitness": 0},
        )
        return 0

    # Else, we check smaller topologies.

    # Select all with the same chromosome except for topology graph to check
    # for self-sorting.

    differ_by_topology = chromosome_generator.select_similar_chromosome(
        chromosome=chromosome,
        free_gene_id=0,
    )

    other_topologies = {}
    current_stoich = cgx.topologies.stoich_map(tstr)
    for other_chromosome in differ_by_topology:
        other_name = (
            f"{other_chromosome.prefix}_"
            f"{other_chromosome.get_separated_string()}"
        )
        other_tstr, _ = other_chromosome.get_topology_information()
        # Only recalculate smaller or equivalent cages.
        if cgx.topologies.stoich_map(other_tstr) <= current_stoich:
            if not database.has_molecule(other_name):
                # Run calculation.
                structure_function(
                    chromosome=other_chromosome,
                    database_path=database_path,
                    calculation_output=calculation_output,
                    structure_output=structure_output,
                    options={},
                )

            other_entry = database.get_entry(other_name)
            other_energy = other_entry.properties["energy_per_bb"]
            other_topologies[other_tstr] = other_energy
    if len(other_topologies) > 0 and min(other_topologies.values()) < energy:
        smaller_is_stable = True
    else:
        smaller_is_stable = False

    fitness = 0 if smaller_is_stable else 1 / (pore_diff + energy)

    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


def structure_function(
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,  # noqa: ARG001
) -> None:
    """Geometry optimisation."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    # Build structure.
    _, topology_fun = chromosome.get_topology_information()
    building_blocks = chromosome.get_building_blocks()
    cage = stk.ConstructedMolecule(topology_fun(building_blocks))
    name = f"{chromosome.prefix}_{chromosome.get_separated_string()}"

    # Select forcefield by chromosome.
    forcefield = chromosome.get_forcefield()

    # Optimise with some procedure.
    conformer = optimise_cage(
        molecule=cage,
        name=name,
        output_dir=calculation_output,
        forcefield=forcefield,
        platform=None,
        database=database,
    )

    if conformer is not None:
        conformer.molecule.write(str(structure_output / f"{name}_optc.mol"))

    # Analyse cage.
    analyse_cage(
        name=name,
        forcefield=forcefield,
        database=database,
        chromosome=chromosome,
    )


def progress_plot(
    generations: list,
    output: pathlib.Path,
    num_generations: int,
) -> None:
    """Make a progress plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fitnesses = [
        generation.calculate_fitness_values() for generation in generations
    ]

    ax.plot(
        [max(i) for i in fitnesses],
        markerfacecolor="#F9A03F",
        label="max",
        lw=2,
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )
    ax.plot(
        [np.mean(i) for i in fitnesses],
        markerfacecolor="#086788",
        lw=2,
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
        label="mean",
    )
    ax.plot(
        [min(i) for i in fitnesses],
        markerfacecolor="#7A8B99",
        label="min",
        lw=2,
        c="k",
        marker="o",
        markersize=10,
        markeredgecolor="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_xlim(0, num_generations)
    ax.set_xticks(range(0, num_generations + 1, 5))
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    # Define working directories.
    wd = (
        pathlib.Path(__file__).resolve().parent
        / ".."
        / ".."
        / "recipes"
        / "recipe_1_output"
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

    # Define a database, and a prefix for naming structure, forcefield and
    # output files.
    prefix = "opt"
    database_path = data_dir / "test.db"
    database = cgx.utilities.AtomliteDatabase(database_path)

    # Define beads.
    bead_library = cgx.molecular.BeadLibrary.from_bead_types(
        # Type and coordination.
        {"a": 3, "b": 2, "c": 2, "o": 2}
    )

    # Define the chromosome generator, holding all the changeable genes.
    chromo_it = cgx.systems_optimisation.ChromosomeGenerator(
        prefix=prefix,
        present_beads=bead_library.get_present_beads(),
        vdw_bond_cutoff=2,
    )
    chromo_it.add_gene(
        iteration=(
            ("2P3", stk.cage.TwoPlusThree),
            ("4P6", stk.cage.FourPlusSix),
            ("4P62", stk.cage.FourPlusSix2),
            ("6P9", stk.cage.SixPlusNine),
            ("8P12", stk.cage.EightPlusTwelve),
        ),
        gene_type="topology",
    )
    # Set some basic building blocks up. This should be run by an algorithm
    # later.
    chromo_it.add_gene(
        iteration=(
            cgx.molecular.TwoC1Arm(
                bead=bead_library.get_from_type("b"),
                abead1=bead_library.get_from_type("c"),
            ),
        ),
        gene_type="precursor",
    )
    chromo_it.add_gene(
        iteration=(
            cgx.molecular.ThreeC1Arm(
                bead=bead_library.get_from_type("a"),
                abead1=bead_library.get_from_type("o"),
            ),
        ),
        gene_type="precursor",
    )

    definer_dict = {
        # Bonds.
        "ao": ("bond", 1.5, 1e5),
        "bc": ("bond", 1.5, 1e5),
        "co": ("bond", 1.0, 1e5),
        "cc": ("bond", 1.0, 1e5),
        "oo": ("bond", 1.0, 1e5),
        # Angles.
        "ccb": ("angle", 180.0, 1e2),
        "ooc": ("angle", 180.0, 1e2),
        "occ": ("angle", 180.0, 1e2),
        "ccc": ("angle", 180.0, 1e2),
        "oco": ("angle", 180.0, 1e2),
        "aoc": ("angle", 180.0, 1e2),
        "aoo": ("angle", 180.0, 1e2),
        "bco": ("angle", tuple(i for i in range(90, 181, 5)), 1e2),
        "cbc": ("angle", 180.0, 1e2),
        "oao": ("angle", tuple(i for i in range(50, 121, 5)), 1e2),
        # Torsions.
        "ocbco": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "o": ("nb", 10.0, 1.0),
    }
    chromo_it.add_forcefield_dict(definer_dict=definer_dict)

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
    structure_calculator = cgx.systems_optimisation.StructureCalculator(
        structure_function=structure_function,
        structure_output=struct_output,
        calculation_output=calc_dir,
        database_path=database_path,
        options={},
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
            logger.info("doing generation %s of seed %s", generation_id, seed)
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
            merged_chromosomes.extend(generation.select_best(selection_size=5))

            generation = cgx.systems_optimisation.Generation(
                chromosomes=chromo_it.dedupe_population(merged_chromosomes),
                fitness_calculator=fitness_calculator,
                structure_calculator=structure_calculator,
                num_processes=num_processes,
            )
            logger.info("new size is %s.", generation.get_generation_size())

            # Build, optimise and analyse each structure.
            generation.run_structures()
            _ = generation.calculate_fitness_values()

            # Add final state to generations.
            generations.append(generation)

            # Select the best of the generation for the next generation.
            best = generation.select_best(selection_size=selection_size)
            generation = cgx.systems_optimisation.Generation(
                chromosomes=chromo_it.dedupe_population(best),
                fitness_calculator=fitness_calculator,
                structure_calculator=structure_calculator,
                num_processes=num_processes,
            )
            logger.info("final size is %s.", generation.get_generation_size())

            progress_plot(
                generations=generations,
                output=figure_dir / f"fitness_progress_{seed}.png",
                num_generations=num_generations,
            )

            # Output best structures as images.
            best_chromosome = generation.select_best(selection_size=1)[0]
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
    ax.axvline(x=2, c="k", zorder=-2)
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
    ax1.set_title(f"E: {isomer_energy()}, F: {fitness_threshold}", fontsize=16)

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
        properties["E_b / kjmol-1"].append(entry.properties["energy_per_bb"])
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

    logging.info("saving %s entries", len(structures))
    cgx.utilities.write_chemiscope_json(
        json_file=data_dir / "space_explored.json.gz",
        structures=structures,
        properties=properties,
        bonds_as_shapes=True,
        meta_dict={
            "name": "Recipe 1 structures.",
            "description": ("Minimal models from recipe 1."),
            "authors": ["Andrew Tarzia"],
            "references": [],
        },
        x_axis_dict={"property": "b_c_o / deg"},
        y_axis_dict={"property": "o_a_o / deg"},
        z_axis_dict={"property": "num_bbs"},
        color_dict={"property": "E_b / kjmol-1", "min": 0, "max": 1.0},
        bond_hex_colour="#919294",
    )


if __name__ == "__main__":
    main()
