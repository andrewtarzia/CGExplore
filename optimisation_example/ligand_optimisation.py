"""Script showing how ligand optimisation can work."""

import itertools as it
import logging
import pathlib
from collections import abc, defaultdict
from dataclasses import dataclass

import cgexplore
import matplotlib.pyplot as plt
import molellipsize as mes
import numpy as np
import openmm
import spindry as spd
import stk


def pymol_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )


def shape_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )


def colours() -> dict[str, str]:
    """Colours map to topologies."""
    return {
        "2P3": "#1f77b4",
        "4P6": "#ff7f0e",
        "4P62": "#2ca02c",
        "6P9": "#d62728",
        "8P12": "#17becf",
    }


def visualise_structures(
    file_prefix: str,
    best_name: str,
    liga_name: str,
    summary_string: str,
    ligand_string: str,
    image_dir: pathlib.Path,
    structure_dir: pathlib.Path,
) -> None:
    """Take structures and make an image of them."""
    best_file = structure_dir / f"{best_name}_optc.mol"
    liga_file = structure_dir / f"{liga_name}_unopt.mol"
    # Make the images with pymol.
    cgexplore.utilities.Pymol(
        output_dir=image_dir,
        file_prefix=f"{file_prefix}_compl",
        settings={
            "grid_mode": 0,
            "rayx": 1000,
            "rayy": 1000,
            "stick_rad": 0.3,
            "vdw": 0,
            "zoom_string": "custom",
            "orient": False,
        },
        pymol_path=pymol_path(),
    ).visualise([best_file], orient_atoms=None)
    cgexplore.utilities.Pymol(
        output_dir=image_dir,
        file_prefix=f"{file_prefix}_guest",
        settings={
            "grid_mode": 0,
            "rayx": 1000,
            "rayy": 1000,
            "stick_rad": 0.3,
            "vdw": 0,
            "zoom_string": "custom",
            "zoom_scale": 1,
            "orient": False,
        },
        pymol_path=pymol_path(),
    ).visualise([liga_file], orient_atoms=None)

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    cgexplore.utilities.add_structure_to_ax(
        ax=axs[0],
        png_file=image_dir / f"{file_prefix}_guest.png",
    )
    cgexplore.utilities.add_text_to_ax(
        x=0.4,
        y=0,
        ax=axs[0],
        text=ligand_string,
    )
    cgexplore.utilities.add_structure_to_ax(
        ax=axs[1],
        png_file=image_dir / f"{file_prefix}_compl.png",
    )
    cgexplore.utilities.add_text_to_ax(
        x=0.4,
        y=0,
        ax=axs[1],
        text=summary_string,
    )

    axs[0].axis("off")
    axs[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        image_dir / f"{file_prefix}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def analyse_complex(
    database: cgexplore.utilities.AtomliteDatabase,
    name: str,
    output_dir: pathlib.Path,  # noqa: ARG001
    forcefield: cgexplore.forcefields.ForceField,
    chromosome: cgexplore.systems_optimisation.Chromosome,
) -> None:
    entry = database.get_entry(key=name)
    properties = entry.properties

    database.add_properties(
        key=name,
        property_dict={
            "prefix": name.split("_")[0],
            "chromosome": tuple(int(i) for i in chromosome.name),
        },
    )

    if "forcefield_dict" not in properties:
        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = forcefield.get_targets()
        s_dict = {}
        e_dict = {}

        for at in ff_targets["nonbondeds"]:
            s_dict[at.bead_class] = at.sigma.value_in_unit(
                openmm.unit.angstrom
            )
            e_dict[at.bead_class] = at.epsilon.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "s_dict": s_dict,
            "e_dict": e_dict,
        }
        database.add_properties(
            key=name,
            property_dict={"forcefield_dict": forcefield_dict},
        )


def fitness_calculator(
    chromosome: cgexplore.systems_optimisation.Chromosome,
    chromosome_generator: cgexplore.systems_optimisation.ChromosomeGenerator,  # noqa: ARG001
    database: cgexplore.utilities.AtomliteDatabase,
    calculation_output: pathlib.Path,  # noqa: ARG001
    structure_output: pathlib.Path,  # noqa: ARG001
) -> float:
    name = f"{chromosome.prefix}_{chromosome.get_string()}"

    entry = database.get_entry(name)

    energy = entry.properties["energy_decomposition"]["potential"]
    centroid_distance = entry.properties["centroid_distance"]
    fitness = np.exp(-0.01 * energy) + (1 / centroid_distance)

    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


def structure_calculator(
    chromosome: cgexplore.systems_optimisation.Chromosome,
    database: cgexplore.utilities.AtomliteDatabase,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    host_structure: stk.BuildingBlock,
) -> None:
    name = f"{chromosome.prefix}_{chromosome.get_string()}"
    (bb,) = chromosome.get_building_blocks()
    # Select forcefield by chromosome.
    forcefield = chromosome.get_forcefield()

    # Optimise with some procedure.
    opt_file = structure_output / f"{name}_optc.mol"
    conformers = {}
    if not opt_file.exists():
        laundry = cgexplore.optimisation.Laundrette(
            num_dockings=10,
            naming_prefix=name,
            output_dir=calculation_output,
            forcefield=forcefield,
            seed=100,
        )
        min_energy_id = "none"
        min_energy = 1e24
        for conformer in laundry.run_dockings(
            host_bb=host_structure,
            guest_bb=bb,
        ):
            id_ = f"{conformer.source}_{conformer.conformer_id}"
            conformers[id_] = conformer
            potential = conformer.energy_decomposition["potential"]
            if potential < min_energy:
                min_energy = conformer.energy_decomposition["potential"]
                min_energy_id = id_

        min_energy_conformer = conformers[min_energy_id]

        # Add to atomlite database.
        database.add_molecule(
            molecule=min_energy_conformer.to_stk_molecule(),
            key=name,
        )
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": (
                    min_energy_conformer.energy_decomposition
                ),
                "source": min_energy_id,
                "optimised": True,
            },
        )

        # Do some analysis while you have the spd.supramolecule.
        database.add_properties(
            key=name,
            property_dict={
                "centroid_distance": spd.calculate_centroid_distance(
                    min_energy_conformer.supramolecule
                ),
                "min_hg_distance": spd.calculate_min_atom_distance(
                    min_energy_conformer.supramolecule
                ),
            },
        )
        min_energy_conformer.to_stk_molecule().write(opt_file)

    if "host_name" not in database.get_entry(key=name).properties:
        cid = chromosome.prefix.replace("opt", "")
        host_name = f"host{cid}"
        host_properties = database.get_entry(key=host_name).properties

        mes_mol = mes.Molecule(bb.to_rdkit_mol(), conformers=[0])
        conf_ellipsoids = mes_mol.get_ellipsoids(
            vdwscale=0.45,
            boxmargin=4.0,
            spacing=0.25,
        )

        database.add_properties(
            key=name,
            property_dict={
                "host_name": host_name,
                "host_pore": host_properties["host_pore"],
                "host_size": host_properties["host_size"],
                "host_shape": host_properties["host_shape"],
                "guest_diameters": conf_ellipsoids[0][1],
                "guest_ratios": mes_mol.get_inertial_ratios(),
            },
        )

    # Analyse cage.
    analyse_complex(
        name=name,
        output_dir=calculation_output,
        forcefield=forcefield,
        database=database,
        chromosome=chromosome,
    )


def progress_plot(
    generations: list,
    output: pathlib.Path,
    num_generations: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fitnesses = [
        generation.calculate_fitness_values() for generation in generations
    ]

    ax.plot(
        [max(i) for i in fitnesses],
        c="#F9A03F",
        label="max",
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="k",
    )
    ax.plot(
        [np.mean(i) for i in fitnesses],
        c="#086788",
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="k",
        label="mean",
    )
    ax.plot(
        [min(i) for i in fitnesses],
        c="#7A8B99",
        label="min",
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_xlim(0, num_generations)
    ax.set_xticks(range(0, num_generations + 1, 5))
    ax.set_ylim(0, None)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


@dataclass
class HostGeneration(cgexplore.systems_optimisation.Generation):
    """Define the chromosomes in a single generation."""

    def calculate_fitness_values(self) -> list[float]:
        """Calculate the fitness of all chromosomes."""
        return [
            self.fitness_calculator(
                chromosome=i,
                chromosome_generator=self.chromosome_generator,
                database=self.database,
                calculation_output=self.calculation_output,
                structure_output=self.structure_output,
            )
            for i in self.chromosomes
        ]

    def run_structures(self, host_structure: stk.BuildingBlock) -> None:
        """Run the production and analyse of all chromosomes."""
        length = len(self.chromosomes)
        for i, chromosome in enumerate(self.chromosomes):
            logging.info(f"building {chromosome} ({i+1} of {length})")
            self.structure_calculator(
                chromosome=chromosome,
                database=self.database,
                calculation_output=self.calculation_output,
                structure_output=self.structure_output,
                host_structure=host_structure,
            )

    def select_best(
        self,
        selection_size: int,
    ) -> abc.Iterable[cgexplore.systems_optimisation.Chromosome]:
        """Select the best in the generation by fitness."""
        temp = [
            (
                i,
                self.fitness_calculator(
                    chromosome=i,
                    chromosome_generator=self.chromosome_generator,
                    database=self.database,
                    calculation_output=self.calculation_output,
                    structure_output=self.structure_output,
                ),
            )
            for i in self.chromosomes
        ]
        best_indices = tuple(
            sorted(range(len(temp)), key=lambda i: temp[i][1], reverse=True)
        )[:selection_size]

        return [self.chromosomes[i] for i in best_indices]


def add_scatter(ax: plt.Axes, x: str, y: str, datas: dict) -> None:
    length = len(datas["fitness"])
    fit_thresh = 10
    high_fit_c = "#F9A03F"
    high_fit_s = 80
    high_fit_m = "o"
    low_fit_c = "#7A8B99"
    low_fit_s = 30
    low_fit_m = "o"

    ax.scatter(
        [
            datas[x][i]
            for i in range(length)
            if datas["fitness"][i] < fit_thresh
        ],
        [
            datas[y][i]
            for i in range(length)
            if datas["fitness"][i] < fit_thresh
        ],
        c=low_fit_c,
        edgecolor="none",
        s=low_fit_s,
        marker=low_fit_m,
        alpha=1.0,
    )
    ax.scatter(
        [
            datas[x][i]
            for i in range(length)
            if datas["fitness"][i] > fit_thresh
        ],
        [
            datas[y][i]
            for i in range(length)
            if datas["fitness"][i] > fit_thresh
        ],
        c=high_fit_c,
        edgecolor="k",
        s=high_fit_s,
        marker=high_fit_m,
        alpha=1.0,
    )


def main() -> None:
    wd = pathlib.Path("/home/atarzia/workingspace/cage_optimisation_tests")
    struct_output = wd / "ligand_structures"
    cgexplore.utilities.check_directory(struct_output)
    calc_dir = wd / "ligand_calculations"
    cgexplore.utilities.check_directory(calc_dir)
    data_dir = wd / "ligand_data"
    cgexplore.utilities.check_directory(data_dir)
    figure_dir = wd / "ligand_figures"
    cgexplore.utilities.check_directory(figure_dir)
    all_dir = figure_dir / "all"
    cgexplore.utilities.check_directory(all_dir)
    best_dir = figure_dir / "best"
    cgexplore.utilities.check_directory(best_dir)

    database = cgexplore.utilities.AtomliteDatabase(data_dir / "test.db")

    xbead = cgexplore.molecular.CgBead(
        element_string="Co",
        bead_class="x",
        bead_type="x",
        coordination=3,
    )
    ybead = cgexplore.molecular.CgBead(
        element_string="Fe",
        bead_class="y",
        bead_type="y",
        coordination=2,
    )
    # Host beads.
    h1bead = cgexplore.molecular.CgBead(
        element_string="Pd",
        bead_class="m",
        bead_type="m",
        coordination=4,
    )
    h2bead = cgexplore.molecular.CgBead(
        element_string="C",
        bead_class="n",
        bead_type="n",
        coordination=3,
    )
    h3bead = cgexplore.molecular.CgBead(
        element_string="Pb",
        bead_class="b",
        bead_type="b",
        coordination=2,
    )
    h4bead = cgexplore.molecular.CgBead(
        element_string="Ba",
        bead_class="a",
        bead_type="a",
        coordination=2,
    )
    h5bead = cgexplore.molecular.CgBead(
        element_string="Ag",
        bead_class="c",
        bead_type="c",
        coordination=2,
    )

    num_beads = 6
    compositions = [
        i
        for i in it.product(range(num_beads + 1), repeat=num_beads)
        if cgexplore.molecular.check_fit(i, num_beads=num_beads, max_shell=6)
    ]
    compositions = sorted(compositions, reverse=True)

    # Settings.
    seeds = [4, 280, 999, 2196]
    num_generations = 20
    selection_size = 10

    # Now we want to optimise this for binding a specific host (actually a
    # series of hosts).
    hosts = [
        stk.BuildingBlock.init_from_file(
            str(wd / "hosts" / "6P8_4C1m1b1_3C1n1b1_f9_optc.mol")
        ),
        stk.BuildingBlock.init_from_file(
            str(wd / "hosts" / "6P8_4C1m1b1_3C1n1b1_f27_optc.mol")
        ),
        stk.BuildingBlock.init_from_file(
            str(wd / "hosts" / "4P6_3C1n1b1_2C1c1a1_f127_optc.mol")
        ),
        stk.BuildingBlock.init_from_file(
            str(wd / "hosts" / "4P6_3C1n1b1_2C1c1a1_f291_optc.mol")
        ),
        stk.BuildingBlock.init_from_file(
            str(wd / "hosts" / "6P12_4C1m1b1_2C1c1a1_f182_optc.mol")
        ),
        stk.BuildingBlock.init_from_file(
            str(wd / "hosts" / "6P12_4C1m1b1_2C1c1a1_f78_optc.mol")
        ),
    ]
    host_info = {
        "host0": ("Pd", 6),
        "host1": ("Pd", 6),
        "host2": ("Ag", 6),
        "host3": ("Ag", 6),
        "host4": ("Pd", 6),
        "host5": ("Pd", 6),
    }
    for host_id, host in enumerate(hosts):
        prefix = f"opt{host_id}"

        # Add some host measures to database.
        host_name = f"host{host_id}"
        if not database.has_molecule(key=host_name):
            database.add_molecule(molecule=host, key=host_name)
            database.add_properties(
                key=host_name,
                property_dict={"prefix": "host"},
            )

        entry = database.get_entry(key=host_name)
        properties = entry.properties

        if "node_shape_measures" not in properties:
            shape_calc = cgexplore.analysis.ShapeMeasure(
                output_dir=(calc_dir / f"{host_name}_nshape"),
                shape_path=shape_path(),
                shape_string=None,
            )

            n_shape_mol = shape_calc.get_shape_molecule_byelement(
                molecule=host,
                element=host_info[host_name][0],
                expected_points=host_info[host_name][1],
            )
            node_shape_measures = shape_calc.calculate(n_shape_mol)

            host_analysis = cgexplore.analysis.GeomMeasure()
            database.add_properties(
                key=host_name,
                property_dict={
                    "host_shape": node_shape_measures,
                    "host_pore": host_analysis.calculate_min_distance(host),
                    "host_size": host_analysis.calculate_max_diameter(host),
                },
            )

        chromosome_gen = cgexplore.systems_optimisation.ChromosomeGenerator(
            prefix=prefix,
            present_beads=(
                xbead,
                ybead,
                h1bead,
                h2bead,
                h3bead,
                h4bead,
                h5bead,
            ),
            vdw_bond_cutoff=2,
        )

        chromosome_gen.add_gene(
            iteration=(
                cgexplore.molecular.PrecursorGenerator(
                    composition=i,
                    present_beads=(
                        xbead,
                        ybead,
                        ybead,
                        ybead,
                        ybead,
                        ybead,
                        ybead,
                    ),
                    binder_beads=(),
                    placer_beads=(),
                    bead_distance=1.5,
                )
                for i in compositions
            ),
            gene_type="precursor",
        )

        # Add modifications to nonbonded interactions.
        nb_epsilons = (1, 5, 10, 15)
        nb_sigmas = (0.7, 1, 1.5, 2)
        definer_dict = {}
        present_beads = (xbead, ybead)
        for options in present_beads:
            type_string = f"{options.bead_type}"
            definer_dict[type_string] = ("nb", nb_epsilons, nb_sigmas)

        # Host nonbonded terms are constant.
        definer_dict["m"] = ("nb", 10, 1)
        definer_dict["n"] = ("nb", 10, 1)
        definer_dict["b"] = ("nb", 10, 1)
        definer_dict["a"] = ("nb", 10, 1)
        definer_dict["c"] = ("nb", 10, 1)

        chromosome_gen.add_forcefield_dict(definer_dict=definer_dict)

        for seed in seeds:
            generator = np.random.default_rng(seed)

            initial_population = chromosome_gen.select_random_population(
                generator,
                size=selection_size,
            )

            # Yield this.
            generations = []

            generation = HostGeneration(
                chromosomes=initial_population,
                chromosome_generator=chromosome_gen,
                fitness_calculator=fitness_calculator,
                structure_calculator=structure_calculator,
                structure_output=struct_output,
                calculation_output=calc_dir,
                database=database,
            )

            generation.run_structures(host)
            _ = generation.calculate_fitness_values()
            generations.append(generation)

            progress_plot(
                generations=generations,
                output=figure_dir / f"fitness_progress_{seed}_{host_id}.png",
                num_generations=num_generations,
            )

            for generation_id in range(1, num_generations + 1):
                logging.info(
                    f"doing generation {generation_id} of seed {seed} with "
                    f"host {host_id}"
                )
                logging.info(
                    f"initial size is {generation.get_generation_size()}."
                )
                logging.info("doing mutations.")
                merged_chromosomes = []
                merged_chromosomes.extend(
                    chromosome_gen.mutate_population(
                        list_of_chromosomes=generation.chromosomes,
                        generator=generator,
                        gene_range=chromosome_gen.get_term_ids(),
                        selection="random",
                        num_to_select=5,
                        database=database,
                    )
                )
                merged_chromosomes.extend(
                    chromosome_gen.mutate_population(
                        list_of_chromosomes=generation.chromosomes,
                        generator=generator,
                        gene_range=chromosome_gen.get_prec_ids(),
                        selection="random",
                        num_to_select=5,
                        database=database,
                    )
                )
                merged_chromosomes.extend(
                    chromosome_gen.mutate_population(
                        list_of_chromosomes=generation.chromosomes,
                        generator=generator,
                        gene_range=chromosome_gen.get_term_ids(),
                        selection="roulette",
                        num_to_select=5,
                        database=database,
                    )
                )
                merged_chromosomes.extend(
                    chromosome_gen.mutate_population(
                        list_of_chromosomes=generation.chromosomes,
                        generator=generator,
                        gene_range=chromosome_gen.get_prec_ids(),
                        selection="roulette",
                        num_to_select=5,
                        database=database,
                    )
                )

                merged_chromosomes.extend(
                    chromosome_gen.crossover_population(
                        list_of_chromosomes=generation.chromosomes,
                        generator=generator,
                        selection="random",
                        num_to_select=5,
                        database=database,
                    )
                )

                merged_chromosomes.extend(
                    chromosome_gen.crossover_population(
                        list_of_chromosomes=generation.chromosomes,
                        generator=generator,
                        selection="roulette",
                        num_to_select=5,
                        database=database,
                    )
                )

                # Add the best 5 to the new generation.
                merged_chromosomes.extend(
                    generation.select_best(selection_size=5)
                )

                generation = HostGeneration(
                    chromosomes=chromosome_gen.dedupe_population(
                        merged_chromosomes
                    ),
                    chromosome_generator=chromosome_gen,
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    structure_output=struct_output,
                    calculation_output=calc_dir,
                    database=database,
                )
                logging.info(
                    f"new size is {generation.get_generation_size()}."
                )

                # Build, optimise and analyse each structure.
                generation.run_structures(host_structure=host)
                _ = generation.calculate_fitness_values()

                # Add final state to generations.
                generations.append(generation)

                # Select the best of the generation for the next generation.
                logging.info("maybe use roulette here?")
                best = generation.select_best(selection_size=selection_size)
                generation = HostGeneration(
                    chromosomes=chromosome_gen.dedupe_population(best),
                    chromosome_generator=chromosome_gen,
                    fitness_calculator=fitness_calculator,
                    structure_calculator=structure_calculator,
                    structure_output=struct_output,
                    calculation_output=calc_dir,
                    database=database,
                )
                logging.info(
                    f"final size is {generation.get_generation_size()}."
                )

                progress_plot(
                    generations=generations,
                    output=figure_dir
                    / f"fitness_progress_{seed}_{host_id}.png",
                    num_generations=num_generations,
                )

                # Output best structures as images.
                best_chromosome = generation.select_best(selection_size=1)[0]
                best_name = (
                    f"{best_chromosome.prefix}_{best_chromosome.get_string()}"
                )
                best_energy = database.get_property(
                    key=f"{best_chromosome.prefix}_{best_chromosome.get_string()}",
                    property_key="energy_decomposition",
                    property_type=dict,
                )["potential"]
                summary_string = f"E={round(best_energy,2)}"
                liga_name = best_chromosome.get_string()[:-4]
                (precursor,) = best_chromosome.get_precursors()
                ligand_string = "".join(str(i) for i in precursor.composition)

                visualise_structures(
                    file_prefix=f"{prefix}_{seed}_g{generation_id}_h{host_id}_best",
                    best_name=best_name,
                    liga_name=liga_name,
                    summary_string=summary_string,
                    ligand_string=ligand_string,
                    image_dir=best_dir,
                    structure_dir=struct_output,
                )

            logging.info(
                f"top scorer is {best_name} (seed: {seed}, host: {host_id})"
            )

        # Report.
        found = set()
        for generation in generations:
            for chromo in generation.chromosomes:
                found.add(chromo.name)
        logging.info(
            f"{len(found)} chromosomes found in EA (of "
            f"{chromosome_gen.get_num_chromosomes()})"
        )

        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))
        (ax, ax1, ax2), (ax3, ax4, ax5) = axs

        plotted = 0
        datas = defaultdict(list)
        for entry in database.get_entries():
            if prefix != entry.properties["prefix"]:
                continue

            datas["fitness"].append(entry.properties["fitness"])
            chromosome = chromosome_gen.select_chromosome(
                entry.properties["chromosome"]
            )
            (precursor,) = chromosome.get_precursors()

            datas["centroid_distance"].append(
                entry.properties["centroid_distance"]
            )
            datas["potential"].append(
                entry.properties["energy_decomposition"]["potential"]
            )

            datas["first_number"].append(precursor.composition[0])

            datas["x_bead_s"].append(
                entry.properties["forcefield_dict"]["s_dict"]["x"]
            )
            datas["x_bead_e"].append(
                entry.properties["forcefield_dict"]["e_dict"]["x"]
            )
            datas["y_bead_s"].append(
                entry.properties["forcefield_dict"]["s_dict"]["y"]
            )
            datas["y_bead_e"].append(
                entry.properties["forcefield_dict"]["e_dict"]["y"]
            )

            datas["diameter_1"].append(entry.properties["guest_diameters"][0])
            datas["diameter_2"].append(entry.properties["guest_diameters"][1])

            ratio_1, ratio_2 = entry.properties["guest_ratios"]["0"]
            datas["ratio_1"].append(ratio_1)
            datas["ratio_2"].append(ratio_2)
            plotted += 1

        add_scatter(ax=ax, x="centroid_distance", y="potential", datas=datas)
        add_scatter(ax=ax4, x="diameter_1", y="diameter_2", datas=datas)
        add_scatter(ax=ax5, x="ratio_1", y="ratio_2", datas=datas)

        length = len(datas["fitness"])
        xwidth = 2
        fitness_lim = (min(datas["fitness"]), max(datas["fitness"]))
        fit_thresh = 10
        for fn in (1, 2, 3, 4, 5, 6):
            xdata = [
                datas["fitness"][i]
                for i in range(length)
                if datas["first_number"][i] == fn
                # and datas["fitness"][i] > fit_thresh
            ]
            if len(xdata) == 0:
                continue

            xbins = np.arange(
                fitness_lim[0] - xwidth, fitness_lim[1] + xwidth, xwidth
            )
            ax1.hist(
                x=xdata,
                bins=xbins,
                density=True,
                # bottom=fn,
                histtype="stepfilled",
                stacked=True,
                linewidth=1.0,
                # facecolor=leg_info()[ami]["c"],
                alpha=1.0,
                # color=leg_info()[ami]['c'],
                # color='white',
                edgecolor="k",
                label=fn,
            )

        xwidth = 0.2
        xlim = (0, 3)
        for x_name in ("x_bead_s", "y_bead_s"):
            xdata = [
                datas[x_name][i]
                for i in range(length)
                if datas["fitness"][i] > fit_thresh
            ]
            if len(xdata) == 0:
                continue

            xbins = np.arange(xlim[0] - xwidth, xlim[1] + xwidth, xwidth)
            ax2.hist(
                x=xdata,
                bins=xbins,
                density=False,
                histtype="stepfilled",
                stacked=True,
                linewidth=1.0,
                alpha=0.8,
                edgecolor="k",
                label=x_name,
            )

        xwidth = 0.5
        xlim = (0, 20)
        for x_name in ("x_bead_e", "y_bead_e"):
            xdata = [
                datas[x_name][i]
                for i in range(length)
                if datas["fitness"][i] > fit_thresh
            ]
            if len(xdata) == 0:
                continue

            xbins = np.arange(xlim[0] - xwidth, xlim[1] + xwidth, xwidth)
            ax3.hist(
                x=xdata,
                bins=xbins,
                density=False,
                histtype="stepfilled",
                stacked=True,
                linewidth=1.0,
                alpha=0.8,
                edgecolor="k",
                label=x_name,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("centroid distance", fontsize=16)
        ax.set_ylabel("energy", fontsize=16)
        ax.set_xlim(-0.5, 4)
        ax.set_ylim(-140, 10)
        ax.set_title(
            f"plotted: {plotted}, found: {len(found)}, "
            f"possible: {chromosome_gen.get_num_chromosomes()}",
            fontsize=16,
        )

        ax1.tick_params(axis="both", which="major", labelsize=16)
        ax1.set_xlabel("fitness", fontsize=16)
        ax1.set_ylabel("frequency", fontsize=16)
        ax1.legend(fontsize=16)
        ax1.set_xlim(0, 120)
        ax1.set_yscale("log")

        ax2.tick_params(axis="both", which="major", labelsize=16)
        ax2.set_xlabel("sigma", fontsize=16)
        ax2.set_ylabel("count high fitness", fontsize=16)
        ax2.legend(fontsize=16)
        ax2.set_xlim(0, 3)

        ax3.tick_params(axis="both", which="major", labelsize=16)
        ax3.set_xlabel("epsilon", fontsize=16)
        ax3.set_ylabel("count high fitness", fontsize=16)
        ax3.legend(fontsize=16)
        ax3.set_xlim(0, 20)

        ax4.tick_params(axis="both", which="major", labelsize=16)
        ax4.set_xlabel("diameter 1", fontsize=16)
        ax4.set_ylabel("diameter 2", fontsize=16)
        ax4.set_xlim(0, 5)
        ax4.set_ylim(0, 10)

        ax5.tick_params(axis="both", which="major", labelsize=16)
        ax5.set_xlabel("NPR1", fontsize=16)
        ax5.set_ylabel("NPR2", fontsize=16)
        ax5.plot((0.5, 0, 1, 0.5), (0.5, 1, 1, 0.5), c="k", lw=1, alpha=0.4)
        ax5.text(x=0.55, y=0.5, s="d", fontsize=16)
        ax5.text(x=0, y=1.05, s="r", fontsize=16)
        ax5.text(x=1, y=1.05, s="s", fontsize=16)
        ax5.set_ylim(0.45, 1.1)

        fig.tight_layout()
        fig.savefig(
            figure_dir / f"space_explored_{host_id}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    main()
