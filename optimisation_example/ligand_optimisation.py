"""Script showing how ligand optimisation can work."""

import itertools as it
import logging
import pathlib
from collections import abc
from dataclasses import dataclass

import cgexplore
import matplotlib.pyplot as plt
import numpy as np
import openmm
import spindry as spd
import stk
from scipy.spatial.distance import cdist


def pymol_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
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


class CgAtom(spd.Atom):
    """A new spindry atom with epsilon."""

    def __init__(
        self,
        id: int,  # noqa: A002
        element_string: str,
        epsilon: float,
        sigma: float,
    ) -> None:
        """Initialize CgAtom."""
        super().__init__(id, element_string)
        self._epsilon = epsilon
        self._sigma = sigma

    def get_epsilon(self) -> float:
        """Get the epsilon value."""
        return self._epsilon

    def get_sigma(self) -> float:
        """Get the sigma value."""
        return self._sigma


class CgPotential(spd.Potential):
    """A Cg non-bonded potential function."""

    def _nonbond_potential(
        self,
        distance: np.ndarray,
        sigmas: np.ndarray,
        epsilons: np.ndarray,
    ) -> np.ndarray:
        """Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """
        return epsilons * (
            (sigmas / distance) ** 12 - (sigmas / distance) ** 6
        )

    def _combine_sigma(self, radii1: float, radii2: float) -> float:
        """Combine radii using Lorentz-Berthelot rules."""
        len1 = len(radii1)
        len2 = len(radii2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = (radii1[i] + radii2[j]) / 2

        return mixed

    def _combine_epsilon(self, e1: float, e2: float) -> float:
        """Combine epsilon using Lorentz-Berthelot rules."""
        len1 = len(e1)
        len2 = len(e2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = np.sqrt(e1[i] * e2[j])

        return mixed

    def _compute_nonbonded_potential(
        self,
        position_matrices: np.ndarray,
        radii: np.ndarray,
        epsilons: np.ndarray,
    ) -> float:

        nonbonded_potential = 0
        for pos_mat_pair, radii_pair, epsilon_pair in zip(
            it.combinations(position_matrices, 2),
            it.combinations(radii, 2),
            it.combinations(epsilons, 2),
            strict=False,
        ):

            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            sigmas = self._combine_sigma(radii_pair[0], radii_pair[1])
            epsilons = self._combine_epsilon(epsilon_pair[0], epsilon_pair[1])
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigmas=sigmas.flatten(),
                    epsilons=epsilons.flatten(),
                )
            )

        return nonbonded_potential

    def compute_potential(self, supramolecule: spd.SupraMolecule) -> float:
        """Compure the potential of the molecule."""
        component_position_matrices = (
            i.get_position_matrix() for i in supramolecule.get_components()
        )
        component_radii = (
            tuple(j.get_sigma() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        component_epsilon = (
            tuple(j.get_epsilon() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        return self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
            epsilons=component_epsilon,
        )


class Laundrette:
    """Class to run rigid-body docking."""

    def __init__(
        self,
        num_dockings: int,
        naming_prefix: str,
        output_dir: pathlib.Path,
        forcefield: cgexplore.forcefields.ForceField,
        seed: int,
    ) -> None:
        """Initialise Laundrette."""
        self._num_dockings = num_dockings
        self._naming_prefix = naming_prefix
        self._output_dir = output_dir
        self._potential = CgPotential()
        self._forcefield = forcefield
        self._rng = np.random.default_rng(seed=seed)

    def _get_supramolecule(
        self,
        hgcomplex: stk.ConstructedMolecule,
    ) -> spd.Potential:
        nonbonded_targets = self._forcefield.get_targets()["nonbondeds"]

        epsilons = []
        sigmas = []
        for atom in hgcomplex.get_atoms():
            atom_estring = atom.__class__.__name__
            cgbead = (
                self._forcefield.get_bead_library().get_cgbead_from_element(
                    atom_estring
                )
            )
            for target_term in nonbonded_targets:
                if target_term.bead_class != cgbead.bead_class:
                    continue
                epsilons.append(
                    target_term.epsilon.value_in_unit(
                        openmm.unit.kilojoules_per_mole
                    )
                )
                sigmas.append(
                    target_term.sigma.value_in_unit(openmm.unit.angstrom)
                )

        return spd.SupraMolecule(
            atoms=(
                CgAtom(
                    id=atom.get_id(),
                    element_string=atom.__class__.__name__,
                    epsilon=epsilons[atom.get_id()],
                    sigma=sigmas[atom.get_id()],
                )
                for atom in hgcomplex.get_atoms()
            ),
            bonds=(
                spd.Bond(
                    id=i,
                    atom_ids=(
                        bond.get_atom1().get_id(),
                        bond.get_atom2().get_id(),
                    ),
                )
                for i, bond in enumerate(hgcomplex.get_bonds())
            ),
            position_matrix=hgcomplex.get_position_matrix(),
        )

    def run_dockings(
        self,
        host_bb: stk.BuildingBlock,
        guest_bb: stk.BuildingBlock,
    ) -> abc.Iterable[cgexplore.molecular.SpindryConformer]:
        """Run the docking algorithm."""
        for docking_id in range(self._num_dockings):
            logging.info(f"docking run: {docking_id+1}")

            guest = stk.host_guest.Guest(
                building_block=guest_bb,
                start_vector=guest_bb.get_direction(),
                end_vector=self._rng.random((1, 3))[0],
                # Change the displacement of the guest.
                displacement=self._rng.random((1, 3))[0],
            )

            hgcomplex = stk.ConstructedMolecule(
                topology_graph=stk.host_guest.Complex(
                    host=stk.BuildingBlock.init_from_molecule(host_bb),
                    guests=guest,
                ),
            )
            supramolecule = self._get_supramolecule(hgcomplex=hgcomplex)

            cg = spd.Spinner(
                step_size=1.0,
                rotation_step_size=2.0,
                num_conformers=200,
                max_attempts=500,
                potential_function=self._potential,
                beta=1.0,
                random_seed=None,
            )
            cid = 1
            for supraconformer in cg.get_conformers(supramolecule):

                yield cgexplore.molecular.SpindryConformer(
                    supramolecule=supraconformer,
                    conformer_id=cid,
                    source=docking_id,
                    energy_decomposition={
                        "potential": supraconformer.get_potential()
                    },
                )
                cid += 1


def calculate_min_atom_distance(supramolecule: spd.SupraMolecule) -> float:
    component_position_matrices = (
        i.get_position_matrix() for i in supramolecule.get_components()
    )

    min_distance = 1e24
    for pos_mat_pair in it.combinations(component_position_matrices, 2):
        pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
        min_distance = min([min_distance, min(pair_dists.flatten())])

    return min_distance


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
        laundry = Laundrette(
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

        # Do some analysis while you have the supramolecule.
        comps = list(min_energy_conformer.supramolecule.get_components())
        host_analysis = cgexplore.analysis.GeomMeasure()
        database.add_properties(
            key=name,
            property_dict={
                "centroid_distance": (
                    np.linalg.norm(
                        comps[0].get_centroid() - comps[1].get_centroid()
                    )
                ),
                "min_hg_distance": calculate_min_atom_distance(
                    min_energy_conformer.supramolecule
                ),
                "host_pore": host_analysis.calculate_min_distance(
                    host_structure
                ),
                "host_size": host_analysis.calculate_max_diameter(
                    host_structure
                ),
            },
        )

        min_energy_conformer.to_stk_molecule().write(opt_file)

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
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


def check_fit(
    chromosome: tuple[int, ...],
    num_beads: int,
    max_shell: int,
) -> bool:
    """Check if chromosome has an allowed topology."""
    if sum(chromosome) != num_beads:
        return False

    idx = chromosome[0]
    fit = True
    sum_g = np.sum(chromosome[:idx]).astype(int)
    while fit and sum_g < num_beads:
        check_chr = False
        for x in range(idx, sum_g + 1):
            if chromosome[x] != 0:
                check_chr = True
        if not check_chr and sum_g < num_beads:
            fit = False
        else:
            # additional requirement to avoid
            # null summation in idx if
            # chromosome[idx] == 0
            if chromosome[idx] != 0:
                idx += chromosome[idx]
            else:
                idx += 1
            sum_g = np.sum(chromosome[:idx])
    if fit:
        for c in chromosome:
            if c > max_shell:
                fit = False
                break
    return fit


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

    abead = cgexplore.molecular.CgBead(
        element_string="Ag",
        bead_class="a",
        bead_type="a",
        coordination=3,
    )
    cbead = cgexplore.molecular.CgBead(
        element_string="Fe",
        bead_class="c",
        bead_type="c",
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

    num_beads = 6
    compositions = [
        i
        for i in it.product(range(num_beads + 1), repeat=num_beads)
        if check_fit(i, num_beads=num_beads, max_shell=6)
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
    ]
    for host_id, host in enumerate(hosts):
        prefix = f"opt{host_id}"

        chromosome_gen = cgexplore.systems_optimisation.ChromosomeGenerator(
            prefix=prefix,
            present_beads=(abead, cbead, h1bead, h2bead, h3bead),
            vdw_bond_cutoff=2,
        )

        chromosome_gen.add_gene(
            iteration=(
                cgexplore.molecular.PrecursorGenerator(
                    composition=i,
                    present_beads=(
                        abead,
                        cbead,
                        cbead,
                        cbead,
                        cbead,
                        cbead,
                        cbead,
                    ),
                    binder_beads=(cbead,),
                    placer_beads=(abead,),
                    bead_distance=1.5,
                )
                for i in compositions
            ),
            gene_type="precursor",
        )

        # Add modifications to nonbonded interactions.
        nb_scale = (1, 5, 10, 15)
        nb_sizes = (0.7, 1, 1.5, 2)
        definer_dict = {}
        present_beads = (abead, cbead)
        for options in present_beads:
            type_string = f"{options.bead_type}"
            definer_dict[type_string] = ("nb", nb_scale, nb_sizes)

        # Host nonbonded terms are constant.
        definer_dict["m"] = ("nb", 10, 1)
        definer_dict["n"] = ("nb", 10, 1)
        definer_dict["b"] = ("nb", 10, 1)

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
                best_file = struct_output / (f"{best_name}_optc.mol")
                cgexplore.utilities.Pymol(
                    output_dir=best_dir,
                    file_prefix=f"{prefix}_{seed}_g{generation_id}_h{host_id}_best",
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
                ).visualise(
                    [best_file],
                    orient_atoms=None,
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

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
        (ax, ax1), (ax2, ax3) = axs

        plotted = 0
        for entry in database.get_entries():

            fitness = fitness_calculator(
                chromosome=chromosome_gen.select_chromosome(
                    tuple(entry.properties["chromosome"])
                ),
                chromosome_generator=chromosome_gen,
                database=database,
                structure_output=struct_output,
                calculation_output=calc_dir,
            )
            chromosome = chromosome_gen.select_chromosome(
                entry.properties["chromosome"]
            )
            (precursor,) = chromosome.get_precursors()

            first_number = precursor.composition[0]

            if "forcefield_dict" not in entry.properties:
                continue
            a_bead_s = entry.properties["forcefield_dict"]["s_dict"]["a"]
            a_bead_e = entry.properties["forcefield_dict"]["e_dict"]["a"]
            c_bead_s = entry.properties["forcefield_dict"]["s_dict"]["c"]
            c_bead_e = entry.properties["forcefield_dict"]["e_dict"]["c"]

            ax.scatter(
                entry.properties["centroid_distance"],
                entry.properties["energy_decomposition"]["potential"],
                c=fitness,
                edgecolor="none",
                s=70,
                marker="o",
                alpha=1.0,
                vmin=0,
                vmax=10,
                cmap="Blues",
            )
            ax1.scatter(
                entry.properties["centroid_distance"],
                entry.properties["energy_decomposition"]["potential"],
                c=first_number,
                edgecolor="none",
                s=70,
                marker="o",
                alpha=1.0,
                vmin=0,
                vmax=6,
                cmap="Blues",
            )
            ax2.scatter(
                a_bead_s,
                c_bead_s,
                c=fitness,
                edgecolor="none",
                s=70,
                marker="o",
                alpha=1.0,
                vmin=0,
                vmax=10,
                cmap="Blues",
            )
            ax3.scatter(
                a_bead_e,
                c_bead_e,
                c=fitness,
                edgecolor="none",
                s=70,
                marker="o",
                alpha=1.0,
                vmin=0,
                vmax=10,
                cmap="Blues",
            )
            plotted += 1

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("centroid distance", fontsize=16)
        ax.set_ylabel("energy", fontsize=16)
        ax.set_title(
            f"plotted: {plotted}, found: {len(found)}, "
            f"possible: {chromosome_gen.get_num_chromosomes()}",
            fontsize=16,
        )
        ax1.tick_params(axis="both", which="major", labelsize=16)
        ax1.set_xlabel("centroid distance", fontsize=16)
        ax1.set_ylabel("energy", fontsize=16)
        ax1.set_title(
            "by first num in composition",
            fontsize=16,
        )

        ax2.tick_params(axis="both", which="major", labelsize=16)
        ax2.set_xlabel("a bead sigma", fontsize=16)
        ax2.set_ylabel("c bead sigma", fontsize=16)

        ax3.tick_params(axis="both", which="major", labelsize=16)
        ax3.set_xlabel("a bead epsilon", fontsize=16)
        ax3.set_ylabel("c bead epsilon", fontsize=16)

        fig.tight_layout()
        fig.savefig(
            figure_dir / f"space_explored_{host_id}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()

    # Visualise all ligand topologies.
    chromosome_gen = cgexplore.systems_optimisation.ChromosomeGenerator(
        prefix=prefix,
        present_beads=(abead, cbead, h1bead, h2bead, h3bead),
        vdw_bond_cutoff=2,
    )

    chromosome_gen.add_gene(
        iteration=(
            cgexplore.molecular.PrecursorGenerator(
                composition=i,
                present_beads=(
                    abead,
                    cbead,
                    cbead,
                    cbead,
                    cbead,
                    cbead,
                    cbead,
                ),
                binder_beads=(cbead,),
                placer_beads=(abead,),
                bead_distance=1.5,
            )
            for i in compositions
        ),
        gene_type="precursor",
    )
    for chromosome in chromosome_gen.yield_chromosomes():
        (bb,) = chromosome.get_building_blocks()
        viz_file = struct_output / f"{chromosome.get_string()}_unopt.mol"
        bb.write(viz_file)
        cgexplore.utilities.Pymol(
            output_dir=all_dir,
            file_prefix=f"{prefix}_{chromosome.get_string()}",
            settings={
                "grid_mode": 0,
                "rayx": 1000,
                "rayy": 1000,
                "stick_rad": 0.7,
                "vdw": 0,
                "zoom_string": "custom",
                "zoom_scale": 1,
                "orient": False,
            },
            pymol_path=pymol_path(),
        ).visualise(
            [viz_file],
            orient_atoms=None,
        )


if __name__ == "__main__":
    main()
