import pathlib

import numpy as np

import cgexplore as cgx

from .case_data import CaseData


def fit_calc(
    chromosome: cgx.systems_optimisation.Chromosome,
    chromosome_generator: cgx.systems_optimisation.ChromosomeGenerator,  # noqa: ARG001
    database_path: pathlib.Path,  # noqa: ARG001
    calculation_output: pathlib.Path,  # noqa: ARG001
    structure_output: pathlib.Path,  # noqa: ARG001
    options: dict,  # noqa: ARG001
) -> float:
    return sum([int(i) for i in chromosome.name])


def str_calc(
    chromosome: cgx.systems_optimisation.Chromosome,
    database_path: pathlib.Path,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    options: dict,
) -> None:
    pass


def test_generation(chromosome_generator: CaseData) -> None:
    """Test :class:`.ChromosomeGenerator`.

    Parameters:

        chromosome_generator:
            The chromosome generator.

    """
    output_dir = pathlib.Path(__file__).resolve().parent / "output"
    chromo_it = chromosome_generator.chromosome_generator
    chromo_it.define_chromosomes()
    size = 10

    population1 = chromo_it.select_random_population(
        generator=np.random.default_rng(109),
        size=size,
    )
    generation = cgx.systems_optimisation.Generation(
        chromosomes=population1,
        num_processes=chromosome_generator.np,
        fitness_calculator=cgx.systems_optimisation.FitnessCalculator(
            fitness_function=fit_calc,
            chromosome_generator=chromo_it,
            structure_output=output_dir,
            calculation_output=output_dir,
            database_path=output_dir / "t.db",
            options={},
        ),
        structure_calculator=cgx.systems_optimisation.StructureCalculator(
            structure_function=str_calc,
            structure_output=output_dir,
            calculation_output=output_dir,
            database_path=output_dir / "t.db",
            options={},
        ),
    )

    assert generation.get_generation_size() == size
    generation.run_structures()
    print(generation.calculate_fitness_values())
    print(generation.select_best(2))
    print(generation.select_worst(2))
    print(generation.calculate_elite_fitness(proportion_threshold=0.1))
    assert (
        generation.calculate_fitness_values()
        == chromosome_generator.pop1_fitness
    )
    assert (
        tuple(i.get_string() for i in generation.select_best(2))
        == chromosome_generator.pop1_best
    )
    assert (
        tuple(i.get_string() for i in generation.select_worst(2))
        == chromosome_generator.pop1_worst
    )
    assert (
        generation.calculate_elite_fitness(proportion_threshold=0.1)
        == chromosome_generator.pop1_elite_fitness
    )
