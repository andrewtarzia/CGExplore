import numpy as np

from .case_data import CaseData


def test_chromosome_generator(chromosome_generator: CaseData) -> None:
    """Test :class:`.ChromosomeGenerator`.

    Parameters:

        chromosome_generator:
            The chromosome generator.

    """
    chromo_it = chromosome_generator.chromosome_generator
    chromo_it.define_chromosomes()
    print(chromosome_generator.chromosome_generator)
    print(chromo_it.chromosome_map)
    print(chromo_it.chromosome_types)
    assert chromo_it.chromosome_map == chromosome_generator.known_map
    assert chromo_it.chromosome_types == chromosome_generator.known_types

    print(chromo_it.get_num_chromosomes())
    print(chromo_it.get_term_ids())
    print(chromo_it.get_topo_ids())
    print(chromo_it.get_va_ids())
    print(chromo_it.get_prec_ids())
    assert chromo_it.get_num_chromosomes() == chromosome_generator.num
    assert chromo_it.get_term_ids() == chromosome_generator.term_ids
    assert chromo_it.get_topo_ids() == chromosome_generator.topo_ids
    assert chromo_it.get_va_ids() == chromosome_generator.va_ids
    assert chromo_it.get_prec_ids() == chromosome_generator.prec_ids

    population1 = chromo_it.select_random_population(
        generator=np.random.default_rng(109),
        size=20,
    )
    population2 = chromo_it.select_random_population(
        generator=np.random.default_rng(9865),
        size=20,
    )
    print(population1)
    print(population2)
    assert len(population1) == len(population2)
    merged = population1 + population2
    for chromosome in merged:
        for i, gene in enumerate(chromosome.name):
            assert gene >= min(chromosome_generator.known_map[i].keys())
            assert gene <= max(chromosome_generator.known_map[i].keys())

    merged = chromo_it.dedupe_population(population1 + population2)
    assert len(merged) == chromosome_generator.merged_size
