#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for new stk EA classes.

Author: Andrew Tarzia

"""

import itertools as it
from collections import defaultdict
import numpy as np
import stk
from stk.utilities import dedupe
import logging

logger = logging.getLogger(__name__)


class RecordFitnessFunction(stk.ea.FitnessCalculator):
    def __init__(
        self,
        fitness_function,
    ):
        """
        Initialize a :class:`.FitnessFunction` instance.

        Parameters
        ----------
        fitness_function : :class:`callable`
            Takes a single parameter, the
            :class:`.MoleculeRecord` whose
            fitness needs to be calculated, and returns its
            fitness value.

        """

        self._fitness_function = fitness_function

    def get_fitness_value(self, record):
        return self._fitness_function(record)


class CgSerial:
    """
    A serial implementation of the default evolutionary algorithm.

    """

    def __init__(
        self,
        initial_population,
        fitness_calculator,
        mutator,
        crosser,
        generation_selector,
        mutation_selector,
        crossover_selector,
        fitness_normalizer,
        key_maker,
        logger,
    ):
        """
        Initialize an :class:`.Implementation` instance.

        """

        self._initial_population = initial_population
        self._fitness_calculator = fitness_calculator
        self._mutator = mutator
        self._crosser = crosser
        self._generation_selector = generation_selector
        self._mutation_selector = mutation_selector
        self._crossover_selector = crossover_selector
        self._fitness_normalizer = fitness_normalizer
        self._key_maker = key_maker
        self._logger = logger

    def _get_generations(self, num_generations, map_):
        def get_mutation_record(batch):
            return self._mutator.mutate(batch[0])

        def get_key(record):
            return self._key_maker.get_key(record.get_molecule())

        population = self._initial_population

        self._logger.info(
            "Calculating fitness values of initial population."
        )
        population = tuple(self._with_fitness_values(map_, population))

        population = tuple(
            self._fitness_normalizer.normalize(
                population=population,
            )
        )
        yield stk.ea.Generation(
            molecule_records=population,
            mutation_records=(),
            crossover_records=(),
        )

        for generation in range(1, num_generations):
            self._logger.info(f"Starting generation {generation}.")
            self._logger.info(f"Population size is {len(population)}.")

            self._logger.info("Doing crossovers.")
            crossover_records = tuple(
                self._get_crossover_records(population)
            )

            self._logger.info("Doing mutations.")
            mutation_records = tuple(
                record
                for record in map(
                    get_mutation_record,
                    self._mutation_selector.select(population),
                )
                if record is not None
            )

            self._logger.info("Calculating fitness values.")

            offspring = (
                record.get_molecule_record()
                for record in crossover_records
            )
            mutants = (
                record.get_molecule_record()
                for record in mutation_records
            )

            population = tuple(
                self._with_fitness_values(
                    map_=map_,
                    population=tuple(
                        dedupe(
                            iterable=it.chain(
                                population, offspring, mutants
                            ),
                            key=get_key,
                        )
                    ),
                )
            )
            population = tuple(
                self._fitness_normalizer.normalize(population)
            )

            population = tuple(
                molecule_record
                for molecule_record, in (
                    self._generation_selector.select(population)
                )
            )

            yield stk.ea.Generation(
                molecule_records=population,
                mutation_records=mutation_records,
                crossover_records=crossover_records,
            )

    def _get_crossover_records(self, population):
        for batch in self._crossover_selector.select(population):
            yield from self._crosser.cross(batch)

    def _with_fitness_values(self, map_, population):
        fitness_values = map_(
            self._fitness_calculator.get_fitness_value,
            population,
        )
        for record, fitness_value in zip(population, fitness_values):
            yield record.with_fitness_value(
                fitness_value=fitness_value,
                normalized=False,
            )

    def get_generations(self, num_generations):
        yield from self._get_generations(num_generations, map)


class CgEvolutionaryAlgorithm(stk.ea.EvolutionaryAlgorithm):
    def __init__(
        self,
        initial_population,
        fitness_calculator,
        mutator,
        crosser,
        generation_selector,
        mutation_selector,
        crossover_selector,
        fitness_normalizer=stk.ea.NullFitnessNormalizer(),
        key_maker=stk.Inchi(),
        num_processes=None,
    ):

        if num_processes == 1:
            self._implementation = CgSerial(
                initial_population=initial_population,
                fitness_calculator=fitness_calculator,
                mutator=mutator,
                crosser=crosser,
                generation_selector=generation_selector,
                mutation_selector=mutation_selector,
                crossover_selector=crossover_selector,
                fitness_normalizer=fitness_normalizer,
                key_maker=key_maker,
                logger=logger,
            )

        else:
            raise NotImplementedError("multiprocessing not implemented")


class RandomCgBead(stk.ea.MoleculeMutator):
    def __init__(
        self,
        bead_library,
        name="RandomCGBead",
        random_seed=None,
    ):
        """
        Initialize a :class:`.RandomBuildingBlock` instance.

        Parameters
        ----------
        bead_library : :class:`tuple` of :class:`.CGBeads`
            A group of beads which are used to replace a bead type in
            molecules being mutated.

        is_replaceable : :class:`callable`
            A function which takes a :class:`.BuildingBlock` and
            returns ``True`` or ``False``. This function is applied to
            every building block in the molecule being mutated.
            Building blocks which returned ``True`` are liable for
            substitution by one of the molecules in `building_blocks`.

        name : :class:`str`, optional
            A name to help identify the mutator instance.

        random_seed : :class:`int`, optional
            The random seed to use.

        """

        self._bead_library = bead_library
        self._name = name
        self._generator = np.random.RandomState(random_seed)

    def _is_in_bead_lib(self, atom, bead_lib):
        return atom.__class__.__name__ in set(
            i.element_string for i in bead_lib
        )

    def _contains_bead_type(self, building_block, element_string):
        return element_string in set(
            atom.__class__.__name__
            for atom in building_block.get_atoms()
        )

    def mutate(self, record):
        original_bbs = tuple(
            record.get_topology_graph().get_building_blocks()
        )
        # Choose the building block which undergoes mutation.
        replaceable_atoms = tuple(
            filter(
                lambda seq: self._is_in_bead_lib(
                    atom=seq,
                    bead_lib=self._bead_library,
                ),
                record.get_molecule().get_atoms(),
            )
        )

        replaced_bead_type = self._generator.choice(
            replaceable_atoms
        ).__class__.__name__

        new_bead_library = tuple(
            i
            for i in self._bead_library
            if i.element_string != replaced_bead_type
        )
        # Choose a replacement building block.
        replacement_bead = self._generator.choice(new_bead_library)

        # Figure out which original bbs contains the replaced bead.
        replacable_building_blocks = tuple(
            i
            for i in original_bbs
            if self._contains_bead_type(
                building_block=i,
                element_string=replaced_bead_type,
            )
        )

        replaced_building_block = self._generator.choice(
            replacable_building_blocks
        )

        map_ = {
            i.__class__.__name__: vars(stk)[i.__class__.__name__]
            for i in replaced_building_block.get_atoms()
        }
        map_[replaced_bead_type] = vars(stk)[
            replacement_bead.element_string
        ]
        new_atoms = []
        for i in replaced_building_block.get_atoms():
            if i.__class__.__name__ != replaced_bead_type:
                new_atoms.append(i)
            else:
                new_atoms.append(
                    map_[replaced_bead_type](
                        id=i.get_id(),
                        charge=i.get_charge(),
                    )
                )

        new_bonds = []
        for i in replaced_building_block.get_bonds():
            if replaced_bead_type not in (
                i.get_atom1().__class__.__name__,
                i.get_atom2().__class__.__name__,
            ):
                new_bonds.append(i)
            else:
                new_bonds.append(
                    stk.Bond(
                        atom1=map_[i.get_atom1().__class__.__name__](
                            id=i.get_atom1().get_id(),
                            charge=i.get_atom1().get_charge(),
                        ),
                        atom2=map_[i.get_atom2().__class__.__name__](
                            id=i.get_atom2().get_id(),
                            charge=i.get_atom2().get_charge(),
                        ),
                        order=i.get_order(),
                        periodicity=i.get_periodicity(),
                    )
                )

        new_fgs = []
        for fg in replaced_building_block.get_functional_groups():

            atom_map = {}
            for atom in fg.get_atoms():
                if atom.__class__.__name__ == replaced_bead_type:
                    new_atom = map_[replaced_bead_type](
                        id=atom.get_id(),
                        charge=atom.get_charge(),
                    )
                    atom_map[atom.get_id()] = new_atom
            new_fg = fg.with_atoms(atom_map)
            new_fgs.append(new_fg)

        # Define new building block based on that.
        replacement = stk.BuildingBlock.init(
            atoms=tuple(new_atoms),
            bonds=tuple(new_bonds),
            position_matrix=(
                replaced_building_block.get_position_matrix()
            ),
            functional_groups=new_fgs,
        )

        # Build the new ConstructedMolecule.
        graph = record.get_topology_graph().with_building_blocks(
            {
                replaced_building_block: replacement,
            }
        )
        return stk.ea.MutationRecord(
            molecule_record=stk.ea.MoleculeRecord(graph),
            mutator_name=self._name,
        )


class CgGeneticRecombination(stk.ea.MoleculeCrosser):
    def __init__(
        self,
        get_gene,
        name="CgGeneticRecombination",
    ):
        """
        Initialize a :class:`CgGeneticRecombination` instance.

        Parameters
        ----------
        get_gene : :class:`callable`
            A :class:`callable`, which takes a :class:`.BuildingBlock`
            object and returns its gene. To produce an offspring, one
            of the building blocks from each gene is picked.

        name : :class:`str`, optional
            A name to identify the crosser instance.

        """

        self._get_gene = get_gene
        self._name = name

    def cross(self, records):
        topology_graphs = (
            record.get_topology_graph() for record in records
        )
        for topology_graph, alleles in it.product(
            topology_graphs,
            self._get_alleles(records),
        ):

            def get_replacement(building_block):
                gene = self._get_gene(building_block)
                return next(
                    allele
                    for allele in alleles
                    if self._get_gene(allele) == gene
                )

            topology_graph = topology_graph.with_building_blocks(
                building_block_map={
                    building_block: get_replacement(building_block)
                    for building_block in topology_graph.get_building_blocks()
                },
            )
            yield stk.ea.CrossoverRecord(
                molecule_record=stk.ea.MoleculeRecord(
                    topology_graph=topology_graph,
                ),
                crosser_name=self._name,
            )

    def _get_alleles(self, records):
        """
        Yield every possible combination of alleles.

        """

        genes = defaultdict(list)
        topology_graphs = (
            record.get_topology_graph() for record in records
        )
        for topology_graph in topology_graphs:
            for allele in topology_graph.get_building_blocks():
                genes[self._get_gene(allele)].append(allele)
        return it.product(*genes.values())
