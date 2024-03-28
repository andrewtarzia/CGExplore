# Distributed under the terms of the MIT License.

"""Optimisation Generation module."""

import logging
from collections import abc
from dataclasses import dataclass

import pathos

from .calculators import FitnessCalculator, StructureCalculator
from .inputs import Chromosome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass(frozen=True, slots=True)
class Generation:
    """Define the chromosomes in a single generation."""

    chromosomes: list[Chromosome]
    fitness_calculator: FitnessCalculator
    structure_calculator: StructureCalculator
    num_processes: int = 1

    def get_generation_size(self) -> int:
        """Get the number of chromosomes in a generation."""
        return len(self.chromosomes)

    def calculate_fitness_values(self) -> list[float]:
        """Calculate the fitness of all chromosomes."""
        length = len(self.chromosomes)
        logging.info(f"calculating fitness values of {length} systems...")
        if self.num_processes > 1:
            with pathos.pools.ProcessPool(self.num_processes) as pool:
                return pool.map(
                    self.fitness_calculator.calculate, self.chromosomes
                )

        return [
            self.fitness_calculator.calculate(chromosome=i)
            for i in self.chromosomes
        ]

    def run_structures(self) -> None:
        """Run the production and analyse of all chromosomes."""
        length = len(self.chromosomes)
        logging.info(f"generating structures of {length} systems...")
        if self.num_processes > 1:
            with pathos.pools.ProcessPool(self.num_processes) as pool:
                pool.map(self.structure_calculator.calculate, self.chromosomes)
        else:
            for chromosome in self.chromosomes:
                self.structure_calculator.calculate(chromosome=chromosome)

    def select_best(self, selection_size: int) -> abc.Iterable[Chromosome]:
        """Select the best in the generation by fitness."""
        temp = [
            (
                i,
                self.fitness_calculator.calculate(chromosome=i),
            )
            for i in self.chromosomes
        ]
        best_indices = tuple(
            sorted(range(len(temp)), key=lambda i: temp[i][1], reverse=True)
        )[:selection_size]

        return [self.chromosomes[i] for i in best_indices]

    def select_worst(self, selection_size: int) -> abc.Iterable[Chromosome]:
        """Select the worst in the generation by fitness."""
        temp = [
            (
                i,
                self.fitness_calculator.calculate(chromosome=i),
            )
            for i in self.chromosomes
        ]
        best_indices = tuple(
            sorted(range(len(temp)), key=lambda i: temp[i][1], reverse=False)
        )[:selection_size]

        return [self.chromosomes[i] for i in best_indices]

    def select_elite(
        self,
        proportion_threshold: float,
    ) -> abc.Iterable[Chromosome]:
        """Select the elite in the generation by fitness."""
        num_in_generation = self.get_generation_size()
        proportion_to_select = round(num_in_generation * proportion_threshold)
        return self.select_best(proportion_to_select)

    def select_all(self) -> abc.Iterable[Chromosome]:
        """Select all in the generation."""
        return list(self.chromosomes)

    def calculate_elite_fitness(
        self,
        proportion_threshold: float,
    ) -> float:
        """Select the elite in the generation by fitness."""
        elite = self.select_elite(proportion_threshold)
        return min(
            self.fitness_calculator.calculate(chromosome=i) for i in elite
        )
