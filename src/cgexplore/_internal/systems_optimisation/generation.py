# Distributed under the terms of the MIT License.

"""Optimisation Generation module."""
import logging
import pathlib
from collections import abc
from dataclasses import dataclass, field

from cgexplore._internal.utilities.databases import AtomliteDatabase

from .inputs import Chromosome, ChromosomeGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Generation:
    """Define the chromosomes in a single generation."""

    chromosomes: list[Chromosome]
    chromosome_generator: ChromosomeGenerator
    fitness_calculator: abc.Callable
    structure_calculator: abc.Callable
    structure_output: pathlib.Path
    calculation_output: pathlib.Path
    database: AtomliteDatabase
    options: dict = field(default_factory=dict)

    def get_generation_size(self) -> int:
        """Get the number of chromosomes in a generation."""
        return len(self.chromosomes)

    def calculate_fitness_values(self) -> list[float]:
        """Calculate the fitness of all chromosomes."""
        return [
            self.fitness_calculator(
                chromosome=i,
                chromosome_generator=self.chromosome_generator,
                database=self.database,
                calculation_output=self.calculation_output,
                structure_output=self.structure_output,
                options=self.options,
            )
            for i in self.chromosomes
        ]

    def run_structures(self) -> None:
        """Run the production and analyse of all chromosomes."""
        length = len(self.chromosomes)
        for i, chromosome in enumerate(self.chromosomes):
            logging.info(f"building {chromosome} ({i+1} of {length})")
            self.structure_calculator(
                chromosome=chromosome,
                database=self.database,
                calculation_output=self.calculation_output,
                structure_output=self.structure_output,
                options=self.options,
            )

    def select_best(self, selection_size: int) -> abc.Iterable[Chromosome]:
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
                    options=self.options,
                ),
            )
            for i in self.chromosomes
        ]
        best_indices = tuple(
            sorted(range(len(temp)), key=lambda i: temp[i][1], reverse=True)
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
