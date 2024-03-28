# Distributed under the terms of the MIT License.

"""Optimisation Generation module."""

import logging
import pathlib
from collections import abc
from dataclasses import dataclass, field

from .inputs import Chromosome, ChromosomeGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass(frozen=True, slots=True)
class StructureCalculator:
    structure_function: abc.Callable
    structure_output: pathlib.Path
    calculation_output: pathlib.Path
    database_path: pathlib.Path
    options: dict = field(default_factory=dict)

    def calculate(self, chromosome: Chromosome) -> None:
        """Run the production and analyse a chromosome."""
        self.structure_function(
            chromosome=chromosome,
            database_path=self.database_path,
            calculation_output=self.calculation_output,
            structure_output=self.structure_output,
            options=self.options,
        )


@dataclass(frozen=True, slots=True)
class FitnessCalculator:
    fitness_function: abc.Callable
    chromosome_generator: ChromosomeGenerator
    structure_output: pathlib.Path
    calculation_output: pathlib.Path
    database_path: pathlib.Path
    options: dict = field(default_factory=dict)

    def calculate(self, chromosome: Chromosome) -> float:
        """Calculate the fitness of a chromosome."""
        return self.fitness_function(
            chromosome=chromosome,
            chromosome_generator=self.chromosome_generator,
            database_path=self.database_path,
            calculation_output=self.calculation_output,
            structure_output=self.structure_output,
            options=self.options,
        )
