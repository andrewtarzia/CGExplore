# Distributed under the terms of the MIT License.

"""Optimisation inputs module.

Author: Andrew Tarzia

"""
import itertools as it
import logging
from collections import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cgexplore._internal.forcefields.forcefield import ForceField
from cgexplore._internal.molecular.beads import CgBead
from cgexplore.utilities import AtomliteDatabase

from .utilities import (
    define_angle,
    define_bond,
    define_cosine_angle,
    define_nonbonded,
    define_torsion,
)

if TYPE_CHECKING:
    from cgexplore._internal.terms.angles import TargetAngle, TargetCosineAngle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Chromosome:
    """Define the genes and properties of a chromosome."""

    name: tuple[int, ...]
    present_beads: tuple[CgBead, ...]
    vdw_bond_cutoff: int
    prefix: str
    gene_dict: dict[int, tuple]
    definer_dict: dict[str, tuple]
    chromosomed_terms: dict[str, list[int]]

    def get_string(self) -> str:
        """Get chromosome name as string."""
        return "".join(str(i) for i in self.name)

    def get_topology_information(self) -> tuple:
        """Get the chromosomes topology information."""
        return next(
            self.gene_dict[i][1]
            for i in self.gene_dict
            if self.gene_dict[i][2] == "topology"
        )

    def get_building_blocks(self) -> tuple:
        """Get the chromosomes building blocks."""
        return tuple(
            self.gene_dict[i][1].get_building_block()
            for i in self.gene_dict
            if self.gene_dict[i][2] == "precursor"
        )

    def get_forcefield(self) -> ForceField:  # noqa: C901
        """Get chromosome forcefield."""
        changed_tuples = tuple(
            self.gene_dict[i][1]
            for i in self.gene_dict
            if self.gene_dict[i][2] == "term"
        )
        # Change to term: (new value, index in interaction list).
        changed_terms = {i[0]: (i[1], i[2]) for i in changed_tuples}

        bond_terms: list = []
        angle_terms: list[TargetAngle | TargetCosineAngle] = []
        torsion_terms: list = []
        nonbonded_terms: list = []
        for key_ in self.definer_dict:
            if key_ in changed_terms:
                ranged_term = self.definer_dict[key_]
                # Get terms based on all changes to this term in chromosome.
                term = [0] * len(ranged_term)
                term[0] = ranged_term[0]

                c_ids_part_of = self.chromosomed_terms[key_]

                # Get the id of chromosomes (key to map), that this term is in.
                ranged_indices = set()
                for c_id in c_ids_part_of:
                    ranged_id_definition = self.gene_dict[c_id][1]

                    gene_value = ranged_id_definition[1]
                    gene_index = ranged_id_definition[2]

                    term[gene_index] = gene_value
                    ranged_indices.add(gene_index)

                for i in range(len(ranged_term[1:])):
                    if i + 1 in ranged_indices:
                        continue
                    term[i + 1] = ranged_term[i + 1]
                term = tuple(term)  # type: ignore[assignment]

            else:
                term = self.definer_dict[key_]  # type: ignore[assignment]

            if term[0] == "bond":
                bond_terms.append(
                    define_bond(
                        interaction_key=key_,
                        interaction_list=term,
                        present_beads=self.present_beads,
                    )
                )

            elif term[0] == "angle":
                angle_terms.append(
                    define_angle(
                        interaction_key=key_,
                        interaction_list=term,
                        present_beads=self.present_beads,
                    )
                )

            elif term[0] == "cosine":
                angle_terms.append(
                    define_cosine_angle(
                        interaction_key=key_,
                        interaction_list=term,
                        present_beads=self.present_beads,
                    )
                )

            elif term[0] == "tors":
                torsion_terms.append(
                    define_torsion(
                        interaction_key=key_,
                        interaction_list=term,
                        present_beads=self.present_beads,
                    )
                )

            elif term[0] == "nb":
                nonbonded_terms.append(
                    define_nonbonded(
                        interaction_key=key_,
                        interaction_list=term,
                        present_beads=self.present_beads,
                    )
                )

        return ForceField(
            identifier=self.get_string(),
            prefix=self.prefix,
            present_beads=self.present_beads,
            bond_targets=tuple(bond_terms),
            angle_targets=tuple(angle_terms),
            torsion_targets=tuple(torsion_terms),
            nonbonded_targets=tuple(nonbonded_terms),
            vdw_bond_cutoff=self.vdw_bond_cutoff,
        )

    def __str__(self) -> str:
        """Return a string representation of the Chromosome."""
        return f"{self.__class__.__name__}({self.get_string()})"

    def __repr__(self) -> str:
        """Return a string representation of the Chromosome."""
        return str(self)


@dataclass
class ChromosomeGenerator:
    """Hold all information for chromosome iteration."""

    present_beads: tuple[CgBead, ...]
    vdw_bond_cutoff: int
    prefix: str
    chromosome_map: dict[int, dict] = field(default_factory=dict)
    chromosome_types: dict[int, str] = field(default_factory=dict)
    chromosomed_terms: dict[str, list[int]] = field(default_factory=dict)
    definer_dict: dict[str, tuple] = field(default_factory=dict)

    def get_num_chromosomes(self) -> int:
        """Return the number of chromosomes."""
        try:
            return len(self.chromosomes)
        except AttributeError:
            return 0

    def add_gene(self, iteration: abc.Iterable, gene_type: str) -> None:
        """Add a gene to the chromosome generator, to be iterated over.

        Parameters:
            iteration:
                The range of values to use in the chromosome.
                For term: use the `add_forcefield_dict`
                For topology:
                For precursor:
                For forcefield:

            gene_type:
                A string defining the gene type.
                Can be `term`, `topology`, `precursor`, `forcefield`.

        """
        if gene_type not in ("term", "topology", "precursor", "forcefield"):
            msg = "gene_type not `term`, `topology`, `precursor`, `forcefield`"
            raise RuntimeError(msg)

        known_types = set(self.chromosome_types.values())
        if "term" in known_types and gene_type == "forcefield":
            msg = "cannot add `forcefield` and `term` in the same chromosome."
            raise RuntimeError(msg)
        if "forcefield" in known_types and gene_type == "term":
            msg = "cannot add `forcefield` and `term` in the same chromosome."
            raise RuntimeError(msg)

        chromo_index = len(self.chromosome_map)
        self.chromosome_types[chromo_index] = gene_type
        self.chromosome_map[chromo_index] = dict(enumerate(iteration))

        if gene_type == "term":
            term_key = next(
                i[0] for i in self.chromosome_map[chromo_index].values()
            )
            if term_key not in self.chromosomed_terms:
                self.chromosomed_terms[term_key] = []
            self.chromosomed_terms[term_key].append(chromo_index)

    def add_forcefield_dict(self, definer_dict: dict[str, tuple]) -> None:
        """Add genes based on a forcefield dictionary.

        Parameters:
            definer_dict:
                The forcefield dictionary.
                Format:

        """
        self.definer_dict = definer_dict
        for key in definer_dict:
            definer = definer_dict[key]
            for comp_id, comp in enumerate(definer[1:]):
                if not isinstance(comp, float | int | str):
                    self.add_gene(
                        iteration=tuple((key, i, comp_id + 1) for i in comp),
                        gene_type="term",
                    )

    def yield_chromosomes(self) -> abc.Iterable[Chromosome]:
        """Yield chromosomes."""
        chromosome_ids = sorted(self.chromosome_map.keys())

        iteration = it.product(
            *(self.chromosome_map[i] for i in chromosome_ids)
        )

        known_types = set(self.chromosome_types.values())
        for chromosome_name in iteration:
            gene_dict = {}
            for gene_id, gene in enumerate(chromosome_name):
                gene_value = self.chromosome_map[gene_id][gene]
                gene_type = self.chromosome_types[gene_id]
                gene_dict[gene_id] = (gene, gene_value, gene_type)

            if "forcefield" in known_types:
                # In this case, the definer dict changes per chromosome!
                ff_id = next(
                    i
                    for i in self.chromosome_types
                    if self.chromosome_types[i] == "forcefield"
                )
                definer_dict = gene_dict[ff_id][1]
            else:
                definer_dict = self.definer_dict

            yield Chromosome(
                name=chromosome_name,
                prefix=self.prefix,
                present_beads=self.present_beads,
                vdw_bond_cutoff=self.vdw_bond_cutoff,
                gene_dict=gene_dict,
                definer_dict=definer_dict,
                chromosomed_terms=self.chromosomed_terms,
            )

    def define_chromosomes(self) -> None:
        """Get all chromosomes that have been defined.

        If the chromosome space is large, this can be super expensive!
        """
        all_chromosomes = {}
        for chromosome in self.yield_chromosomes():
            all_chromosomes[chromosome.name] = chromosome
        self.chromosomes = all_chromosomes
        logging.info(f"there are {len(self.chromosomes)} chromosomes")

    def get_term_ids(self) -> tuple[int, ...]:
        """Get chromosome indices associated with terms."""
        return tuple(
            i
            for i in self.chromosome_types
            if self.chromosome_types[i] == "term"
        )

    def get_topo_ids(self) -> tuple[int, ...]:
        """Get chromosome indices associated with topology."""
        return tuple(
            i
            for i in self.chromosome_types
            if self.chromosome_types[i] == "topology"
        )

    def get_prec_ids(self) -> tuple[int, ...]:
        """Get chromosome indices associated with precursors."""
        return tuple(
            i
            for i in self.chromosome_types
            if self.chromosome_types[i] == "precursor"
        )

    def select_chromosome(self, chromosome: tuple[int, ...]) -> Chromosome:
        """Get chromosome."""
        raise SystemExit("handle chromosomes usage 1")
        return self.chromosomes[chromosome]

    def select_random_population(
        self,
        generator: np.random.Generator,
        size: int,
    ) -> list[Chromosome]:
        """Select `size` instances from population."""
        selected = []

        known_types = set(self.chromosome_types.values())
        for _ in range(size):
            gene_dict = {}
            chromosome_name = [0 for i in range(len(self.chromosome_map))]
            for gene_id in self.chromosome_map:
                chromosome_options = range(len(self.chromosome_map[gene_id]))
                gene = generator.choice(chromosome_options)
                gene_value = self.chromosome_map[gene_id][gene]
                gene_type = self.chromosome_types[gene_id]
                gene_dict[gene_id] = (gene, gene_value, gene_type)
                chromosome_name[gene_id] = gene

            if "forcefield" in known_types:
                # In this case, the definer dict changes per chromosome!
                ff_id = next(
                    i
                    for i in self.chromosome_types
                    if self.chromosome_types[i] == "forcefield"
                )
                definer_dict = gene_dict[ff_id][1]
            else:
                definer_dict = self.definer_dict
            selected.append(
                Chromosome(
                    name=tuple(chromosome_name),
                    prefix=self.prefix,
                    present_beads=self.present_beads,
                    vdw_bond_cutoff=self.vdw_bond_cutoff,
                    gene_dict=gene_dict,
                    definer_dict=definer_dict,
                    chromosomed_terms=self.chromosomed_terms,
                )
            )

        return selected

    def dedupe_population(
        self,
        list_of_chromosomes: list[Chromosome],
    ) -> list[Chromosome]:
        """Deduplicate the list of chromosomes."""
        tuples = {i.name for i in list_of_chromosomes}
        return [self.select_chromosome(i) for i in tuples]

    def select_similar_chromosome(
        self,
        chromosome: Chromosome,
        free_gene_id: int,
    ) -> list[Chromosome]:
        """Select chromosomes where only one gene is allowed to change."""
        raise SystemExit("handle chromosomes usage 2")
        filter_range = [
            i for i in sorted(self.chromosome_map.keys()) if i != free_gene_id
        ]
        filtered_chromosomes = [
            i
            for i in self.chromosomes
            if all(i[k] == chromosome.name[k] for k in filter_range)
        ]
        return [self.select_chromosome(tuple(i)) for i in filtered_chromosomes]

    def mutate_population(  # noqa: PLR0913
        self,
        list_of_chromosomes: list[Chromosome],
        generator: np.random.Generator,
        gene_range: tuple[int, ...],
        selection: str,
        num_to_select: int,
        database: AtomliteDatabase,
    ) -> list[Chromosome]:
        """Mutate a list of chromosomes in the gene range only.

        Available selections for which chromosomes to mutate:
            random - uses generator.choice()
            roulette - adds weight to generator.choice() based on
                fitness/sum(fitness)

        """
        # Define which genes we are mutating.
        filter_range = [
            i
            for i in sorted(self.chromosome_map.keys())
            if i not in gene_range
        ]

        # Select chromosomes to mutate.
        if selection == "random":
            selected = generator.choice(
                np.asarray(list_of_chromosomes),
                size=num_to_select,
            )
        elif selection == "roulette":
            fitness_values: list[float | int] = [
                database.get_entry(f"{i.prefix}_{i.get_string()}").properties[  # type: ignore[misc]
                    "fitness"
                ]
                for i in list_of_chromosomes
            ]
            weights = [i / sum(fitness_values) for i in fitness_values]
            selected = generator.choice(
                np.asarray(list_of_chromosomes),
                size=num_to_select,
                p=weights,
            )

        else:
            msg = f"{selection} is not defined."
            raise RuntimeError(msg)

        raise SystemExit("handle chromosomes usage 3")
        mutated = []
        for chromosome in selected:
            # Filter all chromosomes based on matching to selected chromosome
            # in gene region.
            filtered_chromosomes = [
                i
                for i in self.chromosomes
                if all(i[k] == chromosome.name[k] for k in filter_range)
            ]
            # Add the mutated chromosome selected from the filter chromosomes.
            mutated.append(
                self.select_chromosome(
                    tuple(generator.choice(filtered_chromosomes))
                )
            )

        return mutated

    def crossover_population(  # noqa: PLR0913
        self,
        list_of_chromosomes: list[Chromosome],
        generator: np.random.Generator,
        selection: str,
        num_to_select: int,
        database: AtomliteDatabase,
    ) -> list[Chromosome]:
        # Select chromosomes to cross.
        if selection == "random":
            selected = generator.choice(
                np.asarray(list_of_chromosomes),
                size=(num_to_select, 2),
            )
        elif selection == "roulette":
            fitness_values: list[float | int] = [
                database.get_entry(f"{i.prefix}_{i.get_string()}").properties[  # type: ignore[misc]
                    "fitness"
                ]
                for i in list_of_chromosomes
            ]
            weights = [i / sum(fitness_values) for i in fitness_values]
            selected = generator.choice(
                np.asarray(list_of_chromosomes),
                size=(num_to_select, 2),
                p=weights,
            )
        else:
            msg = f"{selection} is not defined."
            raise RuntimeError(msg)

        crossed = []
        for chromosome1, chromosome2 in selected:
            # Randomly select the genes to cross.
            nums_to_select_from = range(len(chromosome1.name))
            num_to_cross = generator.choice(nums_to_select_from, size=1)
            genes_to_cross = set(
                generator.choice(nums_to_select_from, size=num_to_cross[0])
            )

            # Cross them.
            new_chromosome1 = tuple(
                val if i not in genes_to_cross else chromosome2.name[i]
                for i, val in enumerate(chromosome1.name)
            )
            new_chromosome2 = tuple(
                val if i not in genes_to_cross else chromosome1.name[i]
                for i, val in enumerate(chromosome2.name)
            )

            # Append the new chromosomes.
            crossed.append(self.select_chromosome(new_chromosome1))
            crossed.append(self.select_chromosome(new_chromosome2))

        return crossed

    def __str__(self) -> str:
        """Return a string representation of the ChromosomeGenerator."""
        _num_chromosomes = self.get_num_chromosomes()
        _num_genes = len(self.chromosome_map)
        return f"{self.__class__.__name__}(" f"num_genes={_num_genes})"

    def __repr__(self) -> str:
        """Return a string representation of the Chromosome."""
        return str(self)
