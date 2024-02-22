import logging
import os
import pathlib
from collections import abc, defaultdict
from dataclasses import dataclass

import cgexplore
import matplotlib.pyplot as plt
import numpy as np
import math
import stk
import openmm
import itertools as it


def colours():
    """Colours map to topologies."""
    return {
        "2P3": "#1f77b4",
        "4P6": "#ff7f0e",
        "4P62": "#2ca02c",
        "6P9": "#d62728",
        "8P12": "#17becf",
    }


def analyse_ligand(
    database,
    name,
    file_name,
    output_dir,
    forcefield,
    chromosome,
):
    entry = database.get_entry(key=name)
    molecule = database.get_molecule(key=name)
    properties = entry.properties

    topology_str, _ = chromosome.get_topology_information()

    database.add_properties(
        key=name,
        property_dict={
            "prefix": file_name.split("_")[0],
            "repeating_unit": topology_str,
        },
    )

    if "strain_energy" not in properties:
        energy_decomp = {}
        for component in properties["energy_decomposition"]:
            component_tup = properties["energy_decomposition"][component]
            if component == "total energy":
                energy_decomp[f"{component}_{component_tup[1]}"] = float(
                    component_tup[0]
                )
            else:
                just_name = component.split("'")[1]
                key = f"{just_name}_{component_tup[1]}"
                value = float(component_tup[0])
                if key in energy_decomp:
                    energy_decomp[key] += value
                else:
                    energy_decomp[key] = value
        fin_energy = energy_decomp["total energy_kJ/mol"]
        try:
            assert (
                sum(
                    energy_decomp[i]
                    for i in energy_decomp
                    if "total energy" not in i
                )
                == fin_energy
            )
        except AssertionError as ex:
            ex.add_note(
                "energy decompisition does not sum to total energy for"
                f" {name}: {energy_decomp}"
            )
            raise
        res_dict = {
            "strain_energy": fin_energy,
        }
        database.add_properties(key=name, property_dict=res_dict)

    if "dihedral_data" not in properties:
        # Always want to extract target torions if present.
        g_measure = cgexplore.analysis.GeomMeasure()
        bond_data = g_measure.calculate_bonds(molecule)
        bond_data = {"_".join(i): bond_data[i] for i in bond_data}
        angle_data = g_measure.calculate_angles(molecule)
        angle_data = {"_".join(i): angle_data[i] for i in angle_data}
        dihedral_data = g_measure.calculate_torsions(
            molecule=molecule,
            absolute=True,
            as_search_string=True,
        )
        database.add_properties(
            key=name,
            property_dict={
                "bond_data": bond_data,
                "angle_data": angle_data,
                "dihedral_data": dihedral_data,
            },
        )

    if "forcefield_dict" not in properties:
        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = forcefield.get_targets()
        k_dict = {}
        v_dict = {}

        for bt in ff_targets["bonds"]:
            cp = (bt.type1, bt.type2)
            k_dict["_".join(cp)] = bt.bond_k.value_in_unit(
                openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.nanometer**2
            )
            v_dict["_".join(cp)] = bt.bond_r.value_in_unit(
                openmm.unit.angstrom
            )

        for at in ff_targets["angles"]:
            cp = (at.type1, at.type2, at.type3)
            try:
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(
                    openmm.unit.degrees
                )
            except TypeError:
                # Handle different angle types.
                k_dict["_".join(cp)] = at.angle_k.value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole
                )
                v_dict["_".join(cp)] = at.angle.value_in_unit(
                    openmm.unit.degrees
                )

        for at in ff_targets["torsions"]:
            cp = at.search_string
            k_dict["_".join(cp)] = at.torsion_k.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            v_dict["_".join(cp)] = at.phi0.value_in_unit(openmm.unit.degrees)

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "v_dict": v_dict,
            "k_dict": k_dict,
        }

        database.add_properties(
            key=name,
            property_dict={"forcefield_dict": forcefield_dict},
        )


def fitness_calculator(
    chromosome,
    chromosome_generator,
    database,
    calculation_output,
    structure_output,
):
    conversion_file = calculation_output / "file_names.txt"
    known_conversions = {}
    if conversion_file.exists():
        with open(conversion_file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(",")
                known_conversions[line[0]] = line[1]

    name = f"{chromosome.prefix}_{chromosome.get_string()}"
    file_name = known_conversions[name]
    entry = database.get_entry(name)
    energy = entry.properties["strain_energy"]

    if energy < 1e6:
        fitness = 100
    else:
        fitness = min((100, 1 / energy))

    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


def optimise_ligand(
    molecule: stk.Molecule,
    name: str,
    file_name: str,
    output_dir: pathlib.Path,
    forcefield: cgexplore.forcefields.ForceField,
    platform: str | None,
    database: cgexplore.utilities.AtomliteDatabase,
) -> stk.Molecule:
    """Optimise a building block.

    Keywords:

        molecule:
            The molecule to optimise.

        name:
            Name to use for naming output files. E.g. produces a file
            `{name}_opted1.mol` in `output_dir`.

        output_dir:
            Directory to save outputs of optimisation process.

        forcefield:
            Define the forces used in the molecule.

        platform:
            Which platform to use with OpenMM optimisation. Options are
            `CPU` or `CUDA`. More are available but may not work well
            out of the box.

    Returns:
        An stk molecule.

    """
    opt1_mol_file = pathlib.Path(output_dir) / f"{file_name}_opted1.mol"

    if opt1_mol_file.exists():
        return molecule.with_structure_from_file(str(opt1_mol_file))
    else:
        logging.info(f"optimising {name}, no max_iterations")
        assigned_system = forcefield.assign_terms(
            molecule=molecule,
            name=file_name,
            output_dir=output_dir,
        )
        minimum_energy_conformer = cgexplore.utilities.run_optimisation(
            assigned_system=assigned_system,
            name=name,
            file_suffix="opt1",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        minimum_energy = minimum_energy_conformer.energy_decomposition[
            "total energy"
        ][0]
        if math.isnan(minimum_energy):
            minimum_energy = 1e24
            minimum_energy_conformer = None
        else:
            molecule = minimum_energy_conformer.molecule

        # Run optimisations of series of conformers with shifted out
        # building blocks.
        for test_molecule in cgexplore.utilities.yield_shifted_models(
            molecule,
            forcefield,
            kicks=(1, 2, 3, 4),
        ):
            try:
                conformer = cgexplore.utilities.run_optimisation(
                    assigned_system=cgexplore.forcefields.AssignedSystem(
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
                    # max_iterations=50,
                    platform=platform,
                )
                energy = conformer.energy_decomposition["total energy"][0]
                if math.isnan(energy):
                    continue
                if energy < minimum_energy:
                    minimum_energy = energy
                    minimum_energy_conformer = cgexplore.molecular.Conformer(
                        molecule=conformer.molecule,
                        energy_decomposition=conformer.energy_decomposition,
                        source="kick_opt",
                    )
            except openmm.OpenMMException as error:
                if "Particle coordinate is NaN. " not in str(error):
                    raise error

        if minimum_energy_conformer is None:
            print(minimum_energy, minimum_energy_conformer)
            raise SystemExit

        minimum_energy_conformer.molecule = (
            minimum_energy_conformer.molecule.with_centroid(
                np.array((0, 0, 0))
            )
        )
        minimum_energy_conformer.molecule.write(opt1_mol_file)

        database.add_molecule(
            molecule=minimum_energy_conformer.molecule,
            key=name,
        )
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": (
                    minimum_energy_conformer.energy_decomposition
                ),
                "optimised": True,
                "source": minimum_energy_conformer.source,
                "name": name,
                "file_name": file_name,
            },
        )

    return minimum_energy_conformer.molecule


def structure_calculator(
    chromosome,
    database,
    calculation_output,
    structure_output,
):
    conversion_file = calculation_output / "file_names.txt"
    known_conversions = {}
    if conversion_file.exists():
        with open(conversion_file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(",")
                known_conversions[line[0]] = line[1]

    # Build structure.
    topology_str, topology_fun = chromosome.get_topology_information()
    building_blocks = chromosome.get_building_blocks()
    ligand = topology_fun.construct(building_blocks)

    name = f"{chromosome.prefix}_{chromosome.get_string()}"
    num_ligands_built = len(list(structure_output.glob("*_optl.mol")))
    if name in known_conversions:
        file_name = known_conversions[name]
    else:
        file_name = f"{chromosome.prefix}_{num_ligands_built}"
        known_conversions[name] = file_name

    with open(conversion_file, "a") as f:
        f.write(f"{name},{file_name}\n")

    # Select forcefield by chromosome.
    forcefield = chromosome.get_forcefield()

    # Optimise with some procedure.
    opt_file = str(structure_output / f"{file_name}_optl.mol")
    building_block = optimise_ligand(
        molecule=ligand,
        name=name,
        file_name=file_name,
        output_dir=calculation_output,
        forcefield=forcefield,
        database=database,
        platform=None,
    )
    building_block.write(opt_file)

    # Analyse cage.
    analyse_ligand(
        name=name,
        file_name=file_name,
        output_dir=calculation_output,
        forcefield=forcefield,
        database=database,
        chromosome=chromosome,
    )


def progress_plot(generations, output, num_generations):
    fig, ax = plt.subplots(figsize=(8, 5))
    fitnesses = []
    for generation in generations:
        fitnesses.append(generation.calculate_fitness_values())

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


@dataclass
class PolymerTopology:
    repeating_unit: str

    def construct(
        self,
        building_blocks: abc.Iterable[stk.BuildingBlock],
    ) -> stk.ConstructedMolecule:
        """Construct the molecule."""
        return stk.ConstructedMolecule(
            stk.polymer.Linear(
                building_blocks=building_blocks,
                repeating_unit=self.repeating_unit,
                num_repeating_units=1,
            )
        )


def main():
    raise SystemExit("this does not work yet!")
    wd = pathlib.Path("/home/atarzia/workingspace/cage_optimisation_tests")
    struct_output = wd / "ligand_structures"
    cgexplore.utilities.check_directory(struct_output)
    calc_dir = wd / "ligand_calculations"
    cgexplore.utilities.check_directory(calc_dir)
    data_dir = wd / "ligand_data"
    cgexplore.utilities.check_directory(data_dir)
    figure_dir = wd / "ligand_figures"
    cgexplore.utilities.check_directory(figure_dir)
    best_dir = figure_dir / "best"
    cgexplore.utilities.check_directory(best_dir)

    prefix = "opt"
    database = cgexplore.utilities.AtomliteDatabase(data_dir / "test.db")

    abead = cgexplore.molecular.CgBead(
        element_string="Ag",
        bead_class="a",
        bead_type="a",
        coordination=3,
    )
    bbead = cgexplore.molecular.CgBead(
        element_string="Ba",
        bead_class="b",
        bead_type="b",
        coordination=2,
    )
    cbead = cgexplore.molecular.CgBead(
        element_string="C",
        bead_class="c",
        bead_type="c",
        coordination=2,
    )
    dbead = cgexplore.molecular.CgBead(
        element_string="O",
        bead_class="o",
        bead_type="o",
        coordination=2,
    )
    ebead = cgexplore.molecular.CgBead(
        element_string="N",
        bead_class="n",
        bead_type="n",
        coordination=2,
    )

    chromosome_gen = cgexplore.systems_optimisation.ChromosomeGenerator(
        prefix=prefix,
        present_beads=(abead, bbead, cbead, dbead, ebead),
        vdw_bond_cutoff=2,
    )
    chromosome_gen.add_gene(
        iteration=(
            ("AC", PolymerTopology("AC")),
            ("ABC", PolymerTopology("ABC")),
            ("ABBC", PolymerTopology("ABBC")),
        ),
        gene_type="topology",
    )

    # A precursors.
    chromosome_gen.add_gene(
        iteration=(
            cgexplore.molecular.LinearPrecursor(
                composition=(1,),
                present_beads=(abead, ebead),
                binder_beads=(abead,),
                placer_beads=(ebead,),
            ),
            cgexplore.molecular.TrianglePrecursor(
                present_beads=(abead, bbead, ebead),
                binder_beads=(abead,),
                placer_beads=(bbead, ebead),
            ),
            cgexplore.molecular.SquarePrecursor(
                present_beads=(abead, bbead, bbead, ebead),
                binder_beads=(abead,),
                placer_beads=(bbead, ebead),
            ),
            cgexplore.molecular.SquarePrecursor(
                present_beads=(abead, bbead, ebead, bbead),
                binder_beads=(abead,),
                placer_beads=(bbead, bbead),
            ),
        ),
        gene_type="precursor",
    )

    # B precursors.
    chromosome_gen.add_gene(
        iteration=(
            cgexplore.molecular.LinearPrecursor(
                composition=(1,),
                present_beads=(cbead, cbead),
                binder_beads=(cbead,),
                placer_beads=(cbead,),
            ),
            cgexplore.molecular.LinearPrecursor(
                composition=(2,),
                present_beads=(cbead, dbead, cbead),
                binder_beads=(cbead,),
                placer_beads=(dbead,),
            ),
        ),
        gene_type="precursor",
    )
    # C precursors.
    chromosome_gen.add_gene(
        iteration=(
            cgexplore.molecular.LinearPrecursor(
                composition=(1,),
                present_beads=(abead, ebead),
                binder_beads=(abead,),
                placer_beads=(ebead,),
            ),
            cgexplore.molecular.TrianglePrecursor(
                present_beads=(abead, bbead, ebead),
                binder_beads=(abead,),
                placer_beads=(bbead, ebead),
            ),
            cgexplore.molecular.SquarePrecursor(
                present_beads=(abead, bbead, bbead, ebead),
                binder_beads=(abead,),
                placer_beads=(bbead, ebead),
            ),
            cgexplore.molecular.SquarePrecursor(
                present_beads=(abead, bbead, ebead, bbead),
                binder_beads=(abead,),
                placer_beads=(bbead, bbead),
            ),
        ),
        gene_type="precursor",
    )

    nb_scale = (1, 5, 10, 15)
    nb_sizes = (0.7, 1, 1.5, 2)
    bond_lengths = (1, 1.5, 2, 2.5)
    angle_values = (60, 90, 120, 150, 180)
    torsion_values = (180, 120, 60, 0)
    torsion_strengths = (0, 50)
    definer_dict = {}
    present_beads = (abead, bbead, cbead, dbead, ebead)
    for options in present_beads:
        type_string = f"{options.bead_type}"
        definer_dict[type_string] = ("nb", nb_scale, nb_sizes)
    for options in it.combinations_with_replacement(present_beads, 2):
        type_string = f"{options[0].bead_type}{options[1].bead_type}"
        definer_dict[type_string] = ("bond", bond_lengths, 1e5)
    for options in it.combinations_with_replacement(present_beads, 3):
        type_string = (
            f"{options[0].bead_type}{options[1].bead_type}"
            f"{options[2].bead_type}"
        )
        definer_dict[type_string] = ("angle", angle_values, 1e2)
    for options in it.combinations_with_replacement(present_beads, 4):
        continue
        type_string = (
            f"{options[0].bead_type}{options[1].bead_type}"
            f"{options[2].bead_type}{options[3].bead_type}"
        )
        definer_dict[type_string] = (
            "tors",
            "0123",
            torsion_values,
            torsion_strengths,
            1,
        )
    # Hard code some.
    definer_dict["ao"] = ("bond", 1.5, 1e5)
    definer_dict["bc"] = ("bond", 1.5, 1e5)
    definer_dict["co"] = ("bond", 1.0, 1e5)
    definer_dict["cc"] = ("bond", 1.0, 1e5)
    definer_dict["oo"] = ("bond", 1.0, 1e5)
    definer_dict["ccb"] = ("angle", 180.0, 1e2)
    definer_dict["aoc"] = ("angle", 180.0, 1e2)
    definer_dict["aoo"] = ("angle", 180.0, 1e2)
    definer_dict["bco"] = ("angle", tuple(i for i in range(90, 181, 5)), 1e2)
    definer_dict["cbc"] = ("angle", 180.0, 1e2)
    definer_dict["oao"] = ("angle", tuple(i for i in range(50, 121, 5)), 1e2)

    chromosome_gen.add_forcefield_dict(definer_dict=definer_dict)

    seeds = [4, 280, 999, 2196]
    num_generations = 20
    selection_size = 10

    for seed in seeds:
        generator = np.random.default_rng(seed)

        initial_population = chromosome_gen.select_random_population(
            generator,
            size=selection_size,
        )

        # Yield this.
        generations = []
        generation = cgexplore.systems_optimisation.Generation(
            chromosomes=initial_population,
            chromosome_generator=chromosome_gen,
            fitness_calculator=fitness_calculator,
            structure_calculator=structure_calculator,
            structure_output=struct_output,
            calculation_output=calc_dir,
            database=database,
        )

        generation.run_structures()
        _ = generation.calculate_fitness_values()
        generations.append(generation)

        progress_plot(
            generations=generations,
            output=figure_dir / f"fitness_progress_{seed}.png",
            num_generations=num_generations,
        )
        raise SystemExit
        for generation_id in range(1, num_generations + 1):
            logging.info(f"doing generation {generation_id} of seed {seed}")
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
                    gene_range=chromosome_gen.get_topo_ids(),
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
                    gene_range=chromosome_gen.get_topo_ids(),
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

            # logging.info("Doing crossovers.")
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
            merged_chromosomes.extend(generation.select_best(selection_size=5))

            generation = cgexplore.systems_optimisation.Generation(
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
            logging.info(f"new size is {generation.get_generation_size()}.")

            # Build, optimise and analyse each structure.
            generation.run_structures()
            _ = generation.calculate_fitness_values()

            # Add final state to generations.
            generations.append(generation)

            # Select the best of the generation for the next generation.
            # TODO: maybe make this roulete?
            best = generation.select_best(selection_size=selection_size)
            generation = cgexplore.systems_optimisation.Generation(
                chromosomes=chromosome_gen.dedupe_population(best),
                chromosome_generator=chromosome_gen,
                fitness_calculator=fitness_calculator,
                structure_calculator=structure_calculator,
                structure_output=struct_output,
                calculation_output=calc_dir,
                database=database,
            )
            logging.info(f"final size is {generation.get_generation_size()}.")

            progress_plot(
                generations=generations,
                output=figure_dir / f"fitness_progress_{seed}.png",
                num_generations=num_generations,
            )

            # Output best structures as images.
            best_chromosome = generation.select_best(selection_size=1)[0]
            best_name = (
                f"{best_chromosome.prefix}_{best_chromosome.get_string()}"
            )

        logging.info(f"top scorer is {best_name} (seed: {seed})")

    # Report.
    found = set()
    for generation in generations:
        for chromo in generation.chromosomes:
            found.add(chromo.name)
    logging.info(
        f"{len(found)} chromosomes found in EA (of "
        f"{chromosome_gen.get_num_chromosomes()})"
    )

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    ax, ax1 = axs
    xys = defaultdict(list)
    plotted = 0
    for entry in database.get_entries():
        tstr = entry.properties["topology"]

        fitness = fitness_calculator(
            chromosome=chromosome_gen.select_chromosome(
                tuple(entry.properties["chromosome"])
            ),
            chromosome_generator=chromosome_gen,
            database=database,
            structure_output=struct_output,
            calculation_output=calc_dir,
        )

        ax.scatter(
            entry.properties["opt_pore_data"]["min_distance"],
            entry.properties["energy_per_bb"],
            c=fitness,
            edgecolor="none",
            s=70,
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
        # highfitness = [i[0] for i in xys[(x, y)] if i[2] > 10]
        if len(stable) == 0:
            cmaps = ["white"]
        else:
            cmaps = sorted([colours()[i] for i in stable])

        if len(cmaps) > 8:
            cmaps = ["k"]
        cgexplore.utilities.draw_pie(
            colours=cmaps,
            xpos=x,
            ypos=y,
            size=400,
            ax=ax1,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("pore size", fontsize=16)
    ax.set_ylabel("energy", fontsize=16)
    ax.set_yscale("log")
    ax.set_title(
        f"plotted: {plotted}, found: {len(found)}, "
        f"possible: {chromosome_gen.get_num_chromosomes()}"
    )
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("ditopic", fontsize=16)
    ax1.set_ylabel("tritopic", fontsize=16)
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

    ax1.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        figure_dir / f"space_explored.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
