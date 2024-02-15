import logging
import os
import pathlib
from collections import abc
from dataclasses import dataclass

import cgexplore
import matplotlib.pyplot as plt
import numpy as np
import stk
from openmm import openmm


def shape_path():
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )


def stoich_map(tstr):
    """Stoichiometry maps to the number of building blocks."""
    return {"4P6": 10, "2P3": 5}[tstr]


def node_expected_topologies(tstr):
    """Number of nodes map to topologies."""
    return {"4P6": 4, "2P3": 2}[tstr]


def colours():
    """Colours map to topologies."""
    return {"4P6": "#ff7f0e", "2P3": "#1f77b4"}


def analyse_cage(
    database,
    name,
    output_dir,
    forcefield,
    node_element,
    chromosome,
):
    entry = database.get_entry(key=name)
    molecule = database.get_molecule(key=name)
    properties = entry.properties

    topology_str, _ = chromosome.get_topology_information()
    database.add_properties(
        key=name,
        property_dict={
            "cage_name": name,
            "prefix": name.split("_")[0],
            "chromosome": chromosome.name,
            "topology": topology_str,
        },
    )

    if "energy_per_bb" not in properties:
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
            "energy_per_bb": fin_energy / stoich_map(topology_str),
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
        opt_pore_data = g_measure.calculate_min_distance(molecule)
        database.add_properties(
            key=name,
            property_dict={
                "bond_data": bond_data,
                "angle_data": angle_data,
                "dihedral_data": dihedral_data,
                "opt_pore_data": opt_pore_data,
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
            "vdws": "von",
            "v_dict": v_dict,
            "k_dict": k_dict,
        }

        database.add_properties(
            key=name,
            property_dict={"forcefield_dict": forcefield_dict},
        )

    if "node_shape_measures" not in properties:
        shape_calc = cgexplore.analysis.ShapeMeasure(
            output_dir=(output_dir / f"{name}_nshape"),
            shape_path=shape_path(),
            shape_string=None,
        )

        n_shape_mol = shape_calc.get_shape_molecule_byelement(
            molecule=molecule,
            element=node_element,
            expected_points=node_expected_topologies(topology_str),
        )
        if n_shape_mol is None:
            node_shape_measures = None
        else:

            node_shape_measures = shape_calc.calculate(n_shape_mol)

        database.add_properties(
            key=name,
            property_dict={"node_shape_measures": node_shape_measures},
        )


def optimise_cage(
    molecule,
    name,
    output_dir,
    forcefield,
    platform,
    database,
):

    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        return cgexplore.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)
    ensemble = cgexplore.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
        conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
        data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
        overwrite=True,
    )
    temp_molecule = cgexplore.utilities.run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=50,
        angle_ff_scale=50,
        max_iterations=20,
        platform=platform,
    )

    try:
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
                molecule=temp_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="opt1",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="opt1")
    except openmm.OpenMMException as error:
        if "Particle coordinate is NaN. " not in str(error):
            raise error

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    for test_molecule in cgexplore.utilities.yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
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
            ensemble.add_conformer(conformer=conformer, source="shifted")
        except openmm.OpenMMException as error:
            if "Particle coordinate is NaN. " not in str(error):
                raise error

    # Collect and optimise structures nearby in phase space.
    logging.info(
        f"optimisation of nearby structures of {name} - DOOOO (use filtered chromosome)"
    )
    # neighbour_library = get_neighbour_library(
    #     ffid=forcefield.get_identifier(),
    #     fftype=forcefield.get_prefix(),
    # )
    # for test_molecule in yield_near_models(
    #     molecule=molecule,
    #     name=name,
    #     output_dir=output_dir,
    #     neighbour_library=neighbour_library,
    # ):
    #     conformer = run_optimisation(
    #         assigned_system=AssignedSystem(
    #             molecule=test_molecule,
    #             forcefield_terms=assigned_system.forcefield_terms,
    #             system_xml=assigned_system.system_xml,
    #             topology_xml=assigned_system.topology_xml,
    #             bead_set=assigned_system.bead_set,
    #             vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
    #         ),
    #         name=name,
    #         file_suffix="nopt",
    #         output_dir=output_dir,
    #         # max_iterations=50,
    #         platform=platform,
    #     )
    #     ensemble.add_conformer(conformer=conformer, source="nearby_opt")

    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = cgexplore.utilities.run_soft_md_cycle(
        name=name,
        assigned_system=cgexplore.forcefields.AssignedSystem(
            molecule=ensemble.get_lowest_e_conformer().molecule,
            forcefield_terms=assigned_system.forcefield_terms,
            system_xml=assigned_system.system_xml,
            topology_xml=assigned_system.topology_xml,
            bead_set=assigned_system.bead_set,
            vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
        ),
        output_dir=output_dir,
        suffix="smd",
        bond_ff_scale=10,
        angle_ff_scale=10,
        temperature=300 * openmm.unit.kelvin,
        num_steps=num_steps,
        time_step=0.5 * openmm.unit.femtoseconds,
        friction=1.0 / openmm.unit.picosecond,
        reporting_freq=traj_freq,
        traj_freq=traj_freq,
        platform=platform,
    )
    if soft_md_trajectory is None:
        logging.info(f"!!!!! {name} MD exploded !!!!!")
        # md_exploded = True
        raise ValueError("OpenMM Exception")

    soft_md_data = soft_md_trajectory.get_data()

    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        # md_failed = True
        raise ValueError

    # Go through each conformer from soft MD.
    # Optimise them all.
    for md_conformer in soft_md_trajectory.yield_conformers():
        conformer = cgexplore.utilities.run_optimisation(
            assigned_system=cgexplore.forcefields.AssignedSystem(
                molecule=md_conformer.molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="smd")
    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()

    # Add to atomlite database.
    database.add_molecule(molecule=min_energy_conformer.molecule, key=name)
    database.add_properties(
        key=name,
        property_dict={
            "energy_decomposition": min_energy_conformer.energy_decomposition,
            "source": min_energy_conformer.source,
            "optimised": True,
        },
    )

    return min_energy_conformer


@dataclass
class Generation:
    chromosomes: abc.Iterable[cgexplore.systems_optimisation.Chromosome]
    fitness_function: abc.Callable

    def get_generation_size(self):
        return len(self.chromosomes)

    def get_fitness_values(self):
        return self.fitnesses

    def select_best(self, selection_size, database):
        temp = [
            (i, self.fitness_function(i, database)) for i in self.chromosomes
        ]
        best_indices = tuple(
            sorted(range(len(temp)), key=lambda i: temp[i][1], reverse=False)
        )[:selection_size]
        best = [self.chromosomes[i] for i in best_indices]
        return best

    def calculate_fitness_values(self, database):
        self.fitnesses = [
            self.fitness_function(i, database) for i in self.chromosomes
        ]

    def run_structures(self, calc_dir, struct_output, database):
        length = len(self.chromosomes)
        for i, chromosome in enumerate(self.chromosomes):
            logging.info(f"building {chromosome} ({i} of {length})")
            # Build structure.
            topology_str, topology_fun = chromosome.get_topology_information()
            building_blocks = chromosome.get_building_blocks()
            cage = stk.ConstructedMolecule(topology_fun(building_blocks))
            name = f"{chromosome.prefix}_{chromosome.get_string()}"

            # Select forcefield by chromosome.
            forcefield = chromosome.get_forcefield()

            # Optimise with some procedure.
            conformer = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calc_dir,
                forcefield=forcefield,
                platform=None,
                database=database,
            )

            if conformer is not None:
                conformer.molecule.write(
                    str(struct_output / f"{name}_optc.mol")
                )

            # Analyse cage.
            analyse_cage(
                name=name,
                output_dir=calc_dir,
                forcefield=forcefield,
                node_element="Ag",
                database=database,
                chromosome=chromosome,
            )


def fitness_calculator(chromosome, database):
    name = f"{chromosome.prefix}_{chromosome.get_string()}"
    entry = database.get_entry(name)
    pore = entry.properties["opt_pore_data"]["min_distance"]
    energy = entry.properties["energy_per_bb"]
    pore_diff = abs(2.5 - pore)
    fitness = pore_diff + energy
    return fitness


def progress_plot(generations, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    fitnesses = []
    for generation in generations:
        fitnesses.append(generation.get_fitness_values())
    ax.plot(
        [max(i) for i in fitnesses],
        c="#087E8B",
        lw=2,
        marker="o",
        markersize=10,
        label="max",
    )

    ax.plot(
        [np.mean(i) for i in fitnesses],
        c="#FF5A5F",
        lw=2,
        marker="o",
        markersize=10,
        label="mean",
    )
    ax.plot(
        [min(i) for i in fitnesses],
        c="#6D435A",
        lw=2,
        marker="o",
        markersize=10,
        label="min",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_xlim(0, len(generations))

    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


def main():
    wd = pathlib.Path("/home/atarzia/workingspace/cage_optimisation_tests")
    struct_output = wd / "structures"
    calc_dir = wd / "calculations"
    data_dir = wd / "data"
    figure_dir = wd / "figures"

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

    chromo_it = cgexplore.systems_optimisation.ChromosomeGenerator(
        prefix=prefix,
        present_beads=(abead, bbead, cbead, dbead),
        vdw_bond_cutoff=2,
    )
    chromo_it.add_gene(
        iteration=(
            ("4P6", stk.cage.FourPlusSix),
            ("2P3", stk.cage.TwoPlusThree),
        ),
        gene_type="topology",
    )
    # Set some basic building blocks up. This should be run by an algorithm
    # later.
    chromo_it.add_gene(
        iteration=(
            cgexplore.molecular.TwoC1Arm(bead=bbead, abead1=cbead),
            cgexplore.molecular.TwoC2Arm(
                bead=bbead, abead1=cbead, abead2=cbead
            ),
            cgexplore.molecular.TwoC3Arm(
                bead=bbead, abead1=cbead, abead2=cbead, abead3=cbead
            ),
        ),
        gene_type="precursor",
    )
    chromo_it.add_gene(
        iteration=(
            cgexplore.molecular.ThreeC1Arm(bead=abead, abead1=dbead),
            cgexplore.molecular.ThreeC2Arm(
                bead=abead, abead1=dbead, abead2=dbead
            ),
        ),
        gene_type="precursor",
    )

    definer_dict = {
        # Bonds.
        "ao": ("bond", (1, 1.5), 1e5),
        "bc": ("bond", 2.0, 1e5),
        "co": ("bond", 2.0, 1e5),
        "cc": ("bond", (0.5, 1.0, 1.5), 1e5),
        "oo": ("bond", (0.5, 1.0, 1.5), 1e5),
        # Angles.
        "ccb": ("angle", 180.0, 1e2),
        "ooc": ("angle", 180.0, 1e2),
        "occ": ("angle", 180.0, 1e2),
        "ccc": ("angle", 180.0, 1e2),
        "oco": ("angle", 180.0, 1e2),
        "aoc": ("angle", 180.0, 1e2),
        "aoo": ("angle", 180.0, 1e2),
        "bco": ("angle", (90, 120, 130, 140, 150, 170, 180), (5e1, 1e2)),
        "cbc": ("angle", 180.0, 1e2),
        "oao": ("angle", (60, 70, 90, 110, 120), 1e2),
        # Torsions.
        # Nonbondeds.
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "o": ("nb", 10.0, 1.0),
    }
    chromo_it.add_forcefield_dict(definer_dict=definer_dict)

    chromo_it.define_chromosomes()
    generator = np.random.default_rng(4)
    num_generations = 10
    selection_size = 10

    initial_population = chromo_it.select_random_population(
        generator,
        size=selection_size,
    )

    # Yield this.
    generations = []
    generation = Generation(
        chromosomes=initial_population,
        fitness_function=fitness_calculator,
    )
    generation.run_structures(
        calc_dir=calc_dir,
        struct_output=struct_output,
        database=database,
    )
    generation.calculate_fitness_values(database)
    generations.append(generation)

    logging.info("need better selectors and such")

    for generation_id in range(1, num_generations):
        logging.info(f"doing {generation_id}")
        logging.info(f"initial size is {generation.get_generation_size()}.")
        logging.info("doing mutations.")
        term_mutants = chromo_it.mutate_population(
            generation.chromosomes,
            generator=generator,
            gene_range=chromo_it.get_term_ids(),
        )
        topo_mutants = chromo_it.mutate_population(
            generation.chromosomes,
            generator=generator,
            gene_range=chromo_it.get_topo_ids(),
        )
        prec_mutants = chromo_it.mutate_population(
            generation.chromosomes,
            generator=generator,
            gene_range=chromo_it.get_prec_ids(),
        )

        # logging.info("Doing crossovers.")
        # term_offspring = chromo_it.crossover_population(
        #     generation.chromosomes,
        #     generator=generator,
        # )
        # print(offspring)

        # Merge crossovers and mutations.
        # merged_chromosomes = offspring + mutants

        # Add the best 5 to the new generation.
        best = generation.select_best(
            selection_size=5,
            database=database,
        )

        merged_chromosomes = term_mutants + prec_mutants + topo_mutants + best
        generation = Generation(
            chromosomes=chromo_it.dedupe_population(merged_chromosomes),
            fitness_function=fitness_calculator,
        )
        logging.info(f"new size is {generation.get_generation_size()}.")

        logging.info("running calcs.")
        generation.run_structures(
            calc_dir=calc_dir,
            struct_output=struct_output,
            database=database,
        )
        generation.calculate_fitness_values(database)

        logging.info("selecting best.")
        best = generation.select_best(
            selection_size=selection_size,
            database=database,
        )
        generations.append(generation)

    progress_plot(
        generations=generations,
        output=figure_dir / "fitness_progress.png",
    )

    # Report.
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    ax, ax1 = axs
    found = set()
    for generation in generations:
        for chromo in generation.chromosomes:
            found.add(chromo.name)

    logging.info(
        f"{len(found)} chromosomes found in EA (of "
        f"{len(chromo_it.chromosomes)})"
    )
    for entry in database.get_entries():
        tstr = entry.properties["topology"]

        entry_chromo = tuple(entry.properties["chromosome"])
        if entry_chromo in found:
            ax.scatter(
                entry.properties["opt_pore_data"]["min_distance"],
                entry.properties["energy_per_bb"],
                c="none",
                edgecolor="k",
                s=60,
                marker="o",
                alpha=1.0,
                lw=2,
            )
            ax1.scatter(
                entry.properties["forcefield_dict"]["v_dict"]["b_c_o"],
                entry.properties["forcefield_dict"]["v_dict"]["o_a_o"],
                c="none",
                edgecolor="k",
                s=60,
                marker="o",
                alpha=1.0,
                lw=2,
            )

        ax.scatter(
            entry.properties["opt_pore_data"]["min_distance"],
            entry.properties["energy_per_bb"],
            c=colours()[tstr],
            edgecolor="none",
            s=60,
            marker="o",
            alpha=1.0,
        )
        ax1.scatter(
            entry.properties["forcefield_dict"]["v_dict"]["b_c_o"],
            entry.properties["forcefield_dict"]["v_dict"]["o_a_o"],
            c=colours()[tstr],
            edgecolor="none",
            s=60,
            marker="o",
            alpha=1.0,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("pore size", fontsize=16)
    ax.set_ylabel("energy", fontsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("ditopic", fontsize=16)
    ax1.set_ylabel("tritopic", fontsize=16)

    for tstr in colours():
        ax.scatter(
            None,
            None,
            c=colours()[tstr],
            edgecolor="none",
            s=60,
            marker="o",
            alpha=1.0,
            label=tstr,
        )

    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "space_explored.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
