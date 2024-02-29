"""Script showing how cage optimisation can work."""

import logging
import pathlib
from collections import defaultdict

import cgexplore
import matplotlib.pyplot as plt
import numpy as np
import openmm
import stk


def pymol_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )


def shape_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )


def isomer_energy() -> float:
    return 0.3


def stoich_map(tstr: str) -> int:
    """Stoichiometry maps to the number of building blocks."""
    return {
        "2P3": 5,
        "4P6": 10,
        "4P62": 10,
        "6P9": 15,
        "8P12": 20,
    }[tstr]


def node_expected_topologies(tstr: str) -> int:
    """Number of nodes map to topologies."""
    return {
        "2P3": 2,
        "4P6": 4,
        "4P62": 4,
        "6P9": 6,
        "8P12": 8,
    }[tstr]


def colours() -> dict[str, str]:
    """Colours map to topologies."""
    return {
        "2P3": "#1f77b4",
        "4P6": "#ff7f0e",
        "4P62": "#2ca02c",
        "6P9": "#d62728",
        "8P12": "#17becf",
    }


def analyse_cage(
    database: cgexplore.utilities.AtomliteDatabase,
    name: str,
    output_dir: pathlib.Path,
    forcefield: cgexplore.forcefields.ForceField,
    node_element: str,
    chromosome: cgexplore.systems_optimisation.Chromosome,
) -> None:
    entry = database.get_entry(key=name)
    molecule = database.get_molecule(key=name)
    properties = entry.properties

    topology_str, _ = chromosome.get_topology_information()
    database.add_properties(
        key=name,
        property_dict={
            "cage_name": name,
            "prefix": name.split("_")[0],
            "chromosome": tuple(int(i) for i in chromosome.name),
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
            assert (  # noqa: S101
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
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: cgexplore.forcefields.ForceField,
    platform: str | None,
    database: cgexplore.utilities.AtomliteDatabase,
    chromosome: cgexplore.systems_optimisation.Chromosome,
) -> cgexplore.molecular.Conformer:
    fina_mol_file = output_dir / f"{name}_final.mol"

    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)

        return cgexplore.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if fina_mol_file.exists():
        ensemble = cgexplore.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=output_dir / f"{name}_base.mol",
            conformer_xyz=output_dir / f"{name}_ensemble.xyz",
            data_json=output_dir / f"{name}_ensemble.json",
            overwrite=False,
        )
        conformer = ensemble.get_lowest_e_conformer()
        database.add_molecule(molecule=conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": conformer.energy_decomposition,
                "source": conformer.source,
                "optimised": True,
            },
        )
        return ensemble.get_lowest_e_conformer()

    logging.info(f"optimising {name}")
    assigned_system = forcefield.assign_terms(molecule, name, output_dir)
    ensemble = cgexplore.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemble.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
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
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="opt1")
    except openmm.OpenMMException as error:
        if "Particle coordinate is NaN. " not in str(error):
            raise

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    for test_molecule in cgexplore.utilities.yield_shifted_models(
        temp_molecule,
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
                platform=platform,
            )
            ensemble.add_conformer(conformer=conformer, source="shifted")
        except openmm.OpenMMException as error:
            if "Particle coordinate is NaN. " not in str(error):
                raise

    # Collect and optimise structures nearby in phase space.
    neighbour_library = cgexplore.systems_optimisation.get_neighbour_library(
        chromosome_name=chromosome.name,
        modifiable_gene_ids=(3, 4),
    )
    for test_molecule in cgexplore.systems_optimisation.yield_near_models(
        molecule=molecule,
        name=name,
        output_dir=output_dir,
        neighbour_library=neighbour_library,
    ):
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
            file_suffix="nopt",
            output_dir=output_dir,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="nearby_opt")

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
        raise RuntimeError

    soft_md_data = soft_md_trajectory.get_data()

    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        raise RuntimeError

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
    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


def fitness_calculator(
    chromosome: cgexplore.systems_optimisation.Chromosome,
    chromosome_generator: cgexplore.systems_optimisation.ChromosomeGenerator,
    database: cgexplore.utilities.AtomliteDatabase,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
) -> float:
    target_pore = 2
    name = f"{chromosome.prefix}_{chromosome.get_string()}"
    entry = database.get_entry(name)
    tstr = entry.properties["topology"]
    pore = entry.properties["opt_pore_data"]["min_distance"]
    energy = entry.properties["energy_per_bb"]
    pore_diff = abs(target_pore - pore) / target_pore

    # If energy is too high, return bad fitness.
    if energy > isomer_energy() * 2:
        database.add_properties(
            key=name,
            property_dict={"fitness": 0},
        )
        return 0

    # Else, we check smaller topologies.

    # Select all with the same chromosome except for topology graph to check
    # for self-sorting.

    differ_by_topology = chromosome_generator.select_similar_chromosome(
        chromosome=chromosome,
        free_gene_id=0,
    )

    other_topologies = {}
    current_stoich = stoich_map(tstr)
    for other_chromosome in differ_by_topology:
        other_name = (
            f"{other_chromosome.prefix}_{other_chromosome.get_string()}"
        )
        other_tstr, _ = other_chromosome.get_topology_information()
        # Only recalculate smaller or equivalent cages.
        if stoich_map(other_tstr) <= current_stoich:
            if not database.has_molecule(other_name):
                # Run calculation.
                structure_calculator(
                    chromosome=other_chromosome,
                    database=database,
                    calculation_output=calculation_output,
                    structure_output=structure_output,
                )

            other_entry = database.get_entry(other_name)
            other_energy = other_entry.properties["energy_per_bb"]
            other_topologies[other_tstr] = other_energy
    if len(other_topologies) > 0 and min(other_topologies.values()) < energy:
        smaller_is_stable = True
    else:
        smaller_is_stable = False

    fitness = 0 if smaller_is_stable else 1 / (pore_diff + energy)

    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


def structure_calculator(
    chromosome: cgexplore.systems_optimisation.Chromosome,
    database: cgexplore.utilities.AtomliteDatabase,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
) -> None:
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
        output_dir=calculation_output,
        forcefield=forcefield,
        platform=None,
        database=database,
        chromosome=chromosome,
    )

    if conformer is not None:
        conformer.molecule.write(str(structure_output / f"{name}_optc.mol"))

    # Analyse cage.
    analyse_cage(
        name=name,
        output_dir=calculation_output,
        forcefield=forcefield,
        node_element="Ag",
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


def main() -> None:
    wd = pathlib.Path("/home/atarzia/workingspace/cage_optimisation_tests")
    struct_output = wd / "structures"
    cgexplore.utilities.check_directory(struct_output)
    calc_dir = wd / "calculations"
    cgexplore.utilities.check_directory(calc_dir)
    data_dir = wd / "data"
    cgexplore.utilities.check_directory(data_dir)
    figure_dir = wd / "figures"
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

    chromo_it = cgexplore.systems_optimisation.ChromosomeGenerator(
        prefix=prefix,
        present_beads=(abead, bbead, cbead, dbead),
        vdw_bond_cutoff=2,
    )
    chromo_it.add_gene(
        iteration=(
            ("2P3", stk.cage.TwoPlusThree),
            ("4P6", stk.cage.FourPlusSix),
            ("4P62", stk.cage.FourPlusSix2),
            ("6P9", stk.cage.SixPlusNine),
            ("8P12", stk.cage.EightPlusTwelve),
        ),
        gene_type="topology",
    )
    # Set some basic building blocks up. This should be run by an algorithm
    # later.
    chromo_it.add_gene(
        iteration=(
            cgexplore.molecular.TwoC1Arm(bead=bbead, abead1=cbead),
            # cgexplore.molecular.TwoC2Arm(
            #     bead=bbead, abead1=cbead, abead2=cbead
            # ),
            # cgexplore.molecular.TwoC3Arm(
            #     bead=bbead, abead1=cbead, abead2=cbead, abead3=cbead
            # ),
        ),
        gene_type="precursor",
    )
    chromo_it.add_gene(
        iteration=(
            cgexplore.molecular.ThreeC1Arm(bead=abead, abead1=dbead),
            # cgexplore.molecular.ThreeC2Arm(
            #     bead=abead, abead1=dbead, abead2=dbead
            # ),
        ),
        gene_type="precursor",
    )

    definer_dict = {
        # Bonds.
        "ao": ("bond", 1.5, 1e5),
        "bc": ("bond", 1.5, 1e5),
        "co": ("bond", 1.0, 1e5),
        "cc": ("bond", 1.0, 1e5),
        "oo": ("bond", 1.0, 1e5),
        # Angles.
        "ccb": ("angle", 180.0, 1e2),
        "ooc": ("angle", 180.0, 1e2),
        "occ": ("angle", 180.0, 1e2),
        "ccc": ("angle", 180.0, 1e2),
        "oco": ("angle", 180.0, 1e2),
        "aoc": ("angle", 180.0, 1e2),
        "aoo": ("angle", 180.0, 1e2),
        "bco": ("angle", tuple(i for i in range(90, 181, 5)), 1e2),
        "cbc": ("angle", 180.0, 1e2),
        "oao": ("angle", tuple(i for i in range(50, 121, 5)), 1e2),
        # Torsions.
        "ocbco": ("tors", "0134", 180, 50, 1),
        # Nonbondeds.
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "o": ("nb", 10.0, 1.0),
    }
    chromo_it.add_forcefield_dict(definer_dict=definer_dict)

    chromo_it.define_chromosomes()
    seeds = [4, 280, 999, 2196]
    num_generations = 20
    selection_size = 10

    for seed in seeds:
        generator = np.random.default_rng(seed)

        initial_population = chromo_it.select_random_population(
            generator,
            size=selection_size,
        )

        # Yield this.
        generations = []
        generation = cgexplore.systems_optimisation.Generation(
            chromosomes=initial_population,
            chromosome_generator=chromo_it,
            fitness_calculator=fitness_calculator,
            structure_calculator=structure_calculator,
            structure_output=struct_output,
            calculation_output=calc_dir,
            database=database,
        )
        generation.run_structures()
        _ = generation.calculate_fitness_values()
        generations.append(generation)

        for generation_id in range(1, num_generations + 1):
            logging.info(f"doing generation {generation_id} of seed {seed}")
            logging.info(
                f"initial size is {generation.get_generation_size()}."
            )
            logging.info("doing mutations.")
            merged_chromosomes = []
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    gene_range=chromo_it.get_term_ids(),
                    selection="random",
                    num_to_select=5,
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    gene_range=chromo_it.get_topo_ids(),
                    selection="random",
                    num_to_select=5,
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    gene_range=chromo_it.get_prec_ids(),
                    selection="random",
                    num_to_select=5,
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    gene_range=chromo_it.get_term_ids(),
                    selection="roulette",
                    num_to_select=5,
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    gene_range=chromo_it.get_topo_ids(),
                    selection="roulette",
                    num_to_select=5,
                    database=database,
                )
            )
            merged_chromosomes.extend(
                chromo_it.mutate_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    gene_range=chromo_it.get_prec_ids(),
                    selection="roulette",
                    num_to_select=5,
                    database=database,
                )
            )

            # logging.info("Doing crossovers.")
            merged_chromosomes.extend(
                chromo_it.crossover_population(
                    list_of_chromosomes=generation.chromosomes,
                    generator=generator,
                    selection="random",
                    num_to_select=5,
                    database=database,
                )
            )

            merged_chromosomes.extend(
                chromo_it.crossover_population(
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
                chromosomes=chromo_it.dedupe_population(merged_chromosomes),
                chromosome_generator=chromo_it,
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
            logging.info("maybe make this roulete?")
            best = generation.select_best(selection_size=selection_size)
            generation = cgexplore.systems_optimisation.Generation(
                chromosomes=chromo_it.dedupe_population(best),
                chromosome_generator=chromo_it,
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
            best_file = struct_output / (f"{best_name}_optc.mol")
            cgexplore.utilities.Pymol(
                output_dir=best_dir,
                file_prefix=f"{prefix}_{seed}_g{generation_id}_best",
                settings={
                    "grid_mode": 0,
                    "rayx": 1000,
                    "rayy": 1000,
                    "stick_rad": 0.7,
                    "vdw": 0,
                    "zoom_string": "custom",
                },
                pymol_path=pymol_path(),
            ).visualise(
                [best_file],
                orient_atoms=None,
            )

        logging.info(f"top scorer is {best_name} (seed: {seed})")

    # Report.
    found = set()
    for generation in generations:
        for chromo in generation.chromosomes:
            found.add(chromo.name)
    logging.info(
        f"{len(found)} chromosomes found in EA (of "
        f"{chromo_it.get_num_chromosomes()})"
    )

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    ax, ax1 = axs
    xys = defaultdict(list)
    plotted = 0
    for entry in database.get_entries():
        tstr = entry.properties["topology"]

        fitness = fitness_calculator(
            chromosome=chromo_it.select_chromosome(
                tuple(entry.properties["chromosome"])
            ),
            chromosome_generator=chromo_it,
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

        if len(cmaps) > 8:  # noqa: PLR2004
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
        f"possible: {chromo_it.get_num_chromosomes()}"
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
        figure_dir / "space_explored.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
