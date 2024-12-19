"""Utilities module."""

import logging
import pathlib

import networkx as nx
import numpy as np
import stk
import stko
from openmm import OpenMMException, openmm

from cgexplore._internal.forcefields.assigned_system import AssignedSystem
from cgexplore._internal.forcefields.forcefield import ForceField
from cgexplore._internal.molecular.conformer import Conformer
from cgexplore._internal.molecular.ensembles import Ensemble
from cgexplore._internal.scram.topology_code import TopologyCode
from cgexplore._internal.topologies.custom_topology import CustomTopology
from cgexplore._internal.utilities.databases import AtomliteDatabase
from cgexplore._internal.utilities.generation_utilities import (
    run_constrained_optimisation,
    run_optimisation,
    run_soft_md_cycle,
    yield_shifted_models,
)

from .building_block_enum import BuildingBlockConfiguration
from .enumeration import IHomolepticTopologyIterator, TopologyIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def graph_optimise_cage(  # noqa: PLR0913
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: ForceField,
    platform: str | None,
    database_path: pathlib.Path,
) -> Conformer:
    """Optimise a toy model cage."""
    fina_mol_file = output_dir / f"{name}_wipfinal.mol"

    database = AtomliteDatabase(database_path)
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if fina_mol_file.exists():
        ensemble = Ensemble(
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

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)
    if (output_dir / f"{name}_ensemblewip.xyz").exists():
        (output_dir / f"{name}_ensemblewip.xyz").unlink()
    ensemble = Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemblewip.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
        overwrite=True,
    )
    temp_molecule = run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    conformer = run_optimisation(
        assigned_system=AssignedSystem(
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

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    for test_molecule in yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
    ):
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
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

    stko_graph = stko.Network.init_from_molecule(conformer.molecule)
    for i, nx_positions in enumerate(
        (
            nx.spectral_layout(stko_graph.get_graph(), dim=3),
            nx.get_node_attributes(
                nx.random_geometric_graph(
                    n=conformer.molecule.get_num_atoms(), radius=1, dim=3
                ),
                "pos",
            ),
            nx.spring_layout(stko_graph.get_graph(), dim=3),
            nx.kamada_kawai_layout(stko_graph.get_graph(), dim=3),
        )
    ):
        # We allow these to independantly failed because the nx graphs can
        # be ridiculous.
        for j, scaler in enumerate((5, 10, 15)):
            try:
                pos_mat = np.array([nx_positions[i] for i in nx_positions])
                if pos_mat.shape[1] != 3:  # noqa: PLR2004
                    msg = "built a non 3D graph"
                    raise RuntimeError(msg)

                test_molecule = conformer.molecule.with_position_matrix(
                    pos_mat * scaler
                )
                conformer = run_optimisation(
                    assigned_system=forcefield.assign_terms(
                        test_molecule, name, output_dir
                    ),
                    name=name,
                    file_suffix="nopt",
                    output_dir=output_dir,
                    platform=platform,
                )

                ensemble.add_conformer(conformer=conformer, source=f"nx{i}{j}")
            except OpenMMException:
                logging.info("failed graph opt of %s", name)

    # Try with graph positions.
    rng = np.random.default_rng(seed=100)
    for attempt in range(10):
        pos_mat = rng.random(size=(conformer.molecule.get_num_atoms(), 3))
        test_molecule = conformer.molecule.with_position_matrix(pos_mat * 10)
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
                molecule=test_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix=f"ropt{attempt}",
            output_dir=output_dir,
            platform=platform,
        )

        ensemble.add_conformer(conformer=conformer, source="shifted")

    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        "%s from %s with energy: %s kJ.mol-1",
        min_energy_conformerid,
        min_energy_conformer.source,
        round(min_energy, 2),
    )

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


def optimise_cage(  # noqa: PLR0913, C901, PLR0915, PLR0912
    molecule: stk.Molecule,
    name: str,
    output_dir: pathlib.Path,
    forcefield: ForceField,
    platform: str | None,
    database_path: pathlib.Path,
) -> Conformer:
    """Optimise a toy model cage."""
    fina_mol_file = output_dir / f"{name}_final.mol"

    database = AtomliteDatabase(database_path)
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if fina_mol_file.exists():
        ensemble = Ensemble(
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

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)

    ensemble = Ensemble(
        base_molecule=molecule,
        base_mol_path=output_dir / f"{name}_base.mol",
        conformer_xyz=output_dir / f"{name}_ensemble.xyz",
        data_json=output_dir / f"{name}_ensemble.json",
        overwrite=True,
    )

    temp_molecule = run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    conformer = run_optimisation(
        assigned_system=AssignedSystem(
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

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    for test_molecule in yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
    ):
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
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

    # Add neighbours to systematic scan.
    if "scan" in name:
        if "ufo" in name:
            _, multiplier, sisj = name.split("_")
            si, sj = sisj.split("-")

            potential_names = [
                f"ufoscan_{multiplier}_{int(si)-1}-{int(sj)-1}",
                f"ufoscan_{multiplier}_{int(si)-1}-{int(sj)}",
                f"ufoscan_{multiplier}_{int(si)}-{int(sj)-1}",
            ]

        else:
            si, sj = name.split("_")[1].split("-")

            potential_names = [
                f"scan_{int(si)-1}-{int(sj)-1}",
                f"scan_{int(si)-1}-{int(sj)}",
                f"scan_{int(si)}-{int(sj)-1}",
            ]
    elif "ts_" in name:
        _, tstr, si, sj, _at = name.split("_")

        potential_names = []
        for i in range(6):
            potential_names.extend(
                [
                    f"ts_{tstr}_{int(si)-1}_{int(sj)-1}_{i}",
                    f"ts_{tstr}_{int(si)-1}_{int(sj)}_{i}",
                    f"ts_{tstr}_{int(si)}_{int(sj)-1}_{i}",
                ]
            )
    else:
        potential_names = []

    for potential_name in potential_names:
        potential_file = output_dir / f"{potential_name}_final.mol"
        if not potential_file.exists():
            continue
        test_molecule = temp_molecule.with_structure_from_file(potential_file)
        conformer = run_optimisation(
            assigned_system=AssignedSystem(
                molecule=test_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="ns",
            output_dir=output_dir,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="ns")

    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = run_soft_md_cycle(
        name=name,
        assigned_system=AssignedSystem(
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
    failed_md = False
    if soft_md_trajectory is None:
        failed_md = True

    if not failed_md:
        soft_md_data = soft_md_trajectory.get_data()
        # Check that the trajectory is as long as it should be.
        if len(soft_md_data) != num_steps / traj_freq:
            failed_md = True

        # Go through each conformer from soft MD.
        # Optimise them all.
        for md_conformer in soft_md_trajectory.yield_conformers():
            if failed_md:
                continue
            conformer = run_optimisation(
                assigned_system=AssignedSystem(
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
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        "%s from %s with energy: %s kJ.mol-1",
        min_energy_conformerid,
        min_energy_conformer.source,
        round(min_energy, 2),
    )

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


def try_except_construction(
    iterator: TopologyIterator | IHomolepticTopologyIterator,
    topology_code: TopologyCode,
    building_block_configuration: BuildingBlockConfiguration | None = None,
    vertex_positions: dict[int, np.ndarray] | None = None,
) -> stk.ConstructedMolecule:
    """Try construction with alignment, then without."""
    if building_block_configuration is None:
        bbs = iterator.building_blocks
    else:
        bbs = building_block_configuration.get_building_block_dictionary()

    try:
        # Try with aligning vertices.
        constructed_molecule = stk.ConstructedMolecule(
            CustomTopology(
                building_blocks=bbs,
                vertex_prototypes=iterator.get_vertex_prototypes(
                    unaligning=False
                ),
                # Convert to edge prototypes.
                edge_prototypes=topology_code.edges_from_connection(
                    iterator.get_vertex_prototypes(unaligning=False)
                ),
                vertex_alignments=None,
                vertex_positions=vertex_positions,
                scale_multiplier=iterator.scale_multiplier,
            )
        )

    except ValueError:
        # Try with unaligning.
        constructed_molecule = stk.ConstructedMolecule(
            CustomTopology(
                building_blocks=bbs,
                vertex_prototypes=iterator.get_vertex_prototypes(
                    unaligning=True
                ),
                # Convert to edge prototypes.
                edge_prototypes=topology_code.edges_from_connection(
                    iterator.get_vertex_prototypes(unaligning=True)
                ),
                vertex_alignments=None,
                vertex_positions=vertex_positions,
                scale_multiplier=iterator.scale_multiplier,
            )
        )
    return constructed_molecule
