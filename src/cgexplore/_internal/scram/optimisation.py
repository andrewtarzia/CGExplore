"""Define classes for optimisation of structures."""

import logging
import pathlib
from collections import abc
from copy import deepcopy

import atomlite
import networkx as nx
import numpy as np
import stk
import stko
from scipy import optimize

from cgexplore._internal.forcefields.forcefield import ForceField
from cgexplore._internal.molecular.conformer import Conformer
from cgexplore._internal.systems_optimisation.utilities import (
    get_forcefield_from_dict,
)
from cgexplore._internal.utilities.databases import AtomliteDatabase
from cgexplore._internal.utilities.generation_utilities import run_optimisation
from cgexplore._internal.utilities.utilities import get_energy_per_bb

from .building_block_enum import BuildingBlockConfiguration
from .construction import try_except_construction
from .enumeration import TopologyIterator
from .topology_code import TopologyCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_regraphed_molecule(
    graph_type: str,
    scale: float,
    topology_code: TopologyCode,
    iterator: TopologyIterator,
    bb_config: BuildingBlockConfiguration | None,
) -> stk.ConstructedMolecule:
    """Take a graph that considers all atoms, and get atom positions.

    The initial graph is generated with `stko.Network.init_from_molecule`.

    Parameters:
        graph_type:
            Which networkx layout to use (of `spring_layout`,
            `karama_kawai_layout`).

        scale:
            Scale factor to apply to eventual constructed molecule.

        topology_code:
            The code defining the topology graph.

        iterator:
            The `scram` algorithm used to generate the graph and configuration.

        bb_config:
            The configuration of building blocks on the graph.

    Returns:
        A constructed molecule at (0, 0, 0).

    """
    logger.warning(
        "Caution with this because it currently can change the cis/trans in "
        "m2l4"
    )

    constructed_molecule = try_except_construction(
        iterator=iterator,
        topology_code=topology_code,
        building_block_configuration=bb_config,
        vertex_positions=None,
    )

    stko_graph = stko.Network.init_from_molecule(constructed_molecule)
    if graph_type == "spring":
        nx_positions = nx.spring_layout(stko_graph.get_graph(), dim=3)
    elif graph_type == "kamada":
        nx_positions = nx.kamada_kawai_layout(stko_graph.get_graph(), dim=3)
    else:
        raise NotImplementedError
    pos_mat = np.array([nx_positions[i] for i in nx_positions])
    return constructed_molecule.with_position_matrix(
        pos_mat * float(scale)
    ).with_centroid(np.array((0.0, 0.0, 0.0)))


def get_vertexset_molecule(
    graph_type: str | None,
    scale: float,
    topology_code: TopologyCode,
    iterator: TopologyIterator,
    bb_config: BuildingBlockConfiguration,
) -> stk.ConstructedMolecule:
    """Take a graph and genereate from graph vertex positions.

    Parameters:
        graph_type:
            Which networkx layout to use (of `spring_layout`,
            `kamada_kawai_layout`, `spectral_layout`).

        scale:
            Scale factor to apply to eventual constructed molecule.

        topology_code:
            The code defining the topology graph.

        iterator:
            The `scram` algorithm used to generate the graph and configuration.

        bb_config:
            The configuration of building blocks on the graph.

    Returns:
        A constructed molecule at (0, 0, 0).

    """
    if graph_type is None:
        return try_except_construction(
            iterator=iterator,
            topology_code=topology_code,
            building_block_configuration=bb_config,
            vertex_positions=None,
        )

    nx_graph = topology_code.get_nx_graph()

    if graph_type == "kamada":
        nxpos = nx.kamada_kawai_layout(nx_graph, dim=3)
    elif graph_type == "spring":
        nxpos = nx.spring_layout(nx_graph, dim=3)
    elif graph_type == "spectral":
        nxpos = nx.spectral_layout(nx_graph, dim=3)
    else:
        raise NotImplementedError

    vertex_positions = {
        nidx: np.array(nxpos[nidx]) * float(scale)
        for nidx in topology_code.get_nx_graph().nodes
    }
    return try_except_construction(
        iterator=iterator,
        topology_code=topology_code,
        building_block_configuration=bb_config,
        vertex_positions=vertex_positions,
    )


def target_optimisation(  # noqa: C901, PLR0913, PLR0915
    database_path: pathlib.Path,
    calculation_dir: pathlib.Path,
    target_key: str,
    definer_dict: dict,
    modifiable_terms: list[str],
    forcefield: ForceField,
) -> None:
    """Optimise the FF terms based on a target.

    Currently, only bond and angle terms can be modified. Additionally, there
    are set bounds of plus/minus 5 Angstroms/20 degrees for bonds and angles,
    respectivelly.

    Generates a new forcefield database of outputs in the calculation dir. Also
    adds the properties to the original database.


    Parameters:
        database_path:
            Path to the database holding the target entry for optimisation.
            This will be updated during this function call.

        calculation_dir:
            Path to the calculation directory.

        target_key:
            Key of the target entry.

        definer_dict:
            Dictionary of definer terms. Formatting should match the definer
            dict in `systems_optimisation`.

        modifiable_terms:
            List of modifiable terms. These are keys in the definer dict.

        forcefield:
            Forcefield used during original structure generation to collate
            other details before optimisation.


    """
    target_entry = AtomliteDatabase(database_path).get_entry(target_key)

    num_building_blocks = int(target_entry.properties["num_bbs"])  # type: ignore[arg-type]
    input_cage = stk.BuildingBlock.init_from_rdkit_mol(
        atomlite.json_to_rdkit(target_entry.molecule)
    )
    name = target_key + "_ffopt"
    ff_database_path = calculation_dir / f"{name}_ffopt.db"
    optimised_file = calculation_dir / f"{name}_ffopt.mol"

    if ff_database_path.exists():
        properties = (
            AtomliteDatabase(ff_database_path).get_entry(target_key).properties
        )

    else:
        ff_map = dict(enumerate(modifiable_terms))

        initial_ff_params = []
        bounds = []
        for i in modifiable_terms:
            if definer_dict[i][0] in ("bond", "nb"):
                angle = False
                max_change = 0.5

            elif definer_dict[i][0] in ("angle", "pyramid", "tors"):
                angle = True
                max_change = 20
            else:
                raise RuntimeError

            if (
                definer_dict[i][0] == "bond"
                or definer_dict[i][0] == "angle"
                or definer_dict[i][0] == "pyramid"
            ):
                initial_ff_params.append(definer_dict[i][1])
                value = definer_dict[i][1]
            elif definer_dict[i][0] == "tors" or definer_dict[i][0] == "nb":
                initial_ff_params.append(definer_dict[i][2])
                value = definer_dict[i][2]
            else:
                raise RuntimeError

            bounds.append(
                (
                    max((value - max_change, 0)),
                    value + max_change
                    if not angle
                    else (min((value + max_change, 180))),
                )
            )

        def structure_f(
            params: abc.Sequence[float],
        ) -> Conformer:
            # Get FF.
            temp_definer_dict = deepcopy(definer_dict)
            for i, value in enumerate(params):
                term = ff_map[i]
                if (
                    temp_definer_dict[term][0] == "bond"
                    or temp_definer_dict[term][0] == "angle"
                    or temp_definer_dict[term][0] == "pyramid"
                ):
                    temp_definer_dict[term] = (
                        temp_definer_dict[term][0],
                        value,
                        temp_definer_dict[term][2],
                    )
                elif temp_definer_dict[term][0] == "tors":
                    temp_definer_dict[term] = (
                        temp_definer_dict[term][0],
                        temp_definer_dict[term][1],
                        value,
                        temp_definer_dict[term][3],
                        temp_definer_dict[term][4],
                    )
                elif temp_definer_dict[term][0] == "nb":
                    temp_definer_dict[term] = (
                        temp_definer_dict[term][0],
                        temp_definer_dict[term][1],
                        value,
                    )
                else:
                    raise RuntimeError

            temp_forcefield = get_forcefield_from_dict(
                identifier="ffopt",
                prefix="ffopt",
                vdw_bond_cutoff=forcefield.get_vdw_bond_cutoff(),
                present_beads=forcefield.get_present_beads(),
                definer_dict=temp_definer_dict,
            )

            # Run optimisation.
            return run_optimisation(
                assigned_system=temp_forcefield.assign_terms(
                    input_cage,
                    name,
                    calculation_dir,
                ),
                name=name,
                file_suffix="ffopt",
                output_dir=calculation_dir,
                platform=None,
            )

        def f(params: abc.Sequence[float]) -> float:
            if any(i < 0 for i in params):
                return 100
            conformer = structure_f(params)

            # Return Energy.
            return get_energy_per_bb(
                energy_decomposition=conformer.energy_decomposition,
                number_building_blocks=num_building_blocks,
            )

        result = optimize.dual_annealing(
            f,
            bounds,
            x0=initial_ff_params,
            minimizer_kwargs={"method": "BFGS", "tol": 0.01},
            maxiter=10,
            maxfun=400,
            rng=np.random.default_rng(2785),
        )
        logger.info("optimisation %s with E: %s", result.success, result.fun)

        min_conformer = structure_f(result.x)
        if (
            get_energy_per_bb(
                energy_decomposition=min_conformer.energy_decomposition,
                number_building_blocks=num_building_blocks,
            )
            > result.fun * 1.1
        ):
            raise RuntimeError

        min_conformer.molecule.write(optimised_file)

        properties = {
            "optimisation_success": result.success,
            "optimisation_energy_per_bb": float(result.fun),
            "optimisation_x": [float(i) for i in result.x],
            "optimisation_map": ff_map,  # type:ignore[dict-item]
            "optimisation_rmsd": stko.KabschRmsdCalculator(
                input_cage
            ).calculate(min_conformer.molecule),
        }
        AtomliteDatabase(ff_database_path).add_molecule(
            key=target_key,
            molecule=min_conformer.molecule,
        )
        AtomliteDatabase(ff_database_path).add_properties(
            key=target_key,
            property_dict=properties,
        )

    # Add properties to the entry.
    AtomliteDatabase(database_path).add_properties(
        key=target_key,
        property_dict=properties,
    )
