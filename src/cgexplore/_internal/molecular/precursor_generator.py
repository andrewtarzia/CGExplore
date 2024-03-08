# Distributed under the terms of the MIT License.

"""Classes of topologies of precursors."""

import itertools as it
from collections import abc
from dataclasses import dataclass

import networkx as nx
import numpy as np
import stk
import vabene as vb
from rdkit import RDLogger

from .beads import CgBead, string_to_atom_number
from .utilities import get_rotation, vnorm_r

RDLogger.DisableLog("rdApp.*")


@dataclass
class GeneratedPrecursor:
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]
    molecule: stk.BuildingBlock
    db_key: str
    composition: str

    def __post_init__(self) -> None:
        new_fgs = tuple(
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{self.binder_beads[i].element_string}]"
                    f"[{self.placer_beads[j].element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            )
            for i, j in it.product(
                range(len(self.binder_beads)), range(len(self.placer_beads))
            )
        )
        self.building_block = stk.BuildingBlock.init_from_molecule(
            molecule=self.molecule,
            functional_groups=new_fgs,
        )

    def get_smiles(self) -> str:
        return stk.Smiles().get_key(self.building_block)

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


def check_fit(
    chromosome: tuple[int, ...],
    num_beads: int,
    max_shell: int,
) -> bool:
    """Check if chromosome has an allowed topology."""
    if sum(chromosome) != num_beads:
        return False

    idx = chromosome[0]
    fit = True
    sum_g = np.sum(chromosome[:idx]).astype(int)
    while fit and sum_g < num_beads:
        check_chr = False
        for x in range(idx, sum_g + 1):
            if chromosome[x] != 0:
                check_chr = True
        if not check_chr and sum_g < num_beads:
            fit = False
        else:
            if chromosome[idx] != 0:
                idx += chromosome[idx]
            else:
                idx += 1
            sum_g = np.sum(chromosome[:idx])
    if fit:
        for c in chromosome:
            if c > max_shell:
                fit = False
                break
    return fit


@dataclass
class PrecursorGenerator:
    """Generate custom Precursor based on a composition tuple.

    Define the link from composition to structure:
    """

    composition: tuple[int, ...]
    present_beads: tuple[CgBead, ...]
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]
    bead_distance: float = 4.7

    def _define_graph(self) -> list[tuple]:
        graph = nx.Graph()
        graph.add_node(0)
        node = 1
        edges = []
        for i in range(len(self.composition)):
            for _ in range(self.composition[i]):
                edges.append((i, node))
                graph.add_edge(i, node)
                node += 1

        self.graph = graph
        return edges

    def _get_clusters(self, edges: list[tuple]) -> list[list]:
        tmp_connections = []
        for connection in edges:
            for node in connection:
                if (node in k for k in edges):
                    tmp_connections.append([node, connection])  # noqa: PERF401

        cl_list = []
        for i in range(len(edges) + 1):
            k = [elem[1] for elem in tmp_connections if elem[0] == i]
            cl_list.append(k)

        t = [
            set([val for sublist in elem for val in sublist])  # noqa: C403
            for elem in cl_list
        ]

        cluster = []
        for o, link in enumerate(t):
            li = [m for m in link]  # noqa: C416

            li.remove(o)
            cluster.append([o, li])
        return cluster

    def _place_beads(  # noqa: PLR0913
        self,
        coordinates: np.ndarray,
        num_beads: int,
        center_idx: int,
        cluster: list,
        count_beads: int,
    ) -> np.ndarray:
        ndx = count_beads + 1

        # loop for the first bead to be added
        if center_idx == 0:
            rad = 2 * np.pi / num_beads
            rot_tmp = rad
            for _ in range(num_beads):
                u_tmp = np.dot(get_rotation(rot_tmp), (self.bead_distance, 0))
                coordinates[ndx] = u_tmp + coordinates[center_idx]
                rot_tmp += rad
                ndx += 1
        # attach only one bead
        elif num_beads == 1 and center_idx > 0:
            rad = 2 * np.pi
            v = coordinates[center_idx] - coordinates[np.min(cluster[1])]
            u = vnorm_r(np.dot(get_rotation(rad), v), self.bead_distance)
            coordinates[ndx] = u + coordinates[center_idx]
        # loop for a multiple bead addition
        else:
            rad = -2 * np.pi / (num_beads + 1)
            rot_tmp = rad
            for _ in range(num_beads):
                v_tmp = (
                    coordinates[center_idx] - coordinates[np.min(cluster[1])]
                )
                u_tmp = vnorm_r(
                    np.dot(get_rotation(rot_tmp + np.pi), v_tmp),
                    self.bead_distance,
                )
                coordinates[ndx] = u_tmp + coordinates[center_idx]
                rot_tmp += rad
                ndx += 1
        return coordinates

    def _get_coordinates(self, clusters: list[list]) -> np.ndarray:
        coordinates = np.zeros((sum(self.composition) + 1, 2))
        count = 0
        for i, num_beads in enumerate(self.composition):
            coordinates = self._place_beads(
                coordinates=coordinates,
                num_beads=num_beads,
                center_idx=i,
                cluster=clusters[i],
                count_beads=count,
            )
            count += num_beads

        return coordinates

    def __post_init__(self) -> None:
        edges = self._define_graph()
        clusters = self._get_clusters(edges)
        coordinates = self._get_coordinates(clusters)
        coordinates = np.array(
            [np.array([i[0], i[1], 0]) for i in coordinates]
        )

        atoms = [
            stk.Atom(
                i, string_to_atom_number(self.present_beads[i].element_string)
            )
            for i in self.graph.nodes
        ]
        bonds = []
        bonded = set()
        for cluster in clusters:
            a1id = cluster[0]
            for a2id in cluster[1]:
                bond_pair = tuple(sorted((a1id, a2id)))
                if bond_pair not in bonded:
                    bonds.append(stk.Bond(atoms[a1id], atoms[a2id], order=1))
                    bonded.add(bond_pair)

        model = stk.BuildingBlock.init(
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            position_matrix=coordinates,
        )

        new_fgs = tuple(
            stk.SmartsFunctionalGroupFactory(
                smarts=(
                    f"[{self.binder_beads[i].element_string}]"
                    f"[{self.placer_beads[j].element_string}]"
                ),
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            )
            for i, j in it.product(
                range(len(self.binder_beads)), range(len(self.placer_beads))
            )
        )
        self.building_block = stk.BuildingBlock.init_from_molecule(
            molecule=model,
            functional_groups=new_fgs,
        )

    def get_smiles(self) -> str:
        return stk.Smiles().get_key(self.building_block)

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


@dataclass
class VaBene:
    present_beads: tuple[CgBead, ...]
    building_block: stk.BuildingBlock

    def get_smiles(self) -> str:
        return stk.Smiles().get_key(self.building_block)

    def get_building_block(self) -> stk.BuildingBlock:
        return self.building_block


@dataclass
class VaBeneGenerator:
    """Generate custom Precursor based on vabene algorithm."""

    present_beads: tuple[CgBead, ...]
    num_beads: int
    seed: int
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]
    scale: float

    def _remove_hydrogens(
        self,
        molecule: stk.BuildingBlock,
    ) -> stk.BuildingBlock:
        new_atoms: list[stk.Atom] = []
        atom_id_map = {}
        for atom in molecule.get_atoms():
            if atom.get_atomic_number() == 1:
                continue
            atom_id_map[atom.get_id()] = len(new_atoms)
            new_atoms.append(
                stk.Atom(
                    id=atom_id_map[atom.get_id()],
                    atomic_number=atom.get_atomic_number(),
                    charge=atom.get_charge(),
                )
            )

        new_bonds = [
            stk.Bond(
                atom1=new_atoms[bond.get_atom1().get_id()],
                atom2=new_atoms[bond.get_atom2().get_id()],
                order=1,
            )
            for bond in molecule.get_bonds()
            if bond.get_atom1().get_id() in atom_id_map
            and bond.get_atom2().get_id() in atom_id_map
        ]
        new_position_matrix = [
            molecule.get_position_matrix()[i] for i in atom_id_map
        ]
        return stk.BuildingBlock.init(
            atoms=tuple(new_atoms),
            bonds=tuple(new_bonds),
            position_matrix=np.array(new_position_matrix) * self.scale,
        )

    def __post_init__(self) -> None:
        vb_atoms = tuple(
            vb.Atom(
                atomic_number=string_to_atom_number(i.element_string),
                charge=0,
                max_valence=i.coordination,
            )
            for i in self.present_beads
        )

        self.atom_factory = vb.RandomAtomFactory(
            atoms=vb_atoms,
            num_atoms=self.num_beads,
            random_seed=self.seed,
        )

        self.bond_factory = vb.RandomBondFactory(
            random_seed=self.seed,
            max_bond_order=1,
        )

    def get_precursors(self, num_precursors: int) -> abc.Iterable[VaBene]:
        for _ in range(num_precursors):
            atoms = tuple(self.atom_factory.get_atoms())
            bonds = self.bond_factory.get_bonds(atoms)
            molecule = vb.Molecule(atoms, bonds)
            try:
                stk_molecule = stk.BuildingBlock.init_from_vabene_molecule(
                    molecule
                )
            except RuntimeError:
                # Skip failed ETKDG inputs.
                continue

            stk_molecule = self._remove_hydrogens(stk_molecule)

            new_fgs = tuple(
                stk.SmartsFunctionalGroupFactory(
                    smarts=(
                        f"[{self.binder_beads[i].element_string}]"
                        f"[{self.placer_beads[j].element_string}]"
                    ),
                    bonders=(0,),
                    deleters=(),
                    placers=(0, 1),
                )
                for i, j in it.product(
                    range(len(self.binder_beads)),
                    range(len(self.placer_beads)),
                )
            )
            building_block = stk.BuildingBlock.init_from_molecule(
                molecule=stk_molecule,
                functional_groups=new_fgs,
            )
            yield VaBene(
                building_block=building_block,
                present_beads=self.present_beads,
            )
