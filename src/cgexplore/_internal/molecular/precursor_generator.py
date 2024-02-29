# Distributed under the terms of the MIT License.

"""Classes of topologies of precursors.

Author: Andrew Tarzia

"""

import itertools as it
from dataclasses import dataclass

import networkx as nx
import numpy as np
import stk

from .beads import CgBead, periodic_table
from .utilities import get_rotation, vnorm_r


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

        pt = periodic_table()
        atoms = [
            stk.Atom(i, pt[self.present_beads[i].element_string])
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
