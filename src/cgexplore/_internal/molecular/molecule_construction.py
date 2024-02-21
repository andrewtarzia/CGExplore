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


class Graph:

    def __init__(self, chromosome_list):
        self.chromosome_list = chromosome_list
        self.G = nx.Graph()

    def nodes(self):
        return self.G.nodes

    def edges(self):
        self.G.add_node(0)
        node = 1
        edges = []
        for i in range(len(self.chromosome_list)):
            for j in range(self.chromosome_list[i]):
                edges.append([i, node])
                self.G.add_edge(i, node)
                node += 1
        return edges


def get_rotation(rad):
    rad = rad
    c, s = np.cos(rad), np.sin(rad)
    return np.array(((c, -s), (s, c)))


def Vnorm_R(v, R):
    u = v / np.linalg.norm(v) * R
    return u


def place_beads(coords, n_bead, center_ndx, cluster, count):
    ndx = count + 1
    R = 4.7
    # loop for the first bead to be added
    if center_ndx == 0:
        rad = 2 * np.pi / n_bead
        rot_tmp = rad
        for bead in range(n_bead):
            u_tmp = np.dot(get_rotation(rot_tmp), (R, 0))
            coords[ndx] = u_tmp + coords[center_ndx]
            rot_tmp += rad
            ndx += 1
    # attach only one bead
    elif n_bead == 1 and center_ndx > 0:
        rad = 2 * np.pi
        v = coords[center_ndx] - coords[cluster[1][0]]
        u = Vnorm_R(np.dot(get_rotation(rad), v), R)
        coords[ndx] = u + coords[center_ndx]
    # loop for a multiple bead addition
    else:
        rad = -2 * np.pi / (n_bead + 1)
        rot_tmp = rad
        for bead in range(n_bead):
            v_tmp = coords[center_ndx] - coords[cluster[1][0]]
            u_tmp = Vnorm_R(np.dot(get_rotation(rot_tmp + np.pi), v_tmp), R)
            coords[ndx] = u_tmp + coords[center_ndx]
            rot_tmp += rad
            ndx += 1


def clusters(n):
    tmp_connections = []
    for connection in n:
        for node in connection:
            if (node in k for k in n):
                tmp_connections.append([node, connection])
    cl_list = []
    for i in range(len(n) + 1):
        k = []
        for elem in tmp_connections:
            if elem[0] == i:
                k.append(elem[1])
        cl_list.append(k)
    t = []
    for elem in cl_list:
        t.append(set([val for sublist in elem for val in sublist]))
    cluster = []
    for o, l in enumerate(t):
        li = []
        for m in l:
            li.append(m)
        li.remove(o)
        cluster.append([o, li])
    return cluster


class Coordinates:

    def __init__(self, chromosome, connections, nB=7):
        self.chromosome = chromosome
        self.connections = connections
        self.coords = np.zeros((nB, 2))
        self.count = 0

    def get(self):
        for i, bead in enumerate(self.chromosome):
            place_beads(self.coords, bead, i, self.connections[i], self.count)
            self.count += bead
        return self.coords


@dataclass
class PrecursorGenerator:
    """Generate custom Precursor."""

    composition: tuple[int, ...]
    present_beads: tuple[CgBead, ...]
    binder_beads: tuple[CgBead, ...]
    placer_beads: tuple[CgBead, ...]

    def __post_init__(self) -> None:
        graph = Graph(self.composition)
        n = graph.edges()
        cl = clusters(n)
        coordinates = Coordinates(
            self.composition, cl, sum(self.composition) + 1
        ).get()
        coordinates = np.array(
            [np.array([i[0], i[1], 0]) for i in coordinates]
        )

        pt = periodic_table()
        atoms = [
            stk.Atom(i, pt[self.present_beads[i].element_string])
            for i in graph.nodes()
        ]
        bonds = []
        bonded = set()
        for cluster in cl:
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


class Precursor:
    """Define model precursors."""

    def __init__(self) -> None:
        """Initialize a precursor."""
        self._bead_set: dict[str, CgBead]
        self._building_block: stk.BuildingBlock
        self._name: str
        raise NotImplementedError

    def get_bead_set(self) -> dict[str, CgBead]:
        """Get beads in precursor."""
        return self._bead_set

    def get_building_block(self) -> stk.BuildingBlock:
        """Get building block defined by precursor."""
        return self._building_block

    def get_name(self) -> str:
        """Get name of precursor."""
        return self._name


class FourC0Arm(Precursor):
    """A `FourC0Arm` Precursor."""

    def __init__(self, bead: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"4C0{bead.bead_type}"
        self._bead_set = {bead.bead_type: bead}
        four_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])([Br])[Br]",
            position_matrix=np.array(
                [
                    [-2, 0, -1],
                    [0, 0, 1],
                    [0, -2, -1],
                    [2, 0, 1],
                    [0, 2, 1],
                ]
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=(f"[{bead.element_string}][Br]"),
            bonders=(0,),
            deleters=(1,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=four_c_bb,
            functional_groups=(new_fgs,),
        )


class FourC1Arm(Precursor):
    """A `FourC1Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = f"4C1{bead.bead_type}{abead1.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
        }

        factories = (stk.BromoFactory(placers=(0, 1)),)
        four_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])([Br])[Br]",
            functional_groups=factories,
            position_matrix=np.array(
                [
                    [-2, 0, -1],
                    [0, 0, 1],
                    [0, -2, -1],
                    [2, 0, -1],
                    [0, 2, -1],
                ]
            ),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.small.NCore(
                core_building_block=four_c_bb,
                arm_building_blocks=stk.BuildingBlock(
                    smiles=f"[{abead1.element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=np.array([[-3, 0, 0], [0, 0, 0]]),
                ),
                repeating_unit="AAAA",
            ),
        )
        new_fgs = (
            stk.SmartsFunctionalGroupFactory(
                smarts=(f"[{bead.element_string}][{abead1.element_string}]"),
                bonders=(1,),
                deleters=(),
            ),
        )

        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=new_fgs,
        )


class ThreeC0Arm(Precursor):
    """A `ThreeC0Arm` Precursor."""

    def __init__(self, bead: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"3C0{bead.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
        }
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{bead.element_string}][Br]",
            bonders=(0,),
            deleters=(1,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=three_c_bb,
            functional_groups=(new_fgs,),
        )


class ThreeC1Arm(Precursor):
    """A `ThreeC1Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = f"3C1{bead.bead_type}{abead1.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
        }
        factories = (stk.BromoFactory(placers=(0, 1)),)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0.5],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.small.NCore(
                core_building_block=three_c_bb,
                arm_building_blocks=stk.BuildingBlock(
                    smiles=f"[{abead1.element_string}][Br]",
                    functional_groups=factories,
                    position_matrix=np.array([[-3, 0, 0], [0, 0, 0]]),
                ),
                repeating_unit="AAA",
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{bead.element_string}][{abead1.element_string}]",
            bonders=(1,),
            deleters=(),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )


class ThreeC2Arm(Precursor):
    """A `ThreeC2Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = f"3C2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
        }
        factories = (stk.BromoFactory(placers=(0, 1)),)
        three_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}]([Br])[Br]",
            functional_groups=factories,
            position_matrix=np.array(
                [
                    [-2, 0, 0],
                    [0, 0, 0],
                    [-1.2, 1, 0],
                    [-1.2, -1, 0],
                ]
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{abead2.element_string}]",
            bonders=(0,),
            deleters=(),
        )
        const_mol = stk.ConstructedMolecule(
            topology_graph=stk.small.NCore(
                core_building_block=three_c_bb,
                arm_building_blocks=stk.BuildingBlock(
                    smiles=f"[{abead1.element_string}][{abead2.element_string}]",
                    functional_groups=new_fgs,
                    position_matrix=np.array([[-3, 0, 0], [0, 0, 0]]),
                ),
                repeating_unit="AAA",
            ),
        )

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{abead2.element_string}]",
            bonders=(1,),
            deleters=(),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=const_mol,
            functional_groups=(new_fgs,),
        )


class TwoC0Arm(Precursor):
    """A `TwoC0Arm` Precursor."""

    def __init__(self, bead: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._name = f"2C0{bead.bead_type}"
        self._bead_set = {bead.bead_type: bead}
        core_c_bb = stk.BuildingBlock(
            smiles=f"[Br][{bead.element_string}][Br]",
            position_matrix=np.array([[-3, 0, 0], [0, 0, 0], [3, 0, 0]]),
        )
        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[Br][{bead.element_string}]",
            bonders=(1,),
            deleters=(0,),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock.init_from_molecule(
            molecule=core_c_bb,
            functional_groups=(new_fgs,),
        )


class TwoC1Arm(Precursor):
    """A `TwoC1Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._name = f"2C1{bead.bead_type}{abead1.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead1.element_string}][{bead.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=f"[{abead1.element_string}][{bead.element_string}][{abead1.element_string}]",
            functional_groups=new_fgs,
            position_matrix=np.array([[-3, 0, 0], [0, 0, 0], [3, 0, 0]]),
        )


class TwoC2Arm(Precursor):
    """A `TwoC2Arm` Precursor."""

    def __init__(self, bead: CgBead, abead1: CgBead, abead2: CgBead) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._name = f"2C2{bead.bead_type}{abead1.bead_type}{abead2.bead_type}"
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead2.element_string}X1][{abead1.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{abead1.element_string}]"
                f"[{abead2.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-8, 0, 0],
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                    [8, 0, 0],
                ]
            ),
        )


class TwoC3Arm(Precursor):
    """A `TwoC3Arm` Precursor."""

    def __init__(
        self,
        bead: CgBead,
        abead1: CgBead,
        abead2: CgBead,
        abead3: CgBead,
    ) -> None:
        """Initialize a precursor."""
        self._bead = bead
        self._abead1 = abead1
        self._abead2 = abead2
        self._abead3 = abead3
        self._name = (
            f"2C3{bead.bead_type}{abead1.bead_type}"
            f"{abead2.bead_type}{abead3.bead_type}"
        )
        self._bead_set = {
            bead.bead_type: bead,
            abead1.bead_type: abead1,
            abead2.bead_type: abead2,
            abead3.bead_type: abead3,
        }

        new_fgs = stk.SmartsFunctionalGroupFactory(
            smarts=f"[{abead3.element_string}X1][{abead2.element_string}]",
            bonders=(0,),
            deleters=(),
            placers=(0, 1),
        )
        self._building_block = stk.BuildingBlock(
            smiles=(
                f"[{abead3.element_string}]"
                f"[{abead2.element_string}][{abead1.element_string}]"
                f"[{bead.element_string}][{abead1.element_string}]"
                f"[{abead2.element_string}][{abead3.element_string}]"
            ),
            functional_groups=new_fgs,
            position_matrix=np.array(
                [
                    [-12, 0, 0],
                    [-8, 0, 0],
                    [-3, 0, 0],
                    [0, 0, 0],
                    [3, 0, 0],
                    [8, 0, 0],
                    [12, 0, 0],
                ]
            ),
        )
