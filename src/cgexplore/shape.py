# Distributed under the terms of the MIT License.

"""Module for shape analysis.

Author: Andrew Tarzia

"""

import logging
import os
import pathlib
import shutil
import subprocess as sp

import numpy as np
import stk

from .beads import periodic_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def test_shape_mol(
    topo_expected: dict[str, int],
    atoms: list,
    name: str,
    topo_str: str,
) -> None:
    """Test shape molecule.

    Requires hard-coded assumptions.
    I suggest using the method in `ShapeMeasure`.
    """
    num_atoms = len(atoms)
    if num_atoms != topo_expected[topo_str]:
        msg = (
            f"{topo_str} needs {topo_expected[topo_str]} atoms, not "
            f"{num_atoms}; name={name}"
        )
        raise ValueError(msg)


def fill_position_matrix_molecule(
    molecule: stk.Molecule,
    element: str,
    old_position_matrix: np.ndarray,
) -> tuple[list, list]:
    """Make position matrix from filtering molecule based on element."""
    position_matrix = []
    atoms: list[stk.Atom] = []

    target_anum = periodic_table()[element]
    for atom in molecule.get_atoms():
        atomic_number = atom.get_atomic_number()  # type: ignore[union-attr]
        if atomic_number == target_anum:
            new_atom = stk.Atom(
                id=len(atoms),
                atomic_number=atom.get_atomic_number(),
                charge=atom.get_charge(),
            )
            atoms.append(new_atom)
            position_matrix.append(old_position_matrix[atom.get_id()])

    return position_matrix, atoms


def get_shape_molecule_byelement(
    molecule: stk.Molecule,
    name: str,
    element: str,
    topo_expected: dict[str, int],
) -> stk.Molecule | None:
    """Get shape molecule.

    Requires hard-coded assumptions.
    I suggest using the method in `ShapeMeasure`.
    """
    splits = name.split("_")
    topo_str = splits[0]
    if topo_str not in topo_expected:
        return None

    old_position_matrix = molecule.get_position_matrix()

    position_matrix, atoms = fill_position_matrix_molecule(
        molecule=molecule,
        element=element,
        old_position_matrix=old_position_matrix,
    )

    test_shape_mol(topo_expected, atoms, name, topo_str)
    return stk.BuildingBlock.init(
        atoms=atoms,
        bonds=(),
        position_matrix=np.array(position_matrix),
    )


def fill_position_matrix(
    constructed_molecule: stk.ConstructedMolecule,
    target_bb: stk.Molecule,
    element: str,
    old_position_matrix: np.ndarray,
) -> tuple[list, list]:
    """Make position matrix from filtering molecule based on target BB."""
    position_matrix = []
    atoms: list[stk.Atom] = []

    target_anum = periodic_table()[element]
    for ai in constructed_molecule.get_atom_infos():
        if (
            ai.get_building_block() == target_bb
            and ai.get_building_block_atom() is not None
        ):
            ai_atom = ai.get_building_block_atom()
            ai_atomic_number = ai_atom.get_atomic_number()  # type: ignore[union-attr]
            if ai_atomic_number == target_anum:
                a = ai.get_atom()
                new_atom = stk.Atom(
                    id=len(atoms),
                    atomic_number=a.get_atomic_number(),
                    charge=a.get_charge(),
                )
                atoms.append(new_atom)
                position_matrix.append(old_position_matrix[a.get_id()])

    return position_matrix, atoms


def get_shape_molecule_nodes(
    constructed_molecule: stk.ConstructedMolecule,
    name: str,
    element: str,
    topo_expected: dict[str, int],
) -> stk.Molecule | None:
    """Get shape molecule.

    Requires hard-coded assumptions.
    I suggest using the method in `ShapeMeasure`.
    """
    splits = name.split("_")
    topo_str = splits[0]
    if topo_str not in topo_expected:
        return None
    bbs = list(constructed_molecule.get_building_blocks())
    old_position_matrix = constructed_molecule.get_position_matrix()

    large_c_bb = bbs[0]
    position_matrix, atoms = fill_position_matrix(
        constructed_molecule=constructed_molecule,
        target_bb=large_c_bb,
        element=element,
        old_position_matrix=old_position_matrix,
    )

    test_shape_mol(topo_expected, atoms, name, topo_str)
    return stk.BuildingBlock.init(
        atoms=atoms,
        bonds=(),
        position_matrix=np.array(position_matrix),
    )


def get_shape_molecule_ligands(
    constructed_molecule: stk.ConstructedMolecule,
    name: str,
    element: str,
    topo_expected: dict[str, int],
) -> stk.Molecule | None:
    """Get shape molecule.

    Requires hard-coded assumptions.
    I suggest using the method in `ShapeMeasure`.
    """
    splits = name.split("_")
    topo_str = splits[0]
    bbs = list(constructed_molecule.get_building_blocks())
    old_position_matrix = constructed_molecule.get_position_matrix()

    if topo_str not in topo_expected:
        return None

    two_c_bb = bbs[1]
    position_matrix, atoms = fill_position_matrix(
        constructed_molecule=constructed_molecule,
        target_bb=two_c_bb,
        element=element,
        old_position_matrix=old_position_matrix,
    )

    test_shape_mol(topo_expected, atoms, name, topo_str)
    return stk.BuildingBlock.init(
        atoms=atoms,
        bonds=(),
        position_matrix=np.array(position_matrix),
    )


class ShapeMeasure:
    """Uses Shape [1]_ to calculate the shape of coordinates.

    References:
        .. [1] http://www.ee.ub.edu/

    """

    def __init__(
        self,
        output_dir: pathlib.Path | str,
        shape_path: pathlib.Path | str,
        shape_string: str | None = None,
    ) -> None:
        """Initialize shape measure."""
        self._output_dir = output_dir
        self._shape_path = shape_path
        self._shape_dict: dict[str, dict]
        if shape_string is None:
            self._shape_dict = self.reference_shape_dict()
        else:
            self._shape_dict = {
                shape_string: self.reference_shape_dict()[shape_string]
            }
        self._num_vertex_options = tuple(
            {int(self._shape_dict[i]["vertices"]) for i in self._shape_dict}
        )

    def _test_shape_mol(self, expected_points: int, atoms: list) -> None:
        num_atoms = len(atoms)
        if num_atoms != expected_points:
            msg = (
                f"Expected num points not met ({expected_points}); has "
                f"{num_atoms}"
            )
            raise ValueError(msg)

    def get_shape_molecule_byelement(
        self,
        molecule: stk.Molecule,
        element: str,
        expected_points: int,
    ) -> stk.Molecule | None:
        """Get molecule to analyse by filtering by element."""
        old_position_matrix = molecule.get_position_matrix()

        position_matrix, atoms = fill_position_matrix_molecule(
            molecule=molecule,
            element=element,
            old_position_matrix=old_position_matrix,
        )

        self._test_shape_mol(expected_points, atoms)
        return stk.BuildingBlock.init(
            atoms=atoms,
            bonds=(),
            position_matrix=np.array(position_matrix),
        )

    def reference_shape_dict(self) -> dict[str, dict]:
        """Reference shapes as dictionary."""
        return {
            "L-2": {
                "code": "1",
                "label": "L-2",
                "vertices": "2",
                "shape": "Linear D∞h",
            },
            "vT-2": {
                "code": "2",
                "label": "vT-2",
                "vertices": "2",
                "shape": "Divacant tetrahedron (V-shape, 109.47º) C2v",
            },
            "vOC-2": {
                "code": "3",
                "label": "vOC-2",
                "vertices": "2",
                "shape": "Tetravacant octahedron (L-shape, 90º) C2v",
            },
            "TP-3": {
                "vertices": "3",
                "code": "1",
                "label": "TP-3",
                "shape": "Trigonal planar D3h",
            },
            "vT-3": {
                "vertices": "3",
                "code": "2",
                "label": "vT-3",
                "shape": "Pyramid‡ (vacant tetrahedron) C3v",
            },
            "fac-vOC-3": {
                "vertices": "3",
                "code": "3",
                "label": "fac-vOC-3",
                "shape": "fac-Trivacant octahedron C3v",
            },
            "mer-vOC-3": {
                "vertices": "3",
                "code": "4",
                "label": "mer-vOC-3",
                "shape": "mer-Trivacant octahedron (T-shape) C2v",
            },
            "SP-4": {
                "code": "1",
                "label": "SP-4",
                "vertices": "4",
                "shape": "Square D4h",
            },
            "T-4": {
                "code": "2",
                "label": "T-4",
                "vertices": "4",
                "shape": "Tetrahedron Td",
            },
            "SS-4": {
                "code": "3",
                "label": "SS-4",
                "vertices": "4",
                "shape": "Seesaw or sawhorse‡ (cis-divacant octahedron) C2v",
            },
            "vTBPY-4": {
                "code": "4",
                "label": "vTBPY-4",
                "vertices": "4",
                "shape": "Axially vacant trigonal bipyramid C3v",
            },
            "PP-5": {
                "code": "1",
                "vertices": "5",
                "label": "PP-5",
                "shape": "Pentagon D5h",
            },
            "vOC-5": {
                "code": "2",
                "vertices": "5",
                "label": "vOC-5",
                "shape": "Vacant octahedron‡ (Johnson square pyramid, J1) C4v",
            },
            "TBPY-5": {
                "code": "3",
                "vertices": "5",
                "label": "TBPY-5",
                "shape": "Trigonal bipyramid D3h",
            },
            "SPY-5": {
                "code": "4",
                "vertices": "5",
                "label": "SPY-5",
                "shape": "Square pyramid § C4v",
            },
            "JTBPY-5": {
                "code": "5",
                "vertices": "5",
                "label": "JTBPY-5",
                "shape": "Johnson trigonal bipyramid (J12) D3h",
            },
            "HP-6": {
                "code": "1",
                "label": "HP-6",
                "vertices": "6",
                "shape": "Hexagon D6h",
            },
            "PPY-6": {
                "code": "2",
                "label": "PPY-6",
                "vertices": "6",
                "shape": "Pentagonal pyramid C5v",
            },
            "OC-6": {
                "code": "3",
                "label": "OC-6",
                "vertices": "6",
                "shape": "Octahedron Oh",
            },
            "TPR-6": {
                "code": "4",
                "label": "TPR-6",
                "vertices": "6",
                "shape": "Trigonal prism D3h",
            },
            "JPPY-5": {
                "code": "5",
                "label": "JPPY-5",
                "vertices": "6",
                "shape": "Johnson pentagonal pyramid (J2) C5v",
            },
            "HP-7": {
                "code": "1",
                "vertices": "7",
                "label": "HP-7",
                "shape": "Heptagon D7h",
            },
            "HPY-7": {
                "code": "2",
                "vertices": "7",
                "label": "HPY-7",
                "shape": "Hexagonal pyramid C6v",
            },
            "PBPY-7": {
                "code": "3",
                "vertices": "7",
                "label": "PBPY-7",
                "shape": "Pentagonal bipyramid D5h",
            },
            "COC-7": {
                "code": "4",
                "vertices": "7",
                "label": "COC-7",
                "shape": "Capped octahedron * C3v",
            },
            "CTPR-7": {
                "code": "5",
                "vertices": "7",
                "label": "CTPR-7",
                "shape": "Capped trigonal prism * C2v",
            },
            "JPBPY-7": {
                "code": "6",
                "vertices": "7",
                "label": "JPBPY-7",
                "shape": "Johnson pentagonal bipyramid (J13) D5h",
            },
            "JETPY-7": {
                "code": "7",
                "vertices": "7",
                "label": "JETPY-7",
                "shape": "Elongated triangular pyramid (J7) C3v",
            },
            "OP-8": {
                "code": "1",
                "label": "OP-8",
                "vertices": "8",
                "shape": "Octagon D8h",
            },
            "HPY-8": {
                "code": "2",
                "label": "HPY-8",
                "vertices": "8",
                "shape": "Heptagonal pyramid C7v",
            },
            "HBPY-8": {
                "code": "3",
                "label": "HBPY-8",
                "vertices": "8",
                "shape": "Hexagonal bipyramid D6h",
            },
            "CU-8": {
                "code": "4",
                "label": "CU-8",
                "vertices": "8",
                "shape": "Cube Oh",
            },
            "SAPR-8": {
                "code": "5",
                "label": "SAPR-8",
                "vertices": "8",
                "shape": "Square antiprism D4d",
            },
            "TDD-8": {
                "code": "6",
                "label": "TDD-8",
                "vertices": "8",
                "shape": "Triangular dodecahedron D2d",
            },
            "JGBF-8": {
                "code": "7",
                "label": "JGBF-8",
                "vertices": "8",
                "shape": "Johnson - Gyrobifastigium (J26) D2d",
            },
            "JETBPY-8": {
                "code": "8",
                "label": "JETBPY-8",
                "vertices": "8",
                "shape": "Johnson - Elongated triangular bipyramid (J14) D3h",
            },
            "JBTP-8": {
                "code": "9",
                "label": "JBTP-8",
                "vertices": "8",
                "shape": "Johnson - Biaugmented trigonal prism (J50) C2v",
            },
            "BTPR-8": {
                "code": "10",
                "label": "BTPR-8",
                "vertices": "8",
                "shape": "Biaugmented trigonal prism C2v",
            },
            "JSD-8": {
                "code": "11",
                "label": "JSD-8",
                "vertices": "8",
                "shape": "Snub disphenoid (J84) D2d",
            },
            "TT-8": {
                "code": "12",
                "label": "TT-8",
                "vertices": "8",
                "shape": "Triakis tetrahedron Td",
            },
            "ETBPY-8": {
                "code": "13",
                "label": "ETBPY-8",
                "vertices": "8",
                "shape": "Elongated trigonal bipyramid (see 8) D3h",
            },
            "EP-9": {
                "code": "1",
                "vertices": "9",
                "label": "EP-9",
                "shape": "Enneagon D9h",
            },
            "OPY-9": {
                "code": "2",
                "vertices": "9",
                "label": "OPY-9",
                "shape": "Octagonal pyramid C8v",
            },
            "HBPY-9": {
                "code": "3",
                "vertices": "9",
                "label": "HBPY-9",
                "shape": "Heptagonal bipyramid D7h",
            },
            "JTC-9": {
                "code": "4",
                "vertices": "9",
                "label": "JTC-9",
                "shape": (
                    "Triangular cupola (J3) = trivacant cuboctahedron " "C3v"
                ),
            },
            "JCCU-9": {
                "code": "5",
                "vertices": "9",
                "label": "JCCU-9",
                "shape": ("Capped cube (Elongated square pyramid, J8) C4v"),
            },
            "CCU-9": {
                "code": "6",
                "vertices": "9",
                "label": "CCU-9",
                "shape": "Capped cube C4v",
            },
            "JCSAPR-9": {
                "code": "7",
                "vertices": "9",
                "label": "JCSAPR-9",
                "shape": (
                    "Capped sq. antiprism (Gyroelongated square "
                    "pyramid J10) C4v"
                ),
            },
            "CSAPR-9": {
                "code": "8",
                "vertices": "9",
                "label": "CSAPR-9",
                "shape": "Capped square antiprism C4v",
            },
            "JTCTPR-9": {
                "code": "9",
                "vertices": "9",
                "label": "JTCTPR-9",
                "shape": "Tricapped trigonal prism (J51) D3h",
            },
            "TCTPR-9": {
                "code": "10",
                "vertices": "9",
                "label": "TCTPR-9",
                "shape": "Tricapped trigonal prism D3h",
            },
            "JTDIC-9": {
                "code": "11",
                "vertices": "9",
                "label": "JTDIC-9",
                "shape": "Tridiminished icosahedron (J63) C3v",
            },
            "HH-9": {
                "code": "12",
                "vertices": "9",
                "label": "HH-9",
                "shape": "Hula-hoop C2v",
            },
            "MFF-9": {
                "code": "13",
                "vertices": "9",
                "label": "MFF-9",
                "shape": "Muffin Cs",
            },
            "DP-10": {
                "code": "1",
                "vertices": "10",
                "label": "DP-10",
                "shape": "Decagon D10h",
            },
            "EPY-10": {
                "code": "2",
                "vertices": "10",
                "label": "EPY-10",
                "shape": "Enneagonal pyramid C9v",
            },
            "OBPY-10": {
                "code": "3",
                "vertices": "10",
                "label": "OBPY-10",
                "shape": "Octagonal bipyramid D8h",
            },
            "PPR-10": {
                "code": "4",
                "vertices": "10",
                "label": "PPR-10",
                "shape": "Pentagonal prism D5h",
            },
            "PAPR-10": {
                "code": "5",
                "vertices": "10",
                "label": "PAPR-10",
                "shape": "Pentagonal antiprism D5d",
            },
            "JBCCU-10": {
                "code": "6",
                "vertices": "10",
                "label": "JBCCU-10",
                "shape": (
                    "Bicapped cube (Elongated square bipyramid J15) D4h"
                ),
            },
            "JBCSAPR-10": {
                "code": "7",
                "vertices": "10",
                "label": "JBCSAPR-10",
                "shape": (
                    "Bicapped square antiprism (Gyroelongated square "
                    "bipyramid J17) D4d"
                ),
            },
            "JMBIC-10": {
                "code": "8",
                "vertices": "10",
                "label": "JMBIC-10",
                "shape": "Metabidiminished icosahedron (J62) C2v",
            },
            "JATDI-10": {
                "code": "9",
                "vertices": "10",
                "label": "JATDI-10",
                "shape": "Augmented tridiminished icosahedron (J64) C3v",
            },
            "JSPC-10": {
                "code": "10",
                "vertices": "10",
                "label": "JSPC-10",
                "shape": "Sphenocorona (J87) C2v",
            },
            "SDD-10": {
                "code": "11",
                "vertices": "10",
                "label": "SDD-10",
                "shape": "Staggered dodecahedron (2:6:2) # D2",
            },
            "TD-10": {
                "code": "12",
                "vertices": "10",
                "label": "TD-10",
                "shape": "Tetradecahedron (2:6:2) C2v",
            },
            "HD-10": {
                "code": "13",
                "vertices": "10",
                "label": "HD-10",
                "shape": "Hexadecahedron (2:6:2, or 1:4:4:1) D4h",
            },
            "HP-11": {
                "code": "1",
                "vertices": "11",
                "label": "HP-11",
                "shape": "Hendecagon D11h",
            },
            "DPY-11": {
                "code": "2",
                "vertices": "11",
                "label": "DPY-11",
                "shape": "Decagonal pyramid C10v",
            },
            "EBPY-11": {
                "code": "3",
                "vertices": "11",
                "label": "EBPY-11",
                "shape": "Enneagonal bipyramid D9h",
            },
            "JCPPR-11": {
                "code": "4",
                "vertices": "11",
                "label": "JCPPR-11",
                "shape": (
                    "Capped pent. Prism (Elongated pentagonal pyramid "
                    "J9) C5v"
                ),
            },
            "JCPAPR-11": {
                "code": "5",
                "vertices": "11",
                "label": "JCPAPR-11",
                "shape": (
                    "Capped pent. antiprism (Gyroelongated pentagonal "
                    "pyramid J11) C5v"
                ),
            },
            "JAPPR-11": {
                "code": "6",
                "vertices": "11",
                "label": "JAPPR-11",
                "shape": "Augmented pentagonal prism (J52) C2v",
            },
            "JASPC-11": {
                "code": "7",
                "vertices": "11",
                "label": "JASPC-11",
                "shape": "Augmented sphenocorona (J87) Cs",
            },
            "DP-12": {
                "code": "1",
                "vertices": "12",
                "label": "DP-12",
                "shape": "Dodecagon D12h",
            },
            "HPY-12": {
                "code": "2",
                "vertices": "12",
                "label": "HPY-12",
                "shape": "Hendecagonal pyramid C11v",
            },
            "DBPY-12": {
                "code": "3",
                "vertices": "12",
                "label": "DBPY-12",
                "shape": "Decagonal bipyramid D10h",
            },
            "HPR-12": {
                "code": "4",
                "vertices": "12",
                "label": "HPR-12",
                "shape": "Hexagonal prism D6h",
            },
            "HAPR-12": {
                "code": "5",
                "vertices": "12",
                "label": "HAPR-12",
                "shape": "Hexagonal antiprism D6d",
            },
            "TT-12": {
                "code": "6",
                "vertices": "12",
                "label": "TT-12",
                "shape": "Truncated tetrahedron Td",
            },
            "COC-12": {
                "code": "7",
                "vertices": "12",
                "label": "COC-12",
                "shape": "Cuboctahedron Oh",
            },
            "ACOC-12": {
                "code": "8",
                "vertices": "12",
                "label": "ACOC-12",
                "shape": (
                    "Anticuboctahedron (Triangular orthobicupola J27) " "D3h"
                ),
            },
            "IC-12": {
                "code": "9",
                "vertices": "12",
                "label": "IC-12",
                "shape": "Icosahedron Ih",
            },
            "JSC-12": {
                "code": "10",
                "vertices": "12",
                "label": "JSC-12",
                "shape": "Square cupola (J4) C4v",
            },
            "JEPBPY-12": {
                "code": "11",
                "vertices": "12",
                "label": "JEPBPY-12",
                "shape": "Elongated pentagonal bipyramid (J16) D6h",
            },
            "JBAPPR-12": {
                "code": "12",
                "vertices": "12",
                "label": "JBAPPR-12",
                "shape": "Biaugmented pentagonal prism (J53) C2v",
            },
            "JSPMC-12": {
                "code": "13",
                "vertices": "12",
                "label": "JSPMC-12",
                "shape": "Sphenomegacorona (J88) Cs",
            },
            "DD-20": {
                "code": "1",
                "vertices": "20",
                "label": "DD-20",
                "shape": "Dodecahedron † Ih",
            },
            "TCU-24": {
                "code": "1",
                "vertices": "24",
                "label": "TCU-24",
                "shape": "Truncated cube Oh",
            },
            "TOC-24": {
                "code": "2",
                "vertices": "24",
                "label": "TOC-24",
                "shape": "Truncated octahedron Oh",
            },
        }

    def _collect_all_shape_values(self, output_file: str) -> dict:
        """Collect shape values from output."""
        with open(output_file) as f:
            lines = f.readlines()

        label_idx_map = {}
        values = None
        for line in reversed(lines):
            if "Structure" in line:
                values = [
                    i.strip()
                    for i in line.rstrip().split("]")[1].split(" ")
                    if i.strip()
                ]
                for idx, symb in enumerate(values):
                    label_idx_map[symb] = idx
                break
            float_values = [i.strip() for i in line.rstrip().split(",")]

        if values is None:
            logging.info("no shapes found due to overlapping atoms")
            shapes = {}
        else:
            shapes = {
                i: float(float_values[1 + label_idx_map[i]])
                for i in label_idx_map
            }

        return shapes

    def _write_input_file(
        self,
        input_file: str,
        structure_string: str,
    ) -> None:
        """Write input file for shape."""
        num_vertices = len(structure_string.split("\n")) - 2

        possible_shapes = self._get_possible_shapes(num_vertices)
        shape_numbers = tuple(i["code"] for i in possible_shapes)

        title = "$shape run by Andrew Tarzia - central atom=0 always.\n"
        fix_perm = (
            "%fixperm 0\\n" if num_vertices == 12 else "\n"  # noqa: PLR2004
        )
        size_of_poly = f"{num_vertices} 0\n"
        codes = " ".join(shape_numbers) + "\n"

        string = title + fix_perm + size_of_poly + codes + structure_string

        with open(input_file, "w") as f:
            f.write(string)

    def _run_calculation(self, structure_string: str) -> dict:
        """Calculate the shape of a molecule."""
        input_file = "shp.dat"
        std_out = "shp.out"
        output_file = "shp.tab"

        self._write_input_file(
            input_file=input_file,
            structure_string=structure_string,
        )

        # Note that sp.call will hold the program until completion
        # of the calculation.
        captured_output = sp.run(
            [f"{self._shape_path}", f"{input_file}"],  # noqa: S603
            stdin=sp.PIPE,
            capture_output=True,
            check=True,
            # Shell is required to run complex arguments.
        )

        with open(std_out, "w") as f:
            f.write(str(captured_output.stdout))

        return self._collect_all_shape_values(output_file)

    def _get_centroids(
        self,
        molecule: stk.ConstructedMolecule,
        target_atmnums: tuple[int],
    ) -> list[np.ndarray]:
        bb_ids: dict[int | None, list] = {}
        for ai in molecule.get_atom_infos():
            aibbid = ai.get_building_block_id()
            if ai.get_atom().get_atomic_number() in target_atmnums:
                if aibbid not in bb_ids:
                    bb_ids[aibbid] = []
                bb_ids[aibbid].append(ai.get_atom().get_id())

        centroids = [molecule.get_centroid(atom_ids=bb_ids[n]) for n in bb_ids]

        with open("cents.xyz", "w") as f:
            f.write(f"{len(centroids)}\n\n")
            for c in centroids:
                f.write(f"Zn {c[0]} {c[1]} {c[2]}\n")

        return centroids

    def _get_possible_shapes(self, num_vertices: int) -> tuple[dict, ...]:
        ref_dict = self.reference_shape_dict()
        return tuple(
            ref_dict[i]
            for i in ref_dict
            if int(ref_dict[i]["vertices"]) == num_vertices
        )

    def calculate(self, molecule: stk.Molecule) -> dict:
        """Calculate shape measures for a molecule."""
        output_dir = pathlib.Path(self._output_dir).resolve()

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()

        init_dir = pathlib.Path.cwd()
        try:
            os.chdir(output_dir)
            structure_string = "shape run by AT\n"
            num_centroids = 0
            pos_mat = molecule.get_position_matrix()
            for a in molecule.get_atoms():
                c = pos_mat[a.get_id()]
                structure_string += (
                    f"{a.__class__.__name__} {c[0]} {c[1]} {c[2]}\n"
                )
                num_centroids += 1

            if num_centroids not in self._num_vertex_options:
                msg = (
                    f"you gave {num_centroids} vertices, but expected to "
                    f"calculate shapes with {self._num_vertex_options} options"
                )
                raise ValueError(msg)

            shapes = self._run_calculation(structure_string)

        finally:
            os.chdir(init_dir)

        return shapes

    def calculate_from_centroids(
        self,
        constructed_molecule: stk.ConstructedMolecule,
        target_atmnums: tuple[int],
    ) -> dict:
        """Calculate shape from the centroids of building blocks.

        Currently not implemented well.
        """
        output_dir = pathlib.Path(self._output_dir).resolve()

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()

        init_dir = pathlib.Path.cwd()
        try:
            os.chdir(output_dir)
            centroids = self._get_centroids(
                constructed_molecule, target_atmnums
            )
            structure_string = "shape run by AT\n"
            num_centroids = 0
            for c in centroids:
                structure_string += f"Zn {c[0]} {c[1]} {c[2]}\n"
                num_centroids += 1

            if num_centroids not in self._num_vertex_options:
                msg = (
                    f"you gave {num_centroids} vertices, but expected to "
                    f"calculate shapes with {self._num_vertex_options} options"
                )
                raise ValueError(msg)

            shapes = self._run_calculation(structure_string)

        finally:
            os.chdir(init_dir)

        return shapes


def known_shape_vectors() -> dict[str, dict]:
    """Printed from shape_map.py output.

    May not include all of them, only included ones I deemed relevant.

    """
    return {
        "TP-3": {
            "TP-3": 0.0,
            "vT-3": 0.0,
            "fvOC-3": 0.0,
            "mvOC-3": 6.699,
        },
        "SP-4": {
            "SP-4": 0.0,
            "T-4": 33.333,
            "SS-4": 16.737,
            "vTBPY-4": 34.007,
        },
        "T-4": {
            "SP-4": 33.333,
            "T-4": 0.0,
            "SS-4": 7.213,
            "vTBPY-4": 2.288,
        },
        "TBPY-5": {
            "PP-5": 37.069,
            "vOC-5": 6.699,
            "TBPY-5": 0.0,
            "SPY-5": 5.384,
            "JTBPY-5": 2.941,
        },
        "TPR-6": {
            "HP-6": 33.675,
            "PPY-6": 16.678,
            "OC-6": 16.737,
            "TPR-6": 0.0,
            "JPPY-6": 20.913,
        },
        "OC-6": {
            "HP-6": 33.333,
            "PPY-6": 30.153,
            "OC-6": 0.0,
            "TPR-6": 16.737,
            "JPPY-6": 33.916,
        },
        "CU-8": {
            "OP-8": 38.311,
            "HPY-8": 30.489,
            "HBPY-8": 8.395,
            "CU-8": 0.0,
            "SAPR-8": 10.989,
            "TDD-8": 7.952,
            "JGBF-8": 18.811,
            "JETBPY-8": 25.086,
            "JBTPR-8": 13.285,
            "BTPR-8": 12.725,
            "JSD-8": 14.257,
            "TT-8": 0.953,
            "ETBPY-8": 23.162,
        },
        "SAPR-8": {
            "OP-8": 26.12,
            "HPY-8": 24.402,
            "HBPY-8": 18.458,
            "CU-8": 10.989,
            "SAPR-8": 0.0,
            "TDD-8": 2.848,
            "JGBF-8": 17.259,
            "JETBPY-8": 28.516,
            "JBTPR-8": 2.593,
            "BTPR-8": 2.095,
            "JSD-8": 5.362,
            "TT-8": 11.838,
            "ETBPY-8": 24.048,
        },
    }
