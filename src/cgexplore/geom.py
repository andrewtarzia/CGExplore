# Distributed under the terms of the MIT License.

"""Module for geometry analysis."""

from collections import abc, defaultdict

import stk
import stko
from rdkit.Chem import AllChem

from .torsions import find_torsions
from .utilities import get_dihedral


class GeomMeasure:
    """Class to perform geometry calculations."""

    def __init__(self, target_torsions: abc.Iterable | None = None) -> None:
        """Initialize GeomMeasure."""
        self._stko_analyser = stko.molecule_analysis.GeometryAnalyser()
        if target_torsions is None:
            self._target_torsions = None
        else:
            self._target_torsions = tuple(target_torsions)

    def calculate_min_distance(
        self,
        molecule: stk.Molecule,
    ) -> dict[str, float]:
        """Calculate the minimum distance between beads and centroid."""
        return {
            "min_distance": (
                self._stko_analyser.get_min_centroid_distance(molecule)
            ),
        }

    def _get_paths(
        self,
        molecule: stk.Molecule,
        path_length: int,
    ) -> tuple[tuple[int]]:
        return AllChem.FindAllPathsOfLengthN(
            mol=molecule.to_rdkit_mol(),
            length=path_length,
            useBonds=False,
            useHs=True,
        )

    def calculate_minb2b(self, molecule: stk.Molecule) -> float:
        """Calculate the minimum distance between beads and beads."""
        return self._stko_analyser.get_min_atom_atom_distance(molecule)

    def calculate_bonds(
        self,
        molecule: stk.Molecule,
    ) -> dict[tuple[str, ...], list[float]]:
        """Calculate the bond lengths.

        Uses `stko..molecule_analysis.GeometryAnalyser`
        """
        return self._stko_analyser.calculate_bonds(molecule)

    def calculate_angles(
        self,
        molecule: stk.Molecule,
    ) -> dict[tuple[str, ...], list[float]]:
        """Calculate the angle values.

        Uses `stko..molecule_analysis.GeometryAnalyser`
        """
        return self._stko_analyser.calculate_angles(molecule)

    def calculate_torsions(
        self,
        molecule: stk.Molecule,
        absolute: bool,  # noqa: FBT001
        as_search_string: bool = False,  # noqa: FBT001, FBT002
    ) -> dict[str, list[float]]:
        """Calculate the value of target torsions."""
        if self._target_torsions is None:
            return {}

        torsions = defaultdict(list)
        for target_torsion in self._target_torsions:
            for torsion in find_torsions(
                molecule, len(target_torsion.search_estring)
            ):
                estrings = tuple([i.__class__.__name__ for i in torsion.atoms])
                if estrings not in (
                    target_torsion.search_estring,
                    tuple(reversed(target_torsion.search_estring)),
                ):
                    continue

                # Check if you want the search string as key, or only the
                # measured atoms.
                if as_search_string:
                    torsion_type_option1 = "_".join(estrings)
                    torsion_type_option2 = "_".join(reversed(estrings))
                else:
                    torsion_type_option1 = "_".join(
                        tuple(
                            estrings[i]
                            for i in target_torsion.measured_atom_ids
                        )
                    )
                    torsion_type_option2 = "_".join(
                        tuple(
                            estrings[i]
                            for i in reversed(target_torsion.measured_atom_ids)
                        )
                    )

                if torsion_type_option1 in torsions:
                    key_string = torsion_type_option1
                    new_ids = tuple(
                        torsion.atom_ids[i]
                        for i in target_torsion.measured_atom_ids
                    )
                elif torsion_type_option2 in torsions:
                    key_string = torsion_type_option2
                    new_ids = tuple(
                        torsion.atom_ids[i]
                        for i in reversed(target_torsion.measured_atom_ids)
                    )
                else:
                    key_string = torsion_type_option1
                    new_ids = tuple(
                        torsion.atom_ids[i]
                        for i in target_torsion.measured_atom_ids
                    )

                torsion_value = get_dihedral(
                    pt1=next(iter(molecule.get_atomic_positions(new_ids[0]))),
                    pt2=next(iter(molecule.get_atomic_positions(new_ids[1]))),
                    pt3=next(iter(molecule.get_atomic_positions(new_ids[2]))),
                    pt4=next(iter(molecule.get_atomic_positions(new_ids[3]))),
                )

                if absolute:
                    torsion_value = abs(torsion_value)

                torsions[key_string].append(torsion_value)

        return torsions

    def calculate_radius_gyration(self, molecule: stk.Molecule) -> float:
        """Calculate the radius of gyration.

        Uses `stko..molecule_analysis.GeometryAnalyser`
        """
        return self._stko_analyser.get_radius_gyration(molecule)

    def calculate_max_diameter(self, molecule: stk.Molecule) -> float:
        """Calculate the maximum diameter of the molecule.

        Uses `stko..molecule_analysis.GeometryAnalyser`
        """
        return self._stko_analyser.get_max_diameter(molecule)
