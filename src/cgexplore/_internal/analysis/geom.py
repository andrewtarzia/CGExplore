# Distributed under the terms of the MIT License.

"""Module for geometry analysis."""

import typing
from collections import abc, defaultdict

import stk
import stko
from rdkit.Chem import AllChem

from cgexplore._internal.forcefields.forcefield import ForceField
from cgexplore._internal.terms.torsions import TargetTorsion
from cgexplore._internal.terms.utilities import find_torsions


class GeomMeasure:
    """Class to perform geometry calculations.

    Parameters:
        target_torsions:
            An iterable of `TargetTorsion` defining which torsions to
            capture if you run `calculate_torsions`.

    """

    def __init__(
        self,
        target_torsions: abc.Iterable[TargetTorsion] | None = None,
    ) -> None:
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
        """Calculate the minimum distance between beads and centroid.

        Parameters:
            molecule:
                The molecule to analyse.

        Returns:
            The minimum distance.

        """
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
        """Calculate the minimum distance between beads and beads.

        Parameters:
            molecule:
                The molecule to analyse.

        Returns:
            The minimum bead-to-bead distance.

        """
        return self._stko_analyser.get_min_atom_atom_distance(molecule)

    def calculate_bonds(
        self,
        molecule: stk.Molecule,
    ) -> dict[tuple[str, str], list[float]]:
        """Calculate the bond lengths.

        Uses `stko.molecule_analysis.GeometryAnalyser`

        Parameters:
            molecule:
                The molecule to analyse.

        Returns:
            A dictionary of bond lengths organised by element strings.

        """
        return self._stko_analyser.calculate_bonds(molecule)

    def calculate_angles(
        self,
        molecule: stk.Molecule,
    ) -> dict[tuple[str, str, str], list[float]]:
        """Calculate the angle values.

        Uses `stko.molecule_analysis.GeometryAnalyser`

        Parameters:
            molecule:
                The molecule to analyse.

        Returns:
            A dictionary of angle values organised by element strings.

        """
        return self._stko_analyser.calculate_angles(molecule)

    def calculate_torsions(
        self,
        molecule: stk.Molecule,
        absolute: bool,
        as_search_string: bool = False,
    ) -> dict[str, list[float]]:
        """Calculate the value of target torsions.

        Parameters:
            molecule:
                The molecule to analyse.

            absolute:
                `True` to get the abs(torsion) from 0 degrees to 180 degrees.

            as_search_string:
                Changes the key of the returned dictionary to be defined by
                the search string used in `target_torsions` or as the found
                atom element strings.

        Returns:
            A dictionary of torsion values organised by element strings.

        """
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

                torsion_value = stko.calculate_dihedral(
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

        Uses `stko.molecule_analysis.GeometryAnalyser`

        Parameters:
            molecule:
                The molecule to analyse.

        """
        return self._stko_analyser.get_radius_gyration(molecule)

    def calculate_max_diameter(self, molecule: stk.Molecule) -> float:
        """Calculate the maximum diameter of the molecule.

        Uses `stko.molecule_analysis.GeometryAnalyser`

        Parameters:
            molecule:
                The molecule to analyse.

        """
        return self._stko_analyser.get_max_diameter(molecule)

    @classmethod
    def from_forcefield(
        cls,
        forcefield: ForceField,
    ) -> typing.Self:
        """Get the values in terms of forcefield terms."""
        ff_targets = forcefield.get_targets()

        return cls(target_torsions=ff_targets["torsions"])
