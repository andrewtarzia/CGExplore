"""molecular package."""

from cgexplore._internal.molecular.beads import (
    BeadLibrary,
    CgBead,
    periodic_table,
    string_to_atom_number,
)
from cgexplore._internal.molecular.conformer import Conformer, SpindryConformer
from cgexplore._internal.molecular.ensembles import Ensemble, Timestep
from cgexplore._internal.molecular.molecule_construction import (
    FourC0Arm,
    FourC1Arm,
    LinearPrecursor,
    Precursor,
    SquarePrecursor,
    ThreeC0Arm,
    ThreeC1Arm,
    ThreeC2Arm,
    TrianglePrecursor,
    TwoC0Arm,
    TwoC1Arm,
    TwoC2Arm,
    TwoC3Arm,
)
from cgexplore._internal.molecular.precursor_generator import (
    PrecursorGenerator,
    check_fit,
)

__all__ = [
    "Precursor",
    "PrecursorGenerator",
    "check_fit",
    "TrianglePrecursor",
    "SquarePrecursor",
    "LinearPrecursor",
    "FourC0Arm",
    "FourC1Arm",
    "ThreeC0Arm",
    "ThreeC1Arm",
    "ThreeC2Arm",
    "TwoC0Arm",
    "TwoC1Arm",
    "TwoC2Arm",
    "TwoC3Arm",
    "Conformer",
    "SpindryConformer",
    "Timestep",
    "Ensemble",
    "CgBead",
    "BeadLibrary",
    "periodic_table",
    "string_to_atom_number",
]
