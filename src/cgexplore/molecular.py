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
    GeneratedPrecursor,
    PrecursorGenerator,
    VaBene,
    VaBeneGenerator,
    check_fit,
)
from cgexplore._internal.molecular.utilities import spd_to_stk

__all__ = [
    "Precursor",
    "GeneratedPrecursor",
    "PrecursorGenerator",
    "check_fit",
    "VaBeneGenerator",
    "VaBene",
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
    "spd_to_stk",
    # "ShapePrecursor",
    # "DiatomShape",
    # "HexagonShape",
    # "TriangleShape",
    # "SquareShape",
    # "StarShape",
    # "TdShape",
    # "CuShape",
    # "OcShape",
    # "V2P3Shape",
    # "V2P4Shape",
    # "V4P62Shape",
    # "V6P9Shape",
    # "V8P16Shape",
    # "V10P20Shape",
    # "V12P24Shape",
    # "V12P30Shape",
    # "V20P30Shape",
    # "V24P48Shape",
]
