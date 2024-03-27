import cgexplore
import pytest

from .case_data import CaseData

abead = cgexplore.molecular.CgBead(
    element_string="Ag",
    bead_class="a",
    bead_type="a",
    coordination=3,
)
bbead = cgexplore.molecular.CgBead(
    element_string="Ba",
    bead_class="b",
    bead_type="b",
    coordination=2,
)
cbead = cgexplore.molecular.CgBead(
    element_string="C",
    bead_class="c",
    bead_type="c",
    coordination=2,
)
dbead = cgexplore.molecular.CgBead(
    element_string="O",
    bead_class="o",
    bead_type="o",
    coordination=2,
)

chromo_it = cgexplore.systems_optimisation.ChromosomeGenerator(
    prefix="test",
    present_beads=(abead, bbead, cbead, dbead),
    vdw_bond_cutoff=2,
)
chromo_it.add_gene(
    iteration=("2P3", "4P6", "4P62", "6P9", "8P12"),
    gene_type="topology",
)
chromo_it.add_gene(
    iteration=("Tc2", "Tc3"),
    gene_type="precursor",
)
chromo_it.add_gene(
    iteration=("Tc1",),
    gene_type="precursor",
)
chromo_it.add_gene(
    iteration=((0, 1), (0, 2)),
    gene_type="vertex_alignment",
)
definer_dict = {
    # Bonds.
    "ao": ("bond", 1.5, 1e5),
    "bc": ("bond", 1.5, 1e5),
    "co": ("bond", 1.0, 1e5),
    "cc": ("bond", 1.0, 1e5),
    "oo": ("bond", 1.0, 1e5),
    # Angles.
    "ccb": ("angle", 180.0, 1e2),
    "ooc": ("angle", 180.0, 1e2),
    "occ": ("angle", 180.0, 1e2),
    "ccc": ("angle", 180.0, 1e2),
    "oco": ("angle", 180.0, 1e2),
    "aoc": ("angle", 180.0, 1e2),
    "aoo": ("angle", 180.0, 1e2),
    "bco": ("angle", (90, 170), 1e2),
    "cbc": ("angle", 180.0, 1e2),
    "oao": ("angle", (50, 70), 1e2),
    # Torsions.
    "ocbco": ("tors", "0134", 180, 50, 1),
    # Nonbondeds.
    "a": ("nb", 10.0, 1.0),
    "b": ("nb", 10.0, 1.0),
    "c": ("nb", 10.0, 1.0),
    "o": ("nb", 10.0, 1.0),
}
chromo_it.add_forcefield_dict(definer_dict=definer_dict)

chromo_it2 = cgexplore.systems_optimisation.ChromosomeGenerator(
    prefix="test2",
    present_beads=(),
    vdw_bond_cutoff=2,
)
chromo_it2.add_gene(
    iteration=("2P3",),
    gene_type="topology",
)
chromo_it2.add_gene(
    iteration=("Tc1",),
    gene_type="precursor",
)
chromo_it2.add_gene(
    iteration=("Tc1",),
    gene_type="precursor",
)
definer_dict = {
    # Bonds.
    "ao": ("bond", (1.0, 1.5), (1e5, 2)),
    "bc": ("bond", 1.5, (1e5, 2)),
    "co": ("bond", (1.0, 0.5), 1e5),
}
chromo_it2.add_forcefield_dict(definer_dict=definer_dict)


@pytest.fixture(
    params=(
        lambda name: CaseData(
            chromosome_generator=chromo_it,
            known_map={
                0: {0: "2P3", 1: "4P6", 2: "4P62", 3: "6P9", 4: "8P12"},
                1: {0: "Tc2", 1: "Tc3"},
                2: {0: "Tc1"},
                3: {0: (0, 1), 1: (0, 2)},
                4: {0: ("bco", 90, 1), 1: ("bco", 170, 1)},
                5: {0: ("oao", 50, 1), 1: ("oao", 70, 1)},
            },
            known_types={
                0: "topology",
                1: "precursor",
                2: "precursor",
                3: "vertex_alignment",
                4: "term",
                5: "term",
            },
            num=80,
            term_ids=(4, 5),
            topo_ids=(0,),
            va_ids=(3,),
            prec_ids=(1, 2),
            merged_size=28,
            pop1_fitness=[4, 4, 1, 6, 4, 5, 8, 2, 5, 6],
            pop1_best=("410111", "210111"),
            pop1_worst=("000100", "010001"),
            pop1_elite_fitness=8,
            name=name,
        ),
        lambda name: CaseData(
            chromosome_generator=chromo_it2,
            known_map={
                0: {0: "2P3"},
                1: {0: "Tc1"},
                2: {0: "Tc1"},
                3: {0: ("ao", 1.0, 1), 1: ("ao", 1.5, 1)},
                4: {0: ("ao", 100000.0, 2), 1: ("ao", 2, 2)},
                5: {0: ("bc", 100000.0, 2), 1: ("bc", 2, 2)},
                6: {0: ("co", 1.0, 1), 1: ("co", 0.5, 1)},
            },
            known_types={
                0: "topology",
                1: "precursor",
                2: "precursor",
                3: "term",
                4: "term",
                5: "term",
                6: "term",
            },
            num=16,
            term_ids=(3, 4, 5, 6),
            topo_ids=(0,),
            va_ids=(),
            prec_ids=(1, 2),
            merged_size=16,
            pop1_fitness=[3, 1, 1, 2, 4, 2, 2, 3, 3, 2],
            pop1_best=("0001111", "0001101"),
            pop1_worst=("0000100", "0000100"),
            pop1_elite_fitness=4,
            name=name,
        ),
    )
)
def chromosome_generator(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
