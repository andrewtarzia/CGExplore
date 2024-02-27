"""Script showing how ligand optimisation can work."""

import itertools as it
import pathlib

import cgexplore
import numpy as np


def pymol_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )


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
            # additional requirement to avoid
            # null summation in idx if
            # chromosome[idx] == 0
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


def main() -> None:

    wd = pathlib.Path("/home/atarzia/workingspace/cage_optimisation_tests")
    struct_output = wd / "ligand_structures"
    cgexplore.utilities.check_directory(struct_output)
    figure_dir = wd / "ligand_figures"
    cgexplore.utilities.check_directory(figure_dir)
    all_dir = figure_dir / "all"
    cgexplore.utilities.check_directory(all_dir)

    xbead = cgexplore.molecular.CgBead(
        element_string="Co",
        bead_class="x",
        bead_type="x",
        coordination=3,
    )
    ybead = cgexplore.molecular.CgBead(
        element_string="Fe",
        bead_class="y",
        bead_type="y",
        coordination=2,
    )

    num_beads = 6
    compositions = [
        i
        for i in it.product(range(num_beads + 1), repeat=num_beads)
        if check_fit(i, num_beads=num_beads, max_shell=6)
    ]
    compositions = sorted(compositions, reverse=True)

    # Visualise all ligand topologies.
    chromosome_gen = cgexplore.systems_optimisation.ChromosomeGenerator(
        prefix="liga",
        present_beads=(xbead, ybead),
        vdw_bond_cutoff=2,
    )
    chromosome_gen.add_gene(
        iteration=(
            cgexplore.molecular.PrecursorGenerator(
                composition=i,
                present_beads=(
                    xbead,
                    ybead,
                    ybead,
                    ybead,
                    ybead,
                    ybead,
                    ybead,
                ),
                binder_beads=(),
                placer_beads=(),
                bead_distance=1.5,
            )
            for i in compositions
        ),
        gene_type="precursor",
    )
    for chromosome in chromosome_gen.yield_chromosomes():
        (bb,) = chromosome.get_building_blocks()
        viz_file = struct_output / f"{chromosome.get_string()}_unopt.mol"
        bb.write(viz_file)
        cgexplore.utilities.Pymol(
            output_dir=all_dir,
            file_prefix=f"liga_{chromosome.get_string()}",
            settings={
                "grid_mode": 0,
                "rayx": 1000,
                "rayy": 1000,
                "stick_rad": 0.3,
                "vdw": 0,
                "zoom_string": "custom",
                "zoom_scale": 1,
                "orient": False,
            },
            pymol_path=pymol_path(),
        ).visualise(
            [viz_file],
            orient_atoms=None,
        )


if __name__ == "__main__":
    main()
