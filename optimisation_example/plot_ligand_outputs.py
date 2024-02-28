"""Script showing how ligand optimisation can work."""

import itertools as it
import logging
import pathlib
from collections import abc, defaultdict
from dataclasses import dataclass

import cgexplore
import matplotlib.pyplot as plt
import molellipsize as mes
import numpy as np
import openmm
import spindry as spd
import stk
from scipy.spatial.distance import cdist


def pymol_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )


def shape_path() -> pathlib.Path:
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )


def colours() -> dict[str, str]:
    """Colours map to topologies."""
    return {
        "2P3": "#1f77b4",
        "4P6": "#ff7f0e",
        "4P62": "#2ca02c",
        "6P9": "#d62728",
        "8P12": "#17becf",
    }


def visualise_structures(
    file_prefix: str,
    best_name: str,
    liga_name: str,
    summary_string: str,
    ligand_string: str,
    image_dir: pathlib.Path,
    structure_dir: pathlib.Path,
) -> None:
    """Take structures and make an image of them."""
    best_file = structure_dir / f"{best_name}_optc.mol"
    liga_file = structure_dir / f"{liga_name}_unopt.mol"
    # Make the images with pymol.
    cgexplore.utilities.Pymol(
        output_dir=image_dir,
        file_prefix=f"{file_prefix}_compl",
        settings={
            "grid_mode": 0,
            "rayx": 1000,
            "rayy": 1000,
            "stick_rad": 0.3,
            "vdw": 0,
            "zoom_string": "custom",
            "orient": False,
        },
        pymol_path=pymol_path(),
    ).visualise([best_file], orient_atoms=None)
    cgexplore.utilities.Pymol(
        output_dir=image_dir,
        file_prefix=f"{file_prefix}_guest",
        settings={
            "grid_mode": 0,
            "rayx": 1000,
            "rayy": 1000,
            "stick_rad": 0.3,
            "vdw": 0,
            "zoom_string": "custom",
            "orient": False,
        },
        pymol_path=pymol_path(),
    ).visualise([liga_file], orient_atoms=None)

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    cgexplore.utilities.add_structure_to_ax(
        ax=axs[0],
        png_file=image_dir / f"{file_prefix}_guest.png",
    )
    cgexplore.utilities.add_text_to_ax(
        x=0.4,
        y=0,
        ax=axs[0],
        text=ligand_string,
    )
    cgexplore.utilities.add_structure_to_ax(
        ax=axs[1],
        png_file=image_dir / f"{file_prefix}_compl.png",
    )
    cgexplore.utilities.add_text_to_ax(
        x=0.4,
        y=0,
        ax=axs[1],
        text=summary_string,
    )

    axs[0].axis("off")
    axs[1].axis("off")

    fig.tight_layout()

    fig.savefig(
        image_dir / f"{file_prefix}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()
    raise SystemExit


def analyse_complex(
    database: cgexplore.utilities.AtomliteDatabase,
    name: str,
    output_dir: pathlib.Path,  # noqa: ARG001
    forcefield: cgexplore.forcefields.ForceField,
    chromosome: cgexplore.systems_optimisation.Chromosome,
) -> None:

    entry = database.get_entry(key=name)
    properties = entry.properties

    database.add_properties(
        key=name,
        property_dict={
            "prefix": name.split("_")[0],
            "chromosome": tuple(int(i) for i in chromosome.name),
        },
    )

    if "forcefield_dict" not in properties:
        # This is matched to the existing analysis code. I recommend
        # generalising in the future.
        ff_targets = forcefield.get_targets()
        s_dict = {}
        e_dict = {}

        for at in ff_targets["nonbondeds"]:

            s_dict[at.bead_class] = at.sigma.value_in_unit(
                openmm.unit.angstrom
            )
            e_dict[at.bead_class] = at.epsilon.value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

        forcefield_dict = {
            "ff_id": forcefield.get_identifier(),
            "ff_prefix": forcefield.get_prefix(),
            "s_dict": s_dict,
            "e_dict": e_dict,
        }
        database.add_properties(
            key=name,
            property_dict={"forcefield_dict": forcefield_dict},
        )


def fitness_calculator(
    chromosome: cgexplore.systems_optimisation.Chromosome,
    chromosome_generator: cgexplore.systems_optimisation.ChromosomeGenerator,  # noqa: ARG001
    database: cgexplore.utilities.AtomliteDatabase,
    calculation_output: pathlib.Path,  # noqa: ARG001
    structure_output: pathlib.Path,  # noqa: ARG001
) -> float:

    name = f"{chromosome.prefix}_{chromosome.get_string()}"

    entry = database.get_entry(name)

    energy = entry.properties["energy_decomposition"]["potential"]
    centroid_distance = entry.properties["centroid_distance"]
    fitness = np.exp(-0.01 * energy) + (1 / centroid_distance)

    database.add_properties(
        key=name,
        property_dict={"fitness": fitness},
    )

    return fitness


class CgAtom(spd.Atom):
    """A new spindry atom with epsilon."""

    def __init__(
        self,
        id: int,  # noqa: A002
        element_string: str,
        epsilon: float,
        sigma: float,
    ) -> None:
        """Initialize CgAtom."""
        super().__init__(id, element_string)
        self._epsilon = epsilon
        self._sigma = sigma

    def get_epsilon(self) -> float:
        """Get the epsilon value."""
        return self._epsilon

    def get_sigma(self) -> float:
        """Get the sigma value."""
        return self._sigma


class CgPotential(spd.Potential):
    """A Cg non-bonded potential function."""

    def _nonbond_potential(
        self,
        distance: np.ndarray,
        sigmas: np.ndarray,
        epsilons: np.ndarray,
    ) -> np.ndarray:
        """Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """
        return epsilons * (
            (sigmas / distance) ** 12 - (sigmas / distance) ** 6
        )

    def _combine_sigma(self, radii1: float, radii2: float) -> float:
        """Combine radii using Lorentz-Berthelot rules."""
        len1 = len(radii1)
        len2 = len(radii2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = (radii1[i] + radii2[j]) / 2

        return mixed

    def _combine_epsilon(self, e1: float, e2: float) -> float:
        """Combine epsilon using Lorentz-Berthelot rules."""
        len1 = len(e1)
        len2 = len(e2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = np.sqrt(e1[i] * e2[j])

        return mixed

    def _compute_nonbonded_potential(
        self,
        position_matrices: np.ndarray,
        radii: np.ndarray,
        epsilons: np.ndarray,
    ) -> float:

        nonbonded_potential = 0
        for pos_mat_pair, radii_pair, epsilon_pair in zip(
            it.combinations(position_matrices, 2),
            it.combinations(radii, 2),
            it.combinations(epsilons, 2),
            strict=False,
        ):

            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            sigmas = self._combine_sigma(radii_pair[0], radii_pair[1])
            epsilons = self._combine_epsilon(epsilon_pair[0], epsilon_pair[1])
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigmas=sigmas.flatten(),
                    epsilons=epsilons.flatten(),
                )
            )

        return nonbonded_potential

    def compute_potential(self, supramolecule: spd.SupraMolecule) -> float:
        """Compure the potential of the molecule."""
        component_position_matrices = (
            i.get_position_matrix() for i in supramolecule.get_components()
        )
        component_radii = (
            tuple(j.get_sigma() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        component_epsilon = (
            tuple(j.get_epsilon() for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        return self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
            epsilons=component_epsilon,
        )


class Laundrette:
    """Class to run rigid-body docking."""

    def __init__(
        self,
        num_dockings: int,
        naming_prefix: str,
        output_dir: pathlib.Path,
        forcefield: cgexplore.forcefields.ForceField,
        seed: int,
    ) -> None:
        """Initialise Laundrette."""
        self._num_dockings = num_dockings
        self._naming_prefix = naming_prefix
        self._output_dir = output_dir
        self._potential = CgPotential()
        self._forcefield = forcefield
        self._rng = np.random.default_rng(seed=seed)

    def _get_supramolecule(
        self,
        hgcomplex: stk.ConstructedMolecule,
    ) -> spd.Potential:
        nonbonded_targets = self._forcefield.get_targets()["nonbondeds"]

        epsilons = []
        sigmas = []
        for atom in hgcomplex.get_atoms():
            atom_estring = atom.__class__.__name__
            cgbead = (
                self._forcefield.get_bead_library().get_cgbead_from_element(
                    atom_estring
                )
            )
            for target_term in nonbonded_targets:
                if target_term.bead_class != cgbead.bead_class:
                    continue
                epsilons.append(
                    target_term.epsilon.value_in_unit(
                        openmm.unit.kilojoules_per_mole
                    )
                )
                sigmas.append(
                    target_term.sigma.value_in_unit(openmm.unit.angstrom)
                )

        return spd.SupraMolecule(
            atoms=(
                CgAtom(
                    id=atom.get_id(),
                    element_string=atom.__class__.__name__,
                    epsilon=epsilons[atom.get_id()],
                    sigma=sigmas[atom.get_id()],
                )
                for atom in hgcomplex.get_atoms()
            ),
            bonds=(
                spd.Bond(
                    id=i,
                    atom_ids=(
                        bond.get_atom1().get_id(),
                        bond.get_atom2().get_id(),
                    ),
                )
                for i, bond in enumerate(hgcomplex.get_bonds())
            ),
            position_matrix=hgcomplex.get_position_matrix(),
        )

    def run_dockings(
        self,
        host_bb: stk.BuildingBlock,
        guest_bb: stk.BuildingBlock,
    ) -> abc.Iterable[cgexplore.molecular.SpindryConformer]:
        """Run the docking algorithm."""
        for docking_id in range(self._num_dockings):
            logging.info(f"docking run: {docking_id+1}")

            guest = stk.host_guest.Guest(
                building_block=guest_bb,
                start_vector=guest_bb.get_direction(),
                end_vector=self._rng.random((1, 3))[0],
                # Change the displacement of the guest.
                displacement=self._rng.random((1, 3))[0],
            )

            hgcomplex = stk.ConstructedMolecule(
                topology_graph=stk.host_guest.Complex(
                    host=stk.BuildingBlock.init_from_molecule(host_bb),
                    guests=guest,
                ),
            )
            supramolecule = self._get_supramolecule(hgcomplex=hgcomplex)

            cg = spd.Spinner(
                step_size=1.0,
                rotation_step_size=2.0,
                num_conformers=200,
                max_attempts=500,
                potential_function=self._potential,
                beta=1.0,
                random_seed=None,
            )
            cid = 1
            for supraconformer in cg.get_conformers(supramolecule):

                yield cgexplore.molecular.SpindryConformer(
                    supramolecule=supraconformer,
                    conformer_id=cid,
                    source=docking_id,
                    energy_decomposition={
                        "potential": supraconformer.get_potential()
                    },
                )
                cid += 1


def calculate_min_atom_distance(supramolecule: spd.SupraMolecule) -> float:
    component_position_matrices = (
        i.get_position_matrix() for i in supramolecule.get_components()
    )

    min_distance = 1e24
    for pos_mat_pair in it.combinations(component_position_matrices, 2):
        pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
        min_distance = min([min_distance, min(pair_dists.flatten())])

    return min_distance


def structure_calculator(
    chromosome: cgexplore.systems_optimisation.Chromosome,
    database: cgexplore.utilities.AtomliteDatabase,
    calculation_output: pathlib.Path,
    structure_output: pathlib.Path,
    host_structure: stk.BuildingBlock,
) -> None:

    name = f"{chromosome.prefix}_{chromosome.get_string()}"
    (bb,) = chromosome.get_building_blocks()
    # Select forcefield by chromosome.
    forcefield = chromosome.get_forcefield()

    # Optimise with some procedure.
    opt_file = structure_output / f"{name}_optc.mol"
    conformers = {}
    if not opt_file.exists():
        laundry = Laundrette(
            num_dockings=10,
            naming_prefix=name,
            output_dir=calculation_output,
            forcefield=forcefield,
            seed=100,
        )
        min_energy_id = "none"
        min_energy = 1e24
        for conformer in laundry.run_dockings(
            host_bb=host_structure,
            guest_bb=bb,
        ):
            id_ = f"{conformer.source}_{conformer.conformer_id}"
            conformers[id_] = conformer
            potential = conformer.energy_decomposition["potential"]
            if potential < min_energy:
                min_energy = conformer.energy_decomposition["potential"]
                min_energy_id = id_

        min_energy_conformer = conformers[min_energy_id]

        # Add to atomlite database.
        database.add_molecule(
            molecule=min_energy_conformer.to_stk_molecule(),
            key=name,
        )
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": (
                    min_energy_conformer.energy_decomposition
                ),
                "source": min_energy_id,
                "optimised": True,
            },
        )

        # Do some analysis while you have the supramolecule.
        comps = list(min_energy_conformer.supramolecule.get_components())

        database.add_properties(
            key=name,
            property_dict={
                "centroid_distance": (
                    np.linalg.norm(
                        comps[0].get_centroid() - comps[1].get_centroid()
                    )
                ),
                "min_hg_distance": calculate_min_atom_distance(
                    min_energy_conformer.supramolecule
                ),
            },
        )
        min_energy_conformer.to_stk_molecule().write(opt_file)

    if "host_name" not in database.get_entry(key=name).properties:
        cid = chromosome.prefix.replace("opt", "")
        host_name = f"host{cid}"
        host_properties = database.get_entry(key=host_name).properties

        mes_mol = mes.Molecule(bb.to_rdkit_mol(), conformers=[0])
        conf_ellipsoids = mes_mol.get_ellipsoids(
            vdwscale=0.45,
            boxmargin=4.0,
            spacing=0.25,
        )

        database.add_properties(
            key=name,
            property_dict={
                "host_name": host_name,
                "host_pore": host_properties["host_pore"],
                "host_size": host_properties["host_size"],
                "host_shape": host_properties["host_shape"],
                "guest_diameters": conf_ellipsoids[0][1],
                "guest_ratios": mes_mol.get_inertial_ratios(),
            },
        )

    # Analyse cage.
    analyse_complex(
        name=name,
        output_dir=calculation_output,
        forcefield=forcefield,
        database=database,
        chromosome=chromosome,
    )


def progress_plot(
    generations: list,
    output: pathlib.Path,
    num_generations: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fitnesses = [
        generation.calculate_fitness_values() for generation in generations
    ]

    ax.plot(
        [max(i) for i in fitnesses],
        c="#F9A03F",
        label="max",
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="k",
    )
    ax.plot(
        [np.mean(i) for i in fitnesses],
        c="#086788",
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="k",
        label="mean",
    )
    ax.plot(
        [min(i) for i in fitnesses],
        c="#7A8B99",
        label="min",
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("generation", fontsize=16)
    ax.set_ylabel("fitness", fontsize=16)
    ax.set_xlim(0, num_generations)
    ax.set_xticks(range(0, num_generations + 1, 5))
    ax.set_ylim(0, None)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


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


@dataclass
class HostGeneration(cgexplore.systems_optimisation.Generation):
    """Define the chromosomes in a single generation."""

    def calculate_fitness_values(self) -> list[float]:
        """Calculate the fitness of all chromosomes."""
        return [
            self.fitness_calculator(
                chromosome=i,
                chromosome_generator=self.chromosome_generator,
                database=self.database,
                calculation_output=self.calculation_output,
                structure_output=self.structure_output,
            )
            for i in self.chromosomes
        ]

    def run_structures(self, host_structure: stk.BuildingBlock) -> None:
        """Run the production and analyse of all chromosomes."""
        length = len(self.chromosomes)
        for i, chromosome in enumerate(self.chromosomes):
            logging.info(f"building {chromosome} ({i+1} of {length})")
            self.structure_calculator(
                chromosome=chromosome,
                database=self.database,
                calculation_output=self.calculation_output,
                structure_output=self.structure_output,
                host_structure=host_structure,
            )

    def select_best(
        self,
        selection_size: int,
    ) -> abc.Iterable[cgexplore.systems_optimisation.Chromosome]:
        """Select the best in the generation by fitness."""
        temp = [
            (
                i,
                self.fitness_calculator(
                    chromosome=i,
                    chromosome_generator=self.chromosome_generator,
                    database=self.database,
                    calculation_output=self.calculation_output,
                    structure_output=self.structure_output,
                ),
            )
            for i in self.chromosomes
        ]
        best_indices = tuple(
            sorted(range(len(temp)), key=lambda i: temp[i][1], reverse=True)
        )[:selection_size]

        return [self.chromosomes[i] for i in best_indices]


def add_scatter(ax: plt.Axes, x: str, y: str, datas: dict) -> None:
    length = len(datas["fitness"])
    fit_thresh = 10
    high_fit_c = "#F9A03F"
    high_fit_s = 80
    high_fit_m = "o"
    low_fit_c = "#7A8B99"
    low_fit_s = 30
    low_fit_m = "o"

    ax.scatter(
        [
            datas[x][i]
            for i in range(length)
            if datas["fitness"][i] < fit_thresh
        ],
        [
            datas[y][i]
            for i in range(length)
            if datas["fitness"][i] < fit_thresh
        ],
        c=low_fit_c,
        edgecolor="none",
        s=low_fit_s,
        marker=low_fit_m,
        alpha=1.0,
    )
    ax.scatter(
        [
            datas[x][i]
            for i in range(length)
            if datas["fitness"][i] > fit_thresh
        ],
        [
            datas[y][i]
            for i in range(length)
            if datas["fitness"][i] > fit_thresh
        ],
        c=high_fit_c,
        edgecolor="k",
        s=high_fit_s,
        marker=high_fit_m,
        alpha=1.0,
    )


def main() -> None:

    wd = pathlib.Path("/home/atarzia/workingspace/cage_optimisation_tests")
    struct_output = wd / "ligand_structures"
    cgexplore.utilities.check_directory(struct_output)
    calc_dir = wd / "ligand_calculations"
    cgexplore.utilities.check_directory(calc_dir)
    data_dir = wd / "ligand_data"
    cgexplore.utilities.check_directory(data_dir)
    figure_dir = wd / "ligand_figures"
    cgexplore.utilities.check_directory(figure_dir)
    all_dir = figure_dir / "all"
    cgexplore.utilities.check_directory(all_dir)
    best_dir = figure_dir / "best"
    cgexplore.utilities.check_directory(best_dir)

    database = cgexplore.utilities.AtomliteDatabase(data_dir / "test.db")

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))
    (ax, ax1, ax2), (ax3, ax4, ax5) = axs

    datas = {f"host{i}": defaultdict(list) for i in range(6)}
    host_shapes = {
        "host0": "OC-6",
        "host1": "OC-6",
        "host2": "OC-6",
        "host3": "OC-6",
        "host4": "OC-6",
        "host5": "OC-6",
    }
    plotted = 0
    for entry in database.get_entries():
        properties = entry.properties
        if properties["prefix"] == "host":
            continue

        if "host_name" not in properties:
            continue
        host_name = properties["host_name"]

        datas[host_name]["chromosomes"].append(properties["chromosome"])
        datas[host_name]["fitness"].append(properties["fitness"])
        datas[host_name]["host_shape"].append(
            properties["host_shape"][host_shapes[host_name]]
        )
        datas[host_name]["host_size"].append(properties["host_size"])
        datas[host_name]["host_pore"].append(
            properties["host_pore"]["min_distance"]
        )
        datas[host_name]["guest_diameters"].append(
            properties["guest_diameters"]
        )
        datas[host_name]["centroid_distance"].append(
            properties["centroid_distance"]
        )
        datas[host_name]["potential"].append(
            properties["energy_decomposition"]["potential"]
        )

        plotted += 1

    xwidth = 2
    for host_name in datas:
        xdata = datas[host_name]["fitness"]
        xbins = np.arange(0 - xwidth, 50 + xwidth, xwidth)
        ax.hist(
            x=xdata,
            bins=xbins,
            density=True,
            # bottom=fn,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            # facecolor=leg_info()[ami]["c"],
            alpha=1.0,
            # color=leg_info()[ami]['c'],
            # color='white',
            edgecolor="k",
            label=f"{host_name}: {len(xdata)}",
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("fitness", fontsize=16)
    ax.set_ylabel("frequency", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_yscale("log")

    xys = {i: [] for i in datas if i != "host0"}
    for chromosome in datas["host0"]["chromosomes"]:
        fitness_x = next(
            datas["host0"]["fitness"][i]
            for i in range(len(datas["host0"]["fitness"]))
            if datas["host0"]["chromosomes"][i] == chromosome
        )
        for host_name in xys:
            if host_name == "host0":
                continue
            fitness_y = [
                datas[host_name]["fitness"][i]
                for i in range(len(datas[host_name]["fitness"]))
                if datas[host_name]["chromosomes"][i] == chromosome
            ]
            if len(fitness_y) == 1:
                xys[host_name].append((fitness_x, fitness_y[0]))

    for host_name in xys:
        xs = [i[0] for i in xys[host_name]]
        ys = [i[1] for i in xys[host_name]]
        ax1.scatter(
            xs,
            ys,
            alpha=1.0,
            s=40,
            edgecolor="none",
            label=host_name,
        )
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("host 0 fitness", fontsize=16)
    ax1.set_ylabel("host X fitness", fontsize=16)
    ax1.legend(fontsize=16)
    ax1.plot((0, 50), (0, 50), c="gray", ls="--")

    xwidth = 5
    for host_name in datas:
        xdata = datas[host_name]["potential"]
        xbins = np.arange(-140 - xwidth, 10 + xwidth, xwidth)
        ax2.hist(
            x=xdata,
            bins=xbins,
            density=True,
            # bottom=fn,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            # facecolor=leg_info()[ami]["c"],
            alpha=1.0,
            # color=leg_info()[ami]['c'],
            # color='white',
            edgecolor="k",
        )
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_xlabel("potential", fontsize=16)
    ax2.set_ylabel("frequency", fontsize=16)

    xwidth = 0.2
    for host_name in datas:
        xdata = datas[host_name]["centroid_distance"]
        xbins = np.arange(0 - xwidth, 4 + xwidth, xwidth)
        ax5.hist(
            x=xdata,
            bins=xbins,
            density=True,
            # bottom=fn,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            # facecolor=leg_info()[ami]["c"],
            alpha=1.0,
            # color=leg_info()[ami]['c'],
            # color='white',
            edgecolor="k",
        )
    ax5.tick_params(axis="both", which="major", labelsize=16)
    ax5.set_xlabel("centroid_distance", fontsize=16)
    ax5.set_ylabel("frequency", fontsize=16)

    for host_name in datas:
        xdata = datas[host_name]["host_size"][0]
        ydata = datas[host_name]["host_pore"][0]

        ax3.scatter(
            xdata,
            ydata,
            alpha=1.0,
            s=120,
            edgecolor="k",
            label=host_name,
        )
    ax3.tick_params(axis="both", which="major", labelsize=16)
    ax3.set_xlabel("host max size", fontsize=16)
    ax3.set_ylabel("host pore", fontsize=16)
    ax3.legend(fontsize=16)

    fitness_threshold = 5
    for host_name in datas:
        length = len(datas[host_name]["fitness"])
        xdata = [
            datas[host_name]["guest_diameters"][i][0]
            for i in range(length)
            if datas[host_name]["fitness"][i] > fitness_threshold
        ]
        ydata = [
            datas[host_name]["guest_diameters"][i][1]
            for i in range(length)
            if datas[host_name]["fitness"][i] > fitness_threshold
        ]
        # ydata = [
        #     datas[host_name]["fitness"][i]
        #     for i in range(length)
        #     if datas[host_name]["fitness"][i] > fitness_threshold
        # ]

        ax4.scatter(
            xdata,
            ydata,
            alpha=1.0,
            s=80,
            edgecolor="none",
            label=host_name,
        )
    ax4.tick_params(axis="both", which="major", labelsize=16)
    ax4.set_xlabel("guest min diameter", fontsize=16)
    ax4.set_ylabel("guest mid diameter", fontsize=16)
    ax4.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "host_data.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
