"""Definition of conversion utilities."""

import logging
import pathlib

import numpy as np
import stk
import stko

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def extract_ensemble(molecule: stk.Molecule, crest_run: pathlib.Path) -> dict:
    """Extract and save an ensemble from a crest run."""
    ensemble_dir = crest_run / "ensemble"
    num_atoms = molecule.get_num_atoms()
    ensemble = {}
    ensemble_dir.mkdir(exist_ok=True, parents=True)

    # Calculate geometrical properties.
    conformer_file = crest_run / "crest_conformers.xyz"
    with conformer_file.open("r") as f:
        linex = f.readlines()

    split_line = linex[0].rstrip()
    for i, conformers in enumerate("".join(linex).split(split_line)):
        lines = conformers.split("\n")

        if len(lines) != num_atoms + 3:
            continue

        energy = lines[1]
        position_matrix = []
        for line in lines[2:]:
            splits = line.rstrip().split()
            if len(splits) != 4:  # noqa: PLR2004
                continue
            symb, x, y, z = splits
            x = float(x)
            y = float(y)
            z = float(z)

            position_matrix.append(np.array((x, y, z)))

        conf_molecule = molecule.with_position_matrix(
            np.array(position_matrix)
        )

        calc = stko.molecule_analysis.DitopicThreeSiteAnalyser()

        adjacent_centroids = calc.get_adjacent_centroids(conf_molecule)
        adjacent_distance = np.linalg.norm(
            adjacent_centroids[0] - adjacent_centroids[1]
        )

        ensemble[i] = {
            "energy": float(energy),
            "molecule": conf_molecule,
            "binder_angles": calc.get_binder_angles(conf_molecule),
            "binder_binder_angle": calc.get_binder_binder_angle(conf_molecule),
            "binder_distance": calc.get_binder_distance(conf_molecule),
            "binder_adjacent_torsion": calc.get_binder_adjacent_torsion(
                conf_molecule
            ),
            "adjacent_distance": adjacent_distance,
            "binder_com_angle": calc.get_binder_centroid_angle(conf_molecule),
        }

        ensemble[i]["molecule"].write(ensemble_dir / f"conf_{i}.mol")
    return ensemble


def cgx_optimisation_sequence(
    cage: stk.Molecule,
    name: str,
    calculation_dir: pathlib.Path,
    gulp_path: pathlib.Path,
) -> stk.Molecule:
    """Cage optimisation sequence.

    TODO: This is a placeholder for a more general atomistic cage sequence.

    """
    gulp1_output = calculation_dir / f"{name}_gulp1.mol"
    gulp2_output = calculation_dir / f"{name}_gulp2.mol"

    if not gulp1_output.exists():
        output_dir = calculation_dir / f"{name}_gulp1"

        logging.info("    UFF4MOF optimisation 1 of %s", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_path,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=True,
        )
        gulp_opt.assign_FF(cage)
        gulp1_mol = gulp_opt.optimize(mol=cage)
        gulp1_mol.write(gulp1_output)
    else:
        logging.info("    loading %s", gulp1_output)
        gulp1_mol = cage.with_structure_from_file(gulp1_output)

    if not gulp2_output.exists():
        output_dir = calculation_dir / f"{name}_gulp2"
        logging.info("    UFF4MOF optimisation 2 of %s", name)
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=gulp_path,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=False,
        )
        gulp_opt.assign_FF(gulp1_mol)
        gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
        gulp2_mol.write(gulp2_output)
    else:
        logging.info("    loading %s", gulp2_output)
        gulp2_mol = cage.with_structure_from_file(gulp2_output)

    return cage.with_structure_from_file(gulp2_output)
