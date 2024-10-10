import filecmp
import pathlib

import numpy as np
from rdkit.Chem import AllChem

import cgexplore

from .case_data import CaseData
from .utilities import is_equivalent_molecule


def test_ensemble(ensemble: CaseData) -> None:
    """Test :class:`.Ensemble`.

    Parameters:

        ensemble:
            The molecule.

    """
    output_dir = pathlib.Path(__file__).resolve().parent / "output"
    print(ensemble)
    test = cgexplore.molecular.Ensemble(
        base_molecule=ensemble.molecule,
        base_mol_path=output_dir / f"{ensemble.name}_base.mol",
        conformer_xyz=output_dir / f"{ensemble.name}_ensemble.xyz",
        data_json=output_dir / f"{ensemble.name}_ensemble.json",
        overwrite=True,
    )

    # Write a new ensemble.
    generator = np.random.default_rng(109)
    num_confs = 10
    rdkit_molecule = ensemble.molecule.to_rdkit_mol()
    rdkit_molecule.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = 1000
    cids = AllChem.EmbedMultipleConfs(
        mol=rdkit_molecule,
        numConfs=num_confs,
        params=params,
    )
    built_conformers = []
    for cid in cids:
        pos_mat = rdkit_molecule.GetConformer(id=cid).GetPositions()
        conformer = cgexplore.molecular.Conformer(
            molecule=ensemble.molecule.with_position_matrix(pos_mat),
            energy_decomposition={"total energy": (generator.random(), "")},
            conformer_id=cid,
            source="etkdg",
        )
        built_conformers.append(conformer)
        test.add_conformer(conformer, source="etkdg")

    print(test)
    assert test.get_num_conformers() == num_confs
    # Test I/O unchanged and compare to known.
    test.write_conformers_to_file()

    known_ensemble = cgexplore.molecular.Ensemble(
        base_molecule=ensemble.molecule,
        base_mol_path=output_dir / f"known_{ensemble.name}_base.mol",
        conformer_xyz=output_dir / f"known_{ensemble.name}_ensemble.xyz",
        data_json=output_dir / f"known_{ensemble.name}_ensemble.json",
        overwrite=False,
    )
    assert filecmp.cmp(
        output_dir / f"known_{ensemble.name}_base.mol",
        output_dir / f"{ensemble.name}_base.mol",
    )
    assert filecmp.cmp(
        output_dir / f"known_{ensemble.name}_ensemble.xyz",
        output_dir / f"{ensemble.name}_ensemble.xyz",
    )
    assert filecmp.cmp(
        output_dir / f"known_{ensemble.name}_ensemble.json",
        output_dir / f"{ensemble.name}_ensemble.json",
    )

    is_equivalent_molecule(
        known_ensemble.get_base_molecule(), ensemble.molecule
    )
    is_equivalent_molecule(test.get_base_molecule(), ensemble.molecule)
    assert test.get_molecule_num_atoms() == ensemble.molecule.get_num_atoms()

    for cid in range(num_confs):
        test_conformer = test.get_conformer(cid)
        known_conformer = known_ensemble.get_conformer(cid)
        built_conformer = built_conformers[cid]
        assert test_conformer.conformer_id == cid
        assert known_conformer.conformer_id == cid
        assert built_conformer.conformer_id == cid
        assert (
            test_conformer.energy_decomposition["total energy"][0]
            == known_conformer.energy_decomposition["total energy"][0]
        )
        assert (
            test_conformer.energy_decomposition["total energy"][0]
            == built_conformer.energy_decomposition["total energy"][0]
        )
        is_equivalent_molecule(
            test_conformer.molecule, known_conformer.molecule
        )
        is_equivalent_molecule(
            test_conformer.molecule, built_conformer.molecule
        )

    assert known_ensemble.load_data() == test.load_data()
    assert known_ensemble.load_trajectory() == test.load_trajectory()

    is_equivalent_molecule(
        known_ensemble.get_lowest_e_conformer().molecule,
        test.get_lowest_e_conformer().molecule,
    )

    (output_dir / f"{ensemble.name}_base.mol").unlink()
    (output_dir / f"{ensemble.name}_ensemble.xyz").unlink()
    (output_dir / f"{ensemble.name}_ensemble.json").unlink()
