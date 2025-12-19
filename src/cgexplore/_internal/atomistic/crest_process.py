"""Definition of a crest processing of building blocks."""

import logging
import os
import pathlib
import shutil
import subprocess as sp
import uuid
from collections import abc

import bbprep
import stk
import stko
from rdkit import RDLogger

from .utilities import extract_ditopic_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


class Crest(stko.Optimizer):
    """Run CREST conformer search algorithm.

    TODO: Move into stko.
    """

    def __init__(  # noqa: PLR0913
        self,
        crest_path: pathlib.Path,
        xtb_path: pathlib.Path,
        gfn_method: str = "2",
        output_dir: pathlib.Path | str | None = None,
        num_cores: int = 4,
        charge: int = 0,
        electronic_temperature: float = 300,
        solvent_model: str = "gbsa",
        solvent: str | None = None,
        solvent_grid: str = "normal",
        num_unpaired_electrons: int = 0,
        unlimited_memory: bool = False,
        additional_commands: abc.Sequence[str] = (),
    ) -> None:
        """Initialise calculator."""
        if solvent is not None:
            solvent = solvent.lower()
            if gfn_method in ("gfnff", "0"):
                msg = "XTB: No solvent valid for version", f" {gfn_method!r}."
                raise stko.InvalidSolventError(msg)
            if not stko.is_valid_xtb_solvent(
                gfn_version=int(gfn_method),
                solvent_model=solvent_model,
                solvent=solvent,
            ):
                msg = (
                    f"XTB: Solvent {solvent!r} and model {solvent_model!r}",
                    f" is invalid for version {gfn_method!r}.",
                )
                raise stko.InvalidSolventError(msg)

        self._check_path(crest_path)
        self._check_path(xtb_path)
        self._crest_path = crest_path
        self._xtb_path = xtb_path
        self._gfn_method = (
            f"--gfn{gfn_method}" if gfn_method not in ("gfnff",) else "--gfnff"
        )
        self._output_dir = (
            None if output_dir is None else pathlib.Path(output_dir)
        )
        self._additional_commands = additional_commands
        self._num_cores = str(num_cores)
        self._electronic_temperature = str(electronic_temperature)
        self._solvent_model = solvent_model
        self._solvent = solvent
        self._solvent_grid = solvent_grid
        self._charge = str(charge)
        self._num_unpaired_electrons = str(num_unpaired_electrons)
        self._unlimited_memory = unlimited_memory

    def _check_path(self, path: pathlib.Path | str) -> None:
        path = pathlib.Path(path)
        if not path.exists():
            msg = f"XTB or CREST not found at {path}"
            raise RuntimeError(msg)

    def _write_detailed_control(self) -> None:
        string = f"$gbsa\n   gbsagrid={self._solvent_grid}"

        with pathlib.Path("det_control.in").open("w") as f:
            f.write(string)

    def _is_complete(
        self,
        output_file: pathlib.Path | str,
        output_xyzs: abc.Iterable[pathlib.Path],
    ) -> bool:
        output_file = pathlib.Path(output_file)
        if not output_file.exists():
            # No simulation has been run.
            msg = "CREST run did not start"
            raise stko.NotStartedError(msg)

        for xyz in output_xyzs:
            if not xyz.exists():
                msg = f"CREST run did not complete, {xyz} is not present!"
                raise stko.NotCompletedError(msg)

        return True

    def _run_crest(self, xyz: str, out_file: pathlib.Path | str) -> None:
        out_file = pathlib.Path(out_file)

        # Modify the memory limit.
        memory = "ulimit -s unlimited ;" if self._unlimited_memory else ""

        if self._solvent is not None:
            solvent = f"--{self._solvent_model} {self._solvent}"
        else:
            solvent = ""

        additions = " ".join(self._additional_commands)

        cmd = (
            f"{memory} {self._crest_path} {xyz} "
            f"-xnam {self._xtb_path} "
            f"{solvent} -chrg {self._charge} "
            f"--etemp {self._electronic_temperature}"
            f"-uhf {self._num_unpaired_electrons} "
            f"{self._gfn_method} "
            f"-T {self._num_cores} {additions} -I det_control.in"
        )

        with out_file.open("w") as f:
            # Note that sp.call will hold the program until completion
            # of the calculation.
            sp.call(  # noqa: S602
                cmd,
                stdin=sp.PIPE,
                stdout=f,
                stderr=sp.PIPE,
                # Shell is required to run complex arguments.
                shell=True,
            )

    def optimize(self, molecule: stk.Molecule) -> stk.Molecule:  # type:ignore[override]
        """Optimise a solute-solvent pair."""
        if self._output_dir is None:
            output_dir = pathlib.Path(str(uuid.uuid4().int)).resolve()
        else:
            output_dir = self._output_dir.resolve()

        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir()
        init_dir = pathlib.Path.cwd()
        os.chdir(output_dir)

        try:
            xyz = "input.xyz"
            molecule.write(xyz)
            self._write_detailed_control()

            out_file = "crest.output"

            self._run_crest(xyz=xyz, out_file=out_file)

            # Check if the optimization is complete.
            output_xyzs = [
                pathlib.Path("crest_best.xyz"),
                pathlib.Path("crest_conformers.xyz"),
                pathlib.Path("crest_rotamers.xyz"),
            ]

            opt_complete = self._is_complete(out_file, output_xyzs)

            molecule = molecule.with_structure_from_file(
                pathlib.Path("crest_best.xyz")
            )

        finally:
            os.chdir(init_dir)

        if not opt_complete:
            msg = f"CREST run is incomplete for {molecule}."
            logger.warning(msg)

        return molecule


def run_conformer_analysis(  # noqa: PLR0913
    ligand_name: str,
    molecule: stk.BuildingBlock,
    ligand_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    functional_group_factories: abc.Sequence[stk.FunctionalGroupFactory],
    crest_path: pathlib.Path,
    xtb_path: pathlib.Path,
) -> dict:
    """Analyse conformers.

    TODO: Move into stko.
    """
    opt_file = ligand_dir / f"{ligand_name}_optl.mol"
    crest_run = calculation_dir / f"{ligand_name}_crest"

    molecule = stk.BuildingBlock.init_from_molecule(
        molecule=molecule,
        functional_groups=functional_group_factories,
    )

    # Handle if not ditopic.
    if molecule.get_num_functional_groups() != 2:  # noqa: PLR2004
        molecule = bbprep.FurthestFGs().modify(
            building_block=molecule,
            desired_functional_groups=2,
        )

    if not opt_file.exists():
        # Run calculation.
        optimiser = Crest(
            crest_path=crest_path,
            xtb_path=xtb_path,
            output_dir=crest_run,
            gfn_method="2",
            num_cores=12,
            unlimited_memory=True,
            solvent="dmso",
            solvent_model="alpb",
            solvent_grid="verytight",
            additional_commands=(
                "--optlev extreme",
                # No z matrix sorting.
                "--nosz",
                "--keepdir",
                # Set energy threshold (kcal.mol)
                "--ewin 10",
            ),
            charge=0,
            electronic_temperature=300,
            num_unpaired_electrons=0,
        )

        opt_molecule: stk.Molecule = optimiser.optimize(molecule)
        opt_molecule.write(opt_file)

    return extract_ditopic_ensemble(molecule, crest_run)
