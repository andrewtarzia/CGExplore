# Distributed under the terms of the MIT License.

"""Module for Martini topology.

Contains class in https://github.com/maccallumlab/martini_openmm/tree/master

"""
import contextlib
import logging
import pathlib

from openmm import app, openmm

with contextlib.suppress(ModuleNotFoundError):
    import martini_openmm as martini

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_martini_dir = pathlib.Path(__file__).resolve().parent / "data"


class MartiniTopology:
    """Contains the :class:`MartiniTopFile`."""

    def __init__(self, itp_file: pathlib.Path) -> None:
        """Initialze MartiniTopology."""
        self._itp_file = itp_file
        self._molecule_name = itp_file.name.replace(".itp", "")
        self._top_file = pathlib.Path(str(itp_file).replace(".itp", ".top"))
        self._write_top_file()
        self._topology = martini.MartiniTopFile(
            file=self._top_file,
            periodicBoxVectors=((20.0, 0, 0), (0, 20.0, 0), (0, 0, 20.0)),
            defines=None,
            epsilon_r=15,
        )

    def _write_top_file(self) -> None:
        string = (
            f'#include "{_martini_dir}/martini_v3.0.0.itp"\n'
            # '#include "martini/martini_v3.0.0_ions_v1.itp"\n'
            # '#include "martini/martini_v3.0.0_solvents_v1.itp"\n'
            f'#include "{self._itp_file.name}"\n'
            "\n"
            "[ system ]\n"
            "A single molecule\n"
            "\n"
            "[ molecules ]\n"
            f"{self._molecule_name}    1\n"
            # 'W             8819\n'
            # 'NA             97\n'
            # 'CL             97\n'
        )
        with open(self._top_file, "w") as f:
            f.write(string)

    def get_openmm_topology(self) -> app.topology.Topology:
        """Return OpenMM.Topology object."""
        return self._topology.topology

    def get_openmm_system(self) -> openmm.System:
        """Return OpenMM.System object."""
        return self._topology.create_system()


def get_martini_mass_by_type(bead_type: str) -> float:
    """Get the mass of martini types."""
    bead_size = bead_type[0]
    return {
        "P": 72.0,
        "N": 72.0,
        "C": 72.0,
        "X": 72.0,
        "S": 54.0,
        "T": 36.0,
    }[bead_size]
