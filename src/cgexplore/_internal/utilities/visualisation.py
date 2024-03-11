# Distributed under the terms of the MIT License.

"""Module for pymol visualisation.

Author: Andrew Tarzia

"""

import pathlib
import subprocess as sp

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import stk


class Pymol:
    """Pymol visualiser."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        file_prefix: str,
        pymol_path: pathlib.Path,
        settings: dict | None = None,
    ) -> None:
        """Initialize Pymol visualiser."""
        self._output_dir = output_dir
        self._file_prefix = file_prefix
        self._pymol = pymol_path
        if settings is None:
            self._settings = self._default_settings()
        else:
            for setting in self._default_settings():
                sett = self._default_settings()[setting]
                if setting not in settings:
                    settings[setting] = sett
            self._settings = settings

    def _default_settings(self) -> dict:
        return {
            "grid_mode": 1,
            "stick_rad": 0.7,
            "vdw": 0.8,
            "rayx": 2400,
            "rayy": 2400,
            "zoom_string": "zoom",
            "zoom_scale": 2,
            "orient": True,
        }

    def _get_zoom_string(self, structure_files: list, zoom_scale: int) -> str:
        max_max_diam = 0.0
        for fi in structure_files:
            max_diam = stk.BuildingBlock.init_from_file(
                path=str(fi),
            ).get_maximum_diameter()
            max_max_diam = max((max_diam, max_max_diam))
        return f"zoom center, {max_max_diam/zoom_scale}"

    def _write_pymol_script(
        self,
        structure_files: list,
        structure_colours: list | None,
        pml_file: pathlib.Path,
        orient_atoms: str | None,
    ) -> None:
        if self._settings["zoom_string"] == "custom":
            zoom_string = self._get_zoom_string(
                structure_files=structure_files,
                zoom_scale=self._settings["zoom_scale"],
            )
        else:
            zoom_string = self._settings["zoom_string"]

        if structure_colours is None:
            show_colours = ["#000000" for i in structure_files]
        else:
            show_colours = structure_colours

        if self._settings["orient"]:
            if orient_atoms is None:
                orient_string = "orient"
            else:
                orient_string = f"orient (name {orient_atoms}*)"
        else:
            orient_string = ""

        lstring = ""
        cstring = ""
        lnames = []
        for sf, col in zip(structure_files, show_colours, strict=True):
            lstring += f"load {sf}\n"
            lname = str(sf.name).replace(".mol", "")
            lnames.append(lname)
            tcol = col.replace("#", "0x")
            if structure_colours is None:
                cstring += "color orange, (name C*)\n"
            else:
                cstring += f"color {tcol}, {lname}\n"

        string = (
            f"{lstring}\n"
            f"{cstring}\n"
            f"set grid_mode, {self._settings['grid_mode']}\n"
            "as sticks\n"
            f"set stick_radius, {self._settings['stick_rad']}\n"
            "show spheres\n"
            f"alter all,vdw={self._settings['vdw']}\n"
            "rebuild\n"
            f"{orient_string}\n"
            # "zoom center, 25\n"
            f"{zoom_string}\n"
            "bg_color white\n"
            # "set ray_trace_mode, 1\n"
            f"ray {self._settings['rayx']}, {self._settings['rayy']}\n"
            f"png {self._output_dir / self._file_prefix}.png\n"
            "quit\n"
        )

        with open(pml_file, "w") as f:
            f.write(string)

    def visualise(
        self,
        structure_files: list,
        structure_colours: list | None = None,
        orient_atoms: str | None = None,
    ) -> None:
        """Run pymol to visualise a molecule."""
        pml_file = self._output_dir / f"{self._file_prefix}.pml"
        self._write_pymol_script(
            structure_files=structure_files,
            structure_colours=structure_colours,
            pml_file=pml_file,
            orient_atoms=orient_atoms,
        )
        _ = sp.run(
            [f"{self._pymol}", "-c", "-q", f"{pml_file}"],  # noqa: S603
            stdin=sp.PIPE,
            capture_output=True,
            check=True,
        )

    def _write_host_guest_pymol_script(  # noqa: C901, PLR0913, PLR0912
        self,
        structure_files: list[pathlib.Path],
        host_atoms: list[str],
        host_transperancy: float,
        radii: dict[str, float],
        epsilons: dict[str, float],
        epsilon_min: float,
        epsilon_max: float,
        epsilon_palette: str,
        structure_colours: list | None,
        pml_file: pathlib.Path,
        orient_atoms: str | None,
    ) -> None:
        if self._settings["zoom_string"] == "custom":
            zoom_string = self._get_zoom_string(
                structure_files=structure_files,
                zoom_scale=self._settings["zoom_scale"],
            )
        else:
            zoom_string = self._settings["zoom_string"]

        if structure_colours is None:
            show_colours = ["#000000" for i in structure_files]
        else:
            show_colours = structure_colours

        if self._settings["orient"]:
            if orient_atoms is None:
                orient_string = "orient"
            else:
                orient_string = f"orient (name {orient_atoms}*)"
        else:
            orient_string = ""

        lstring = ""
        cstring = ""
        lnames = []
        for sf, col in zip(structure_files, show_colours, strict=True):
            lstring += f"load {sf}\n"
            lname = str(sf.name).replace(".mol", "")
            lnames.append(lname)
            tcol = col.replace("#", "0x")
            if structure_colours is None:
                cstring += "color orange, (name C*)\n"
            else:
                cstring += f"color {tcol}, {lname}\n"

        selection_string = ""
        for i, element in enumerate(host_atoms):
            if i == 0:
                selection_string += f"select cage, name {element}"
            else:
                selection_string += f" | name {element}"

        radii_string = ""
        for element in radii:
            r = 0 if element in host_atoms else radii[element] / 2
            radii_string += f"alter (name {element}),vdw={r}\n"
        transperancy_string = (
            f"set sphere_transparency, {host_transperancy}, cage"
        )

        epsilon_string = ""
        for element in epsilons:
            b = epsilons[element]
            if element in host_atoms:
                continue
            epsilon_string += f"alter (name {element}),b={b}\n"
        spectrum_string = (
            f"spectrum b, {epsilon_palette}, not cage, {epsilon_min}, "
            f"{epsilon_max}\n"
        )

        string = (
            f"{lstring}\n"
            f"{cstring}\n"
            f"set grid_mode, {self._settings['grid_mode']}\n"
            f"{selection_string}\n"
            "as spheres, (not cage)\n"
            f"{radii_string}\n"
            f"{epsilon_string}\n"
            "rebuild\n"
            f"{orient_string}\n"
            f"{transperancy_string}\n"
            f"{spectrum_string}\n"
            f"{zoom_string}\n"
            "show lines\n"
            "bg_color white\n"
            f"ray {self._settings['rayx']}, {self._settings['rayy']}\n"
            f"png {self._output_dir / self._file_prefix}.png\n"
            "quit\n"
        )

        with open(pml_file, "w") as f:
            f.write(string)

    def visualise_host_guest(  # noqa: PLR0913
        self,
        structure_file: pathlib.Path,
        host_atoms: list[str],
        host_transperancy: float,
        radii: dict[str, float],
        epsilons: dict[str, float],
        epsilon_min: float,
        epsilon_max: float,
        epsilon_palette: str,
        structure_colour: str | None = None,
        orient_atoms: str | None = None,
    ) -> None:
        """Run pymol to visualise a molecule."""
        pml_file = self._output_dir / f"{self._file_prefix}.pml"
        self._write_host_guest_pymol_script(
            structure_files=[structure_file]
            if structure_file is not None
            else None,
            structure_colours=[structure_colour]
            if structure_colour is not None
            else None,
            host_atoms=host_atoms,
            host_transperancy=host_transperancy,
            radii=radii,
            epsilons=epsilons,
            pml_file=pml_file,
            orient_atoms=orient_atoms,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            epsilon_palette=epsilon_palette,
        )
        _ = sp.run(
            [f"{self._pymol}", "-c", "-q", f"{pml_file}"],  # noqa: S603
            stdin=sp.PIPE,
            capture_output=True,
            check=True,
        )


def add_text_to_ax(x: float, y: float, ax: plt.Axes, text: str) -> None:
    """Add a string to an axis."""
    ax.text(x=x, y=y, s=text, fontsize=16, transform=ax.transAxes)


def add_structure_to_ax(ax: plt.Axes, png_file: pathlib.Path) -> None:
    """Add an image to an axis."""
    img = mpimg.imread(png_file)
    ax.imshow(img)
