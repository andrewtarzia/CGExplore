"""Define the environment variables."""

import pathlib

from cgexplore.utilities import check_directory


def project_dir() -> pathlib.Path:
    """A directory."""
    path = pathlib.Path("/home/atarzia/workingspace/GeomCGstk/")
    check_directory(path)
    return path


def cages() -> pathlib.Path:
    """A directory."""
    path = project_dir() / pathlib.Path("cages/")
    check_directory(path)
    return path


def structures() -> pathlib.Path:
    """A directory."""
    path = cages() / pathlib.Path("structures/")
    check_directory(path)
    return path


def figures() -> pathlib.Path:
    """A directory."""
    path = cages() / pathlib.Path("figures/")
    check_directory(path)
    return path


def ligands() -> pathlib.Path:
    """A directory."""
    path = cages() / pathlib.Path("ligands/")
    check_directory(path)
    return path


def calculations() -> pathlib.Path:
    """A directory."""
    path = cages() / pathlib.Path("calculations/")
    check_directory(path)
    return path


def outputdata() -> pathlib.Path:
    """A directory."""
    path = cages() / pathlib.Path("outputdata/")
    check_directory(path)
    return path


def pymol_path() -> pathlib.Path:
    """A directory."""
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )


def shape_path() -> pathlib.Path:
    """A directory."""
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )
