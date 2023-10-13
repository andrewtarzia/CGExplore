import pathlib

from cgexplore.utilities import check_directory


def project_dir():
    path = pathlib.Path("/home/atarzia/workingspace/GeomCGstk/")
    check_directory(path)
    return path


def cages():
    path = project_dir() / pathlib.Path("cages/")
    check_directory(path)
    return path


def structures():
    path = cages() / pathlib.Path("structures/")
    check_directory(path)
    return path


def figures():
    path = cages() / pathlib.Path("figures/")
    check_directory(path)
    return path


def ligands():
    path = cages() / pathlib.Path("ligands/")
    check_directory(path)
    return path


def calculations():
    path = cages() / pathlib.Path("calculations/")
    check_directory(path)
    return path


def outputdata():
    path = cages() / pathlib.Path("outputdata/")
    check_directory(path)
    return path


def pymol_path():
    return pathlib.Path(
        "/home/atarzia/software/pymol-open-source-build/bin/pymol"
    )


def shape_path():
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )
