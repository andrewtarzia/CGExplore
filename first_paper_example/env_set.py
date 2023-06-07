import pathlib

from .utilities import check_directory


def project_dir():
    path = pathlib.Path("/home/atarzia/workingspace/GeomCGstk/")
    check_directory(path)
    return path


def sphere():
    path = project_dir() / pathlib.Path("spheres/")
    check_directory(path)
    return path


def cubism():
    path = project_dir() / pathlib.Path("cubism/")
    check_directory(path)
    return path


def mnl2n():
    path = project_dir() / pathlib.Path("mnl2n/")
    check_directory(path)
    return path


def unsymm():
    path = project_dir() / pathlib.Path("unsymm/")
    check_directory(path)
    return path


def guests():
    path = project_dir() / pathlib.Path("guest/")
    check_directory(path)
    return path


def fourplussix():
    path = project_dir() / pathlib.Path("fourplussix/")
    check_directory(path)
    return path


def cages():
    path = project_dir() / pathlib.Path("cages/")
    check_directory(path)
    return path


def gulp_path():
    return pathlib.Path("/home/atarzia/software/gulp-6.1/Src/gulp")


def pymol_path():
    return pathlib.Path(
        "/home/atarzia/pymol-open-source-build/bin/pymol"
    )


def shape_path():
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )
