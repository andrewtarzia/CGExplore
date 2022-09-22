import pathlib
import os


def project_dir():
    path = pathlib.Path("/home/atarzia/workingspace/GeomCGstk/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def cubism():
    path = project_dir() / pathlib.Path("cubism/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def cubism_figures():
    path = cubism() / pathlib.Path("figures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def cubism_structures():
    path = cubism() / pathlib.Path("structures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def cubism_calculations():
    path = cubism() / pathlib.Path("calculations/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def gulp_path():
    return pathlib.Path("/home/atarzia/software/gulp-6.1/Src/gulp")


def shape_path():
    return pathlib.Path(
        "/home/atarzia/software/shape_2.1_linux_64/"
        "SHAPE_2.1_linux_64/shape_2.1_linux64"
    )
