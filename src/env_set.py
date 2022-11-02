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


def mnl2n():
    path = project_dir() / pathlib.Path("mnl2n/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def mnl2n_figures():
    path = mnl2n() / pathlib.Path("figures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def mnl2n_structures():
    path = mnl2n() / pathlib.Path("structures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def guest():
    path = project_dir() / pathlib.Path("guest/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def guest_figures():
    path = guest() / pathlib.Path("figures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def guest_structures():
    path = guest() / pathlib.Path("structures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def mnl2n_calculations():
    path = mnl2n() / pathlib.Path("calculations/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def fourplussix():
    path = project_dir() / pathlib.Path("fourplussix/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def fourplussix_figures():
    path = fourplussix() / pathlib.Path("figures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def fourplussix_structures():
    path = fourplussix() / pathlib.Path("structures/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def fourplussix_optimisation():
    path = fourplussix() / pathlib.Path("optimisation/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def fourplussix_hg_optimisation():
    path = fourplussix() / pathlib.Path("hg_optimisation/")
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def fourplussix_calculations():
    path = fourplussix() / pathlib.Path("calculations/")
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
