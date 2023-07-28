from cgexplore.generation_utilities import (
    collect_custom_torsion,
    target_torsions,
)
import cgexplore


def test_present_torsions(molecule):
    """
    Test methods toward :meth:`.Optimizer._yield_custom_torsions`.

    Parameters:

        molecule:
            The molecule.

    Returns:

        None : :class:`NoneType`

    """

    targets = target_torsions(
        bead_set=molecule.bead_set,
        custom_torsion_option=molecule.custom_torsion_definition,
    )
    print(targets)
    assert targets == molecule.custom_torsion_set

    custom_torsion_set = collect_custom_torsion(
        custom_torsion_options={"ton": molecule.custom_torsion_definition},
        custom_torsion="ton",
        bead_set=molecule.bead_set,
    )
    print(custom_torsion_set)
    assert custom_torsion_set == molecule.custom_torsion_set

    optimizer = cgexplore.CGOptimizer(
        param_pool=molecule.bead_set,
        custom_torsion_set=custom_torsion_set,
        bonds=False,
        angles=False,
        torsions=False,
        vdw=False,
    )

    count = 0
    for torsion_info in optimizer._yield_custom_torsions(molecule.molecule):
        print(torsion_info)
        assert torsion_info in molecule.present_torsions
        count += 1

    assert count == len(molecule.present_torsions)
