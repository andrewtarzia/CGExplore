import os
import pathlib

from cgexplore.errors import ForceFieldUnitError


def test_system_xml_writer(molecule):
    """Test methods toward :meth:`.ForceField.get_xml_string`.

    Parameters:

        molecule:
            The molecule.

    Returns:
        None : :class:`NoneType`

    """
    try:
        syst_xml_file = pathlib.Path(
            os.path.dirname(os.path.realpath(__file__))
        ) / (
            f"{molecule.name}_{molecule.forcefield.get_prefix()}_"
            f"{molecule.name}_syst.xml"
        )
        topo_xml_file = pathlib.Path(
            os.path.dirname(os.path.realpath(__file__))
        ) / (
            f"{molecule.name}_{molecule.forcefield.get_prefix()}_"
            f"{molecule.name}_topo.xml"
        )
        saved_syst_xml_file = pathlib.Path(
            os.path.dirname(os.path.realpath(__file__))
        ) / (
            f"{molecule.name}_{molecule.forcefield.get_prefix()}_"
            f"{molecule.name}_syst_saved.xml"
        )

        assigned_system = molecule.forcefield.assign_terms(
            molecule=molecule.molecule,
            output_dir=pathlib.Path(
                os.path.dirname(os.path.realpath(__file__))
            ),
            name=molecule.name,
        )
        assigned_system.get_openmm_system()
        with open(syst_xml_file) as f:
            xml_string = f.read()
        with open(saved_syst_xml_file) as f:
            test_xml_string = f.read()
        print(xml_string)
        assert xml_string == test_xml_string
        os.system(f"rm {syst_xml_file}")
        os.system(f"rm {topo_xml_file}")
    except ForceFieldUnitError:
        assert molecule.num_forcefields == 0
