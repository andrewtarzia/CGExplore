Terms
=====

A package of the forcefield terms used in `CGExplore`.

As with the discussion of the :class:`cgexplore.forcefields.ForceFieldLibrary`,
all terms have the actual term (e.g., :class:`cgexplore.terms.Bond`) and a
`Target` and `TargerRange` class, which assist the user in defining variable
forcefields for a single term.

.. important::

  **Warning**: The distinction between a bead class and a bead type is
  currently unclear. In most cases, you can use them interchangebly. The idea
  was to have a broader class, with type being a subset, but that does not
  seem useful.

Custom forces
-------------

When assigning target terms using the classes below, you can select the force.
Most are standard OpenMM forces, which can be found in their documentation.
Others are defined here:

.. toctree::
  :maxdepth: 1

  cosine_periodic_angle_force <_autosummary/cgexplore.forcefields.cosine_periodic_angle_force>
  custom_excluded_volume_force <_autosummary/cgexplore.forcefields.custom_excluded_volume_force>
  custom_lennard_jones_force <_autosummary/cgexplore.forcefields.custom_lennard_jones_force>



Bonds
-----

.. toctree::
  :maxdepth: 1

  TargetBond <_autosummary/cgexplore.terms.TargetBond>
  TargetBondRange <_autosummary/cgexplore.terms.TargetBondRange>
  TargetMartiniBond <_autosummary/cgexplore.terms.TargetMartiniBond>
  MartiniBondRange <_autosummary/cgexplore.terms.MartiniBondRange>

Angles
------

.. toctree::
  :maxdepth: 1

  TargetAngle <_autosummary/cgexplore.terms.TargetAngle>
  TargetAngleRange <_autosummary/cgexplore.terms.TargetAngleRange>
  TargetPyramidAngle <_autosummary/cgexplore.terms.TargetPyramidAngle>
  PyramidAngleRange <_autosummary/cgexplore.terms.PyramidAngleRange>
  CosineAngle <_autosummary/cgexplore.terms.CosineAngle>
  TargetCosineAngle <_autosummary/cgexplore.terms.TargetCosineAngle>
  TargetCosineAngleRange <_autosummary/cgexplore.terms.TargetCosineAngleRange>
  TargetMartiniAngle <_autosummary/cgexplore.terms.TargetMartiniAngle>
  MartiniAngleRange <_autosummary/cgexplore.terms.MartiniAngleRange>
  FoundAngle <_autosummary/cgexplore.terms.FoundAngle>



Torsions
--------

.. toctree::
  :maxdepth: 1

  TargetTorsion <_autosummary/cgexplore.terms.TargetTorsion>
  TargetTorsionRange <_autosummary/cgexplore.terms.TargetTorsionRange>
  TargetMartiniTorsion <_autosummary/cgexplore.terms.TargetMartiniTorsion>
  FoundTorsion <_autosummary/cgexplore.terms.FoundTorsion>

Nonbonded
---------

Unlike other terms, nonbonded terms are usually set at the bead class level,
not bead type. Although the distinction between the two remains muddy.

.. toctree::
  :maxdepth: 1

  TargetNonbonded <_autosummary/cgexplore.terms.TargetNonbonded>
  TargetNonbondedRange <_autosummary/cgexplore.terms.TargetNonbondedRange>


Assigned classes
----------------

These classes are not often used by the user, but the forcefield assigns them
based on targets used above.

.. toctree::
  :maxdepth: 1

  Bond <_autosummary/cgexplore.terms.Bond>
  Angle <_autosummary/cgexplore.terms.Angle>
  Torsion <_autosummary/cgexplore.terms.Torsion>
  Nonbonded <_autosummary/cgexplore.terms.Nonbonded>

Utilities
---------

.. toctree::
  :maxdepth: 1

  find_angles <_autosummary/cgexplore.terms.find_angles>
  find_torsions <_autosummary/cgexplore.terms.find_torsions>

