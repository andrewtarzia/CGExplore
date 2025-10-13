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

- :doc:`cosine_periodic_angle_force <_autosummary/cgexplore.forcefields.cosine_periodic_angle_force>`
- :doc:`custom_excluded_volume_force <_autosummary/cgexplore.forcefields.custom_excluded_volume_force>`
- :doc:`custom_lennard_jones_force <_autosummary/cgexplore.forcefields.custom_lennard_jones_force>`



Bonds
-----

- :doc:`TargetBond <_autosummary/cgexplore.terms.TargetBond>`
- :doc:`TargetBondRange <_autosummary/cgexplore.terms.TargetBondRange>`
- :doc:`TargetMartiniBond <_autosummary/cgexplore.terms.TargetMartiniBond>`
- :doc:`MartiniBondRange <_autosummary/cgexplore.terms.MartiniBondRange>`

Angles
------

- :doc:`TargetAngle <_autosummary/cgexplore.terms.TargetAngle>`
- :doc:`TargetAngleRange <_autosummary/cgexplore.terms.TargetAngleRange>`
- :doc:`TargetPyramidAngle <_autosummary/cgexplore.terms.TargetPyramidAngle>`
- :doc:`PyramidAngleRange <_autosummary/cgexplore.terms.PyramidAngleRange>`
- :doc:`CosineAngle <_autosummary/cgexplore.terms.CosineAngle>`
- :doc:`TargetCosineAngle <_autosummary/cgexplore.terms.TargetCosineAngle>`
- :doc:`TargetCosineAngleRange <_autosummary/cgexplore.terms.TargetCosineAngleRange>`
- :doc:`TargetMartiniAngle <_autosummary/cgexplore.terms.TargetMartiniAngle>`
- :doc:`MartiniAngleRange <_autosummary/cgexplore.terms.MartiniAngleRange>`
- :doc:`FoundAngle <_autosummary/cgexplore.terms.FoundAngle>`



Torsions
--------

- :doc:`TargetTorsion <_autosummary/cgexplore.terms.TargetTorsion>`
- :doc:`TargetTorsionRange <_autosummary/cgexplore.terms.TargetTorsionRange>`
- :doc:`TargetMartiniTorsion <_autosummary/cgexplore.terms.TargetMartiniTorsion>`
- :doc:`FoundTorsion <_autosummary/cgexplore.terms.FoundTorsion>`

Nonbonded
---------

Unlike other terms, nonbonded terms are usually set at the bead class level,
not bead type. Although the distinction between the two remains muddy.

- :doc:`TargetNonbonded <_autosummary/cgexplore.terms.TargetNonbonded>`
- :doc:`TargetNonbondedRange <_autosummary/cgexplore.terms.TargetNonbondedRange>`


Assigned classes
----------------

These classes are not often used by the user, but the forcefield assigns them
based on targets used above.

- :doc:`Bond <_autosummary/cgexplore.terms.Bond>`
- :doc:`Angle <_autosummary/cgexplore.terms.Angle>`
- :doc:`Torsion <_autosummary/cgexplore.terms.Torsion>`
- :doc:`Nonbonded <_autosummary/cgexplore.terms.Nonbonded>`

Utilities
---------

- :doc:`find_angles <_autosummary/cgexplore.terms.find_angles>`
- :doc:`find_torsions <_autosummary/cgexplore.terms.find_torsions>`
