Forcefields
===========

A package of defining the forcefields used in `CGExplore`.
All of these forcefields interface with `OpenMM`.

.. toctree::
  :maxdepth: 1

  Forcefields module <_autosummary/cgexplore.forcefields>


There is a lot of flexibility in how this code is used, however, `OpenMM`
requires a :class:`cgexplore.forcefields.ForcedSystem`.

.. note::

   Actually use :class:`cgexplore.forcefields.AssignedSystem` or
   :class:`cgexplore.forcefields.MartiniSystem`.



Libraries
---------

These classes allow the user to automatically define a series of forcefields
that vary some parameters in a systematic way in a brute-force combinatorial
way.

.. toctree::
  :maxdepth: 1

  ForceFieldLibrary <_autosummary/cgexplore.forcefields.ForceFieldLibrary>
  MartiniForceFieldLibrary <_autosummary/cgexplore.forcefields.MartiniForceFieldLibrary>


Forcefields
-----------

Classes defining the forcefield before it is assigned to any molecule. This
provides an interface for the user to set the target terms of the forcefield
that can then be assigned to any molecule.

.. toctree::
  :maxdepth: 1

  ForceField <_autosummary/cgexplore.forcefields.ForceField>
  MartiniForceField <_autosummary/cgexplore.forcefields.MartiniForceField>


Forced systems
--------------

Here, the user creates the object that interfaces between a forcefield and an
`OpenMM` simulation.

.. toctree::
  :maxdepth: 1

  ForcedSystem <_autosummary/cgexplore.forcefields.ForcedSystem>
  AssignedSystem <_autosummary/cgexplore.forcefields.AssignedSystem>
  MartiniSystem <_autosummary/cgexplore.forcefields.MartiniSystem>


Custom forces
-------------

.. toctree::
  :maxdepth: 1

  cosine_periodic_angle_force <_autosummary/cgexplore.forcefields.cosine_periodic_angle_force>
  custom_excluded_volume_force <_autosummary/cgexplore.forcefields.custom_excluded_volume_force>


Utilities
---------

.. toctree::
  :maxdepth: 1

  angle_between <_autosummary/cgexplore.forcefields.angle_between>
  unit_vector <_autosummary/cgexplore.forcefields.unit_vector>
  MartiniTopology <_autosummary/cgexplore.forcefields.MartiniTopology>
  get_martini_mass_by_type <_autosummary/cgexplore.forcefields.get_martini_mass_by_type>


.. toctree::
  :maxdepth: 1

