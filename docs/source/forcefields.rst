Forcefields
===========

A package of defining the forcefields used in `CGExplore`.
All of these forcefields interface with `OpenMM`.
There is a lot of flexibility in how this code is used, however, `OpenMM`
requires a :class:`cgexplore.forcefields.AssignedSystem`.

.. note::

   Actually use :class:`cgexplore.forcefields.AssignedSystem` or
   :class:`cgexplore.forcefields.MartiniSystem`.

.. note::

   Developing libraries of forcefields for iteration can be much simpler
   in the Systems Optimisation module, using the `definer dictionary`
   interface and chromosomes (see the `optimisation_example`).

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

  AssignedSystem <_autosummary/cgexplore.forcefields.AssignedSystem>
  MartiniSystem <_autosummary/cgexplore.forcefields.MartiniSystem>


Utilities
---------

.. toctree::
  :maxdepth: 1

  MartiniTopology <_autosummary/cgexplore.forcefields.MartiniTopology>
  get_martini_mass_by_type <_autosummary/cgexplore.forcefields.get_martini_mass_by_type>


.. toctree::
  :maxdepth: 1
