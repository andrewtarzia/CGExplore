Executables
===========

A series of helpful executables included in the installation.


Databasing
----------

Getting the energies of systems in the database
...............................................

.. code-block:: bash

  get-energies path/to/database.db name min_energy max_energy

energy is always `energy_per_bb`

name can be all

min and max can be ignored

Deleting a property from all entries in a database
..................................................

.. code-block:: bash

  delete-property path/to/database.db STRING.