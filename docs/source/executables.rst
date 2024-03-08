Executables
===========

A series of helpful executables included in the installation.


Databasing
----------

Getting the values of systems in the database
.............................................

.. code-block:: bash

  get-values path/to/database.db --name name --path key1 key2..

This allows the listing of the values in a database for entries (all if
``name`` not set). Each ``key`` in ``path`` is a step into the dictionary
``entry.properties``.

If you do not know what is in the database, you can run

.. code-block:: bash

  get-values path/to/database.db --paths

This will show all paths in ``entry.properties``. **Not all entries will have
all properties.**

Rows of this output can be copied as ``key1``, ``key2``... by splitting on `.`.


Getting the energies of systems in the database
...............................................

.. code-block:: bash

  get-energies path/to/database.db --name name --min min_energy --max max_energy

Here, the energy is always ``energy_per_bb``. ``name`` is optional, it will go
through the whole database if not set.


Deleting a property from all entries in a database
..................................................

.. code-block:: bash

  delete-property path/to/database.db --name name --path path

In this case you must use the JSON path of format: ``"\$key1.key2"``. This is
explained further in the ``atomlite``
`documentation <https://atomlite.readthedocs.io/en/latest/index.html#examples-valid-property-paths>`_
