# Distributed under the terms of the MIT License.

"""Module for containing exceptions.

Author: Andrew Tarzia

"""


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class ForceFieldUnitError(Exception):
    """Error found in units of forcefield term."""


class ForceFieldUnavailableError(Exception):
    """Error found assigning forcefield term."""
