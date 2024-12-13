# Distributed under the terms of the MIT License.

"""Module for containing exceptions."""


class ForceFieldUnitError(Exception):
    """Error found in units of forcefield term."""


class ForceFieldUnavailableError(Exception):
    """Error found assigning forcefield term."""
