#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for containing exceptions.

Author: Andrew Tarzia

"""


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class ForcefieldUnitError(Exception):
    pass


class ForcefieldUnavailableError(Exception):
    pass
