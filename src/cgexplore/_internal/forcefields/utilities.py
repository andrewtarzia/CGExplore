# Distributed under the terms of the MIT License.

"""Utilities module."""

from openmm import openmm


def custom_excluded_volume_force() -> openmm.CustomNonbondedForce:
    """Define Custom Excluded Volume force."""
    energy_expression = "epsilon*((sigma)/(r))^12;"
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    custom_force = openmm.CustomNonbondedForce(energy_expression)
    custom_force.addPerParticleParameter("sigma")
    custom_force.addPerParticleParameter("epsilon")
    return custom_force


def custom_lennard_jones_force() -> openmm.CustomNonbondedForce:
    """Define Custom Lennard Jones force."""
    energy_expression = "epsilon*(((sigma)/(r))^12 - ((sigma)/(r))^6);"
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    custom_force = openmm.CustomNonbondedForce(energy_expression)
    custom_force.addPerParticleParameter("sigma")
    custom_force.addPerParticleParameter("epsilon")
    return custom_force


def cosine_periodic_angle_force() -> openmm.CustomAngleForce:
    """Define Custom Angle force."""
    energy_expression = "F*C*(1-A*cos(n * theta));"
    energy_expression += "A = b*(min_n);"
    energy_expression += "C = (n^2 * k)/2;"
    energy_expression += "F = (2/(n ^ 2));"
    custom_force = openmm.CustomAngleForce(energy_expression)
    custom_force.addPerAngleParameter("k")
    custom_force.addPerAngleParameter("n")
    custom_force.addPerAngleParameter("b")
    custom_force.addPerAngleParameter("min_n")
    return custom_force
