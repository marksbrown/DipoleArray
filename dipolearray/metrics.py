"""
Metrics and functionality for optimising the reflection metasurface
"""

from numpy import pi, arccos, arctan2, sum, cos, shape, real

Degrees = pi/180


def proportion_within_segment(outgoing_vectors, farfield, polar_range=None):
    """
    Calculates the sum of farfield within the polar range
    """

    min_theta = min(polar_range)
    max_theta = max(polar_range)

    min_condition = outgoing_vectors[..., 2] < cos(min_theta)
    max_condition = outgoing_vectors[..., 2] > cos(max_theta)
    condition = min_condition & max_condition

    return real(sum(farfield[condition])/sum(farfield))
