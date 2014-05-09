"""
Metrics and functionality for optimising the reflection metasurface
"""

from numpy import pi, arccos, arctan2

Degrees = pi/180


def proportion_within_segment(farfield, vectors, polar_range=None, azimuthal_range=None, verbose=0):
    """
    Calculates the sum of farfield within the polar range and azimuthal ranges
    """

    theta = arccos(vectors[..., 2])

    if verbose > 0:
        print("minimum theta is {}\nmaximum theta is {}\n".format(min(theta)/Degrees, max(theta)/Degrees))

    condition = (theta >= polar_range[0]) & (theta <= polar_range[1])

    if azimuthal_range is not None:
        phi = arctan2(vectors[..., 1], vectors[..., 0])

        if verbose > 0:
            print("minimum phi is {}\nmaximum phi is {}\n".format(min(phi)/Degrees, max(phi)/Degrees))

        (phi >= azimuthal_range[0]) & (phi <= azimuthal_range[1])
        condition &= (phi >= azimuthal_range[0]) & (phi <= azimuthal_range[1])

    return sum(farfield[condition])/sum(farfield)
