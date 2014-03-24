from __future__ import division, print_function
from nose.tools import *
import dipolearray as da
from itertools import combinations
from numpy import linspace, pi, shape, cross
from numpy.testing import assert_allclose


def setup():
    pass

def teardown():
    pass

def test_spherical_unit_vectors():
    """
    Test A) Does the function accept all predictable inputs?
    Test B) Are the outputs always orthogonal?
    """
    Degrees = pi / 180
    for a, b in combinations(2 * (0, 5 * Degrees, linspace(0, 180, 5) * Degrees), 2):
        r, theta, phi = da.spherical_unit_vectors(a, b)
        assert_allclose(cross(r, theta), phi, err_msg='vectors should be orthogonal')




def test_radial_direction_vector():
    """
    Test A) Does the function accept all predictable inputs?
    """
    Degrees = pi / 180
    for a, b, c in combinations(2 * (0, 5 * Degrees, linspace(0, 180, 5) * Degrees), 3):
        adir = da.radial_direction_vector(a, b, c)

