"""
Tests for Dipole Array
"""
from __future__ import division, print_function
from nose.tools import *
import dipolearray as da
from itertools import combinations
from numpy import linspace, pi, cross, meshgrid
from numpy.testing import assert_allclose
from random import randint


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


def test_structure_factor():
    """
    Test A) does the analytical expression match the brute force for various periodic structure?
    """

    nm = 1e-9
    Degrees = pi/180

    N1 = (1, randint(2, 200))
    N2 = (1, randint(2, 200))
    k = (2*pi)/(randint(300, 1000)*nm)

    steptheta = randint(100, 200)
    stepphi = randint(20, 400)

    print("wavenumber is {:2.0f} with array of{}x{}, with {} and {} steps in theta and phi respectively".format(k, N1[1], N2[1], steptheta, stepphi))

    SqL = [100*nm,0*Degrees,100*nm,90*Degrees]##Square Lattice
    square_array = da.metasurface(N1, N2, SqL)

    n0 = da.radial_direction_vector(30*Degrees, 90*Degrees)
    plane_waves = da.light(k, n0, steptheta = steptheta, stepphi = stepphi)

    Fa = square_array.structure_factor('z', plane_waves, dist='analytical', verbose=2)
    Fs = square_array.structure_factor('z', plane_waves, dist='sum', verbose=2)

    assert_allclose(Fa, Fs, err_msg = "structure factors do not match from both methods!")


def test_radial_direction_vector():
    """
    Test A) Does the function accept all predictable inputs?
    """
    Degrees = pi / 180
    for a, b, c in combinations(2 * (0, 5 * Degrees, linspace(0, 180, 5) * Degrees), 3):
        adir = da.radial_direction_vector(a, b, c)


