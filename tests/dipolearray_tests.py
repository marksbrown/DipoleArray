from __future__ import division, print_function
from nose.tools import *
import dipolearray as da
from itertools import combinations
from numpy import linspace, pi, shape


def setup():
    pass

def teardown():
    pass

def test_radial_direction_vector():
    Degrees = pi / 180
    for j, (a, b, c) in enumerate(combinations(2 * (0, 5 * Degrees, linspace(0, 180, 5) * Degrees), 3)):
        adir = da.radial_direction_vector(a, b, c)


def test_radial_direction_vector():
    Degrees = pi / 180
    for j, (a, b, c) in enumerate(combinations(2 * (0, 5 * Degrees, linspace(0, 180, 5) * Degrees), 3)):
        adir = da.radial_direction_vector(a, b, c)

