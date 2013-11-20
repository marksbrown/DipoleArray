'''
Definition(s) of dipole arrays
'''

from __future__ import division, print_function
from numpy import array, arccos, arctan2, meshgrid, zeros, cos, sin, shape


def IncidentWavePhaseTerms(incidentdirection=array([0, 0, 1])):
    '''
    Converts
    '''

    theta = lambda adir: arccos(adir[..., 2])
    phi = lambda adir: arctan2(adir[..., 1], adir[..., 0])

    return theta(incidentdirection), phi(incidentdirection)


def IncidentPhaseArray():
    pass


def getpositions(N1, N2, lc):
    '''
    Returns cartesian position of an array of points defined by the lattice
    _lc_ with min/max numbers defined by the tuples _N1_ and _N2_
    '''
    u1, u2 = meshgrid(range(*N1), range(*N2))  # grid of integers

    if len(lc) == 4:
        dx, tx, dy, ty = lc
        dvary = 0
        tvary = 0
    elif len(lc) == 6:
        dx, tx, dy, ty, dvary, tvary = lc

    xpos = lambda u, d, t: u * d * cos(t)
    ypos = lambda u, d, t: u * d * sin(t)

    X = xpos(u1, dx, tx) + xpos(u2, dy, ty) + xpos(u1 ** 2, dvary, tvary)
    Y = ypos(u1, dx, tx) + ypos(u2, dy, ty) + ypos(u2 ** 2, dvary, tvary)
    Z = zeros(shape(X))

    return X, Y, Z
