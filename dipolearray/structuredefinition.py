'''
Definition(s) of dipole arrays
'''

from __future__ import division, print_function
from numpy import array, arccos, arctan2, meshgrid, zeros, cos, sin, shape, dot

def IncidentPhaseArray(n0, R, k, verbose=0,**kwargs):
    '''
    Returns phase addition term due to incident wave
    '''
    p0 = kwargs.get('intersection', array([0,0,0])) #planes cross at origin
        
    return k*(dot(R,n0)+dot(p0,n0))

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
